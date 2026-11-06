"""Thin structured-output wrapper around the Anthropic SDK, plus a scripted fake for tests."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Callable, Optional, Protocol, TypeVar

from pydantic import BaseModel

from .config import Settings
from .models import LLMCallLog

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

Effort = str  # "low" | "medium" | "high" | "xhigh" | "max"


class LLMStalled(Exception):
    """A model call exceeded the hard wall-clock deadline without returning; the attempt was abandoned."""


class LLMRefusal(RuntimeError):
    pass


class LLMConfigError(RuntimeError):
    """A request the API rejected because of account/credential configuration, with a fix hint."""


class LLM(Protocol):
    def structured(self, *, task: str, system: str, user: str, output_model: type[T], effort: Effort = "high",
                   max_tokens: int = 16000, model: str | None = None) -> T: ...

    @property
    def call_log(self) -> list[LLMCallLog]: ...


class AnthropicLLM:
    """Structured generation through ``client.beta.messages.stream`` (streaming is mandatory for large outputs).

    Server-side refusal fallbacks are enabled by default (``RGA_FALLBACKS=0`` disables);
    if the API rejects the fallback parameter the call is retried without it.
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, settings: Settings | None = None, model: str | None = None):
        import anthropic  # imported lazily so tests never need credentials

        self._anthropic = anthropic
        self.settings = settings or Settings()
        self.model = model or self.settings.model
        # Streams can stay silent for a long time while the model reasons over large prompts; the SDK default
        # (10 min) is too short for validation judgements. Mid-stream drops are retried in _request.
        self.client = anthropic.Anthropic(max_retries=4, timeout=anthropic.Timeout(self.settings.llm_timeout, connect=15.0))
        self._use_fallbacks = self.settings.use_fallbacks
        self.transient_retries = 3
        # Hard wall-clock limit per attempt. The SDK timeout only bounds individual socket reads; a call that stalls
        # inside the client (DNS, a wedged connection pool, a half-closed socket) would otherwise block forever.
        self.hard_deadline = float(self.settings.llm_timeout) + 120.0
        self.on_event: Optional[Callable[[str], None]] = None  # progress sink for retry notices (GUI/CLI log)
        self._log: list[LLMCallLog] = []
        self._no_fallback_models: set[str] = set()  # models whose API rejected the fallbacks parameter

    def _emit(self, msg: str) -> None:
        log.warning(msg)
        if self.on_event:
            try:
                self.on_event(msg)
            except Exception:  # noqa: BLE001 - never let a UI sink break a model call
                pass

    def _stream_once(self, kwargs: dict):
        with self.client.beta.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    @property
    def call_log(self) -> list[LLMCallLog]:
        return self._log

    def _request(self, kwargs: dict):
        """Stream the response and return the final parsed message, retrying dropped streams and 5xx/overload errors.

        The SDK only retries the initial request; a read timeout or disconnect in the middle of a stream surfaces here.
        """
        a = self._anthropic
        for attempt in range(self.transient_retries + 1):
            # Each attempt runs in its own thread so a stalled call can be abandoned at the hard deadline.
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-call")
            fut = pool.submit(self._stream_once, kwargs)
            pool.shutdown(wait=False)
            try:
                return fut.result(timeout=self.hard_deadline)
            except FuturesTimeout:
                err: Exception = LLMStalled(f"model call produced nothing for {self.hard_deadline:.0f}s; abandoned")
            except (a.APITimeoutError, a.APIConnectionError) as e:
                err = e
            except a.APIStatusError as e:
                if e.status_code in (429, 529) or e.status_code >= 500:
                    err = e
                else:
                    raise
            if attempt >= self.transient_retries:
                raise err
            delay = 15 * (attempt + 1)
            self._emit(f"Model call failed ({type(err).__name__}: {str(err)[:160]}); retrying in {delay}s "
                       f"(attempt {attempt + 1}/{self.transient_retries})")
            time.sleep(delay)
        raise RuntimeError("unreachable")

    def structured(self, *, task: str, system: str, user: str, output_model: type[T], effort: Effort = "high",
                   max_tokens: int = 16000, model: str | None = None) -> T:
        """One structured call. ``model`` overrides the default model for this call only (e.g. a cheaper screening model).

        Safe to call from several threads at once: the SDK client is thread-safe and this object only appends to its log.
        """
        model = model or self.model
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=output_model,
            output_config={"effort": effort},
        )
        if self._use_fallbacks and model not in self._no_fallback_models:
            kwargs["betas"] = [self.FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        if self.settings.workspace_id:
            kwargs["workspace_id"] = self.settings.workspace_id
        try:
            response = self._request(kwargs)
        except self._anthropic.BadRequestError as e:
            if "workspace" in str(e).lower():
                raise LLMConfigError(
                    "The API key is organisation-wide, so the API needs a workspace id. Either set "
                    "ANTHROPIC_WORKSPACE_ID (Console > Settings > Workspaces > open the workspace > copy its id, "
                    "wrkspc_...) in .env or the GUI credentials panel, or create a new key scoped to a workspace "
                    "(Console > API Keys > Create Key > choose a workspace). Original error: " + e.message
                ) from e
            if "fallbacks" in kwargs and "fallback" in str(e).lower():
                log.warning("API rejected fallbacks parameter for %s; retrying without it (%s)", model, e.message)
                self._no_fallback_models.add(model)
                kwargs.pop("betas", None)
                kwargs.pop("fallbacks", None)
                response = self._request(kwargs)
            else:
                raise
        usage = response.usage
        self._log.append(
            LLMCallLog(
                task=task,
                model=response.model,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            )
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(f"Model declined task {task!r}: {getattr(details, 'explanation', '')}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"Task {task!r} hit max_tokens={max_tokens}; output truncated")
        parsed = response.parsed_output
        if parsed is None:
            raise RuntimeError(f"Task {task!r} returned no parsed output")
        return parsed


def screen_model_for(settings: Settings, llm: LLM) -> str | None:
    """The model to use for title/abstract screening, or None to use the main model.

    ``RGA_SCREEN_MODEL`` (or ``--screen-model``) wins; the value ``main`` disables the cheaper model. Otherwise the
    provider preset's ``fast_model`` is used when it differs from the main model.
    """
    from .providers import PROVIDERS

    chosen = (settings.screen_model or "").strip()
    if chosen.lower() in ("main", "same", "none"):
        return None
    if chosen:
        return chosen
    preset = PROVIDERS.get(settings.llm_provider)
    fast = preset.fast_model if preset else ""
    return fast if fast and fast != getattr(llm, "model", None) else None


def build_llm(settings: Settings | None = None, provider: str | None = None, model: str | None = None) -> LLM:
    """Create the LLM backend for the configured provider (Anthropic native, or any OpenAI-compatible API)."""
    from .providers import PROVIDERS

    settings = settings or Settings()
    name = provider or settings.llm_provider
    preset = PROVIDERS[name]
    if preset.kind == "anthropic":
        return AnthropicLLM(settings, model=model or settings.llm_model or settings.model)
    from .llm_openai import OpenAICompatibleLLM

    return OpenAICompatibleLLM(settings, provider=name, model=model)


Handler = Callable[[str, str], BaseModel]


class ScriptedLLM:
    """Deterministic fake for tests: handlers keyed by output model type.

    A handler receives ``(system, user)`` and returns an instance of the output model.
    Handlers may be plain instances (returned each time) or callables.
    """

    def __init__(self, handlers: dict[type[BaseModel], Handler | BaseModel] | None = None):
        self.handlers = dict(handlers or {})
        self.calls: list[tuple[str, type[BaseModel], str]] = []
        self._log: list[LLMCallLog] = []

    @property
    def call_log(self) -> list[LLMCallLog]:
        return self._log

    def on(self, output_model: type[BaseModel], handler: Handler | BaseModel) -> "ScriptedLLM":
        self.handlers[output_model] = handler
        return self

    def structured(self, *, task: str, system: str, user: str, output_model: type[T], effort: Effort = "high",
                   max_tokens: int = 16000, model: str | None = None) -> T:
        self.calls.append((task, output_model, user))
        self._log.append(LLMCallLog(task=task, model=model or "scripted"))
        handler = self.handlers.get(output_model)
        if handler is None:
            raise KeyError(f"ScriptedLLM has no handler for {output_model.__name__} (task={task})")
        result = handler(system, user) if callable(handler) else handler
        if not isinstance(result, output_model):
            result = output_model.model_validate(result)
        return result
