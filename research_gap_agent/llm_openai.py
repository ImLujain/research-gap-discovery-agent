"""Structured generation through any OpenAI-compatible chat-completions API.

Strategy for structured output, most to least capable, falling back on a 400 from the server:
1. ``response_format={"type": "json_schema", ..., "strict": true}`` with a strict-compatible schema
2. the same without ``strict``
3. ``response_format={"type": "json_object"}`` with the schema in the system prompt
4. plain text with the schema in the system prompt and JSON extracted from the reply
Every reply is validated with Pydantic; one corrective retry is made on validation failure.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings
from .models import LLMCallLog
from .providers import PROVIDERS, ProviderPreset, resolve_api_key

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_UNSUPPORTED_STRICT_KEYS = {"default", "format", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                            "minLength", "maxLength", "minItems", "maxItems", "pattern", "examples"}


def strict_schema(schema: dict) -> dict:
    """Make a Pydantic JSON schema acceptable to OpenAI strict mode: every property required,
    additionalProperties false, unsupported validation keywords removed."""
    schema = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k in list(node):
                if k in _UNSUPPORTED_STRICT_KEYS:
                    del node[k]
            if node.get("type") == "object" or "properties" in node:
                props = node.get("properties") or {}
                node["properties"] = props
                node["required"] = list(props)
                node["additionalProperties"] = False
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


def extract_json(text: str) -> str:
    """Pull a JSON object out of a reply that may wrap it in prose or a code fence."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        return m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


class OpenAICompatibleLLM:
    def __init__(self, settings: Optional[Settings] = None, provider: Optional[str] = None,
                 model: Optional[str] = None, client: Any = None):
        self.settings = settings or Settings()
        self.preset: ProviderPreset = PROVIDERS[provider or self.settings.llm_provider]
        self.model = model or self.settings.llm_model or self.preset.default_model
        if not self.model:
            raise ValueError(f"Provider {self.preset.name!r} needs a model: set RGA_LLM_MODEL or --model")
        self.base_url = self.settings.llm_base_url or self.preset.base_url
        self._log: list[LLMCallLog] = []
        self.on_event = None  # progress sink for retry notices (GUI/CLI log)
        self._mode = 0  # index into the fallback chain; sticks once a mode works
        self._token_param = "max_completion_tokens" if self.preset.name in ("openai", "azure") else "max_tokens"
        self._send_effort = self.preset.reasoning_effort or self.settings.llm_reasoning_effort
        self.client = client or self._make_client()

    def _make_client(self):
        import openai

        self._openai = openai
        key = resolve_api_key(self.preset)
        if self.preset.needs_key and not key:
            raise ValueError(f"No API key for provider {self.preset.name!r}: set {self.preset.key_env} (or RGA_LLM_API_KEY)")
        if self.preset.kind == "azure":
            endpoint = self.settings.azure_endpoint
            if not endpoint:
                raise ValueError("Azure provider needs AZURE_OPENAI_ENDPOINT")
            return openai.AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=self.settings.azure_api_version,
                                      max_retries=4, timeout=self.settings.llm_timeout)
        return openai.OpenAI(api_key=key or "none", base_url=self.base_url, max_retries=4, timeout=self.settings.llm_timeout)

    @property
    def call_log(self) -> list[LLMCallLog]:
        return self._log

    # ------------------------------------------------------------------
    def _request_kwargs(self, system: str, user: str, schema: dict, name: str, mode: int, max_tokens: int,
                        effort: str) -> dict:
        schema_hint = ("\n\nRespond ONLY with a JSON object that conforms exactly to this JSON schema (no prose, no code fence):\n"
                       + json.dumps(schema))
        sys_text = system if mode < 2 else system + schema_hint
        kw: dict = {"model": self.model, "messages": [{"role": "system", "content": sys_text},
                                                      {"role": "user", "content": user}],
                    self._token_param: max_tokens}
        if mode == 0:
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": name, "schema": strict_schema(schema), "strict": True}}
        elif mode == 1:
            kw["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}
        elif mode == 2:
            kw["response_format"] = {"type": "json_object"}
        if self._send_effort:
            kw["reasoning_effort"] = _EFFORT_MAP.get(effort, "high")
        return kw

    def _is_bad_request(self, e: Exception) -> bool:
        openai_mod = getattr(self, "_openai", None)
        if openai_mod is not None and isinstance(e, openai_mod.BadRequestError):
            return True
        return type(e).__name__ == "BadRequestError"

    def _call(self, kw: dict):
        return self.client.chat.completions.create(**kw)

    def structured(self, *, task: str, system: str, user: str, output_model: type[T], effort: str = "high",
                   max_tokens: int = 16000) -> T:
        schema = output_model.model_json_schema()
        name = re.sub(r"[^a-zA-Z0-9_]", "_", output_model.__name__)[:64]
        start_mode = 0 if self.preset.strict_json_schema else 1
        mode = max(self._mode, start_mode)
        response = None
        while response is None:
            kw = self._request_kwargs(system, user, schema, name, mode, max_tokens, effort)
            try:
                response = self._call(kw)
            except Exception as e:  # noqa: BLE001
                if not self._is_bad_request(e):
                    raise
                msg = str(e)
                if self._token_param == "max_completion_tokens" and "max_completion_tokens" in msg:
                    self._token_param = "max_tokens"
                    continue
                if self._token_param == "max_tokens" and "max_completion_tokens" in msg:
                    self._token_param = "max_completion_tokens"
                    continue
                if self._send_effort and "reasoning_effort" in msg:
                    self._send_effort = False
                    continue
                if mode < 3:
                    log.info("%s rejected structured-output mode %d (%s); falling back", self.preset.name, mode, msg[:120])
                    mode += 1
                    continue
                raise
        self._mode = mode
        return self._parse(response, output_model, task, schema, kw)

    def _parse(self, response, output_model: type[T], task: str, schema: dict, kw: dict, retried: bool = False) -> T:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        self._log.append(LLMCallLog(task=task, model=getattr(response, "model", self.model) or self.model,
                                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                                    output_tokens=getattr(usage, "completion_tokens", 0) or 0))
        finish = getattr(choice, "finish_reason", None)
        if finish == "length":
            raise RuntimeError(f"Task {task!r} hit the output token limit; output truncated")
        if getattr(choice.message, "refusal", None):
            raise RuntimeError(f"Model refused task {task!r}: {choice.message.refusal}")
        text = choice.message.content or ""
        try:
            return output_model.model_validate_json(extract_json(text))
        except (ValidationError, ValueError) as e:
            if retried:
                raise RuntimeError(f"Task {task!r}: model output did not match the schema after a retry: {e}") from e
            msg = f"{self.preset.name}: invalid structured output for {task}, retrying with the validation error"
            log.warning(msg)
            if self.on_event:
                self.on_event(msg)
            kw2 = copy.deepcopy(kw)
            kw2["messages"].append({"role": "assistant", "content": text})
            kw2["messages"].append({"role": "user", "content":
                                    "Your previous reply was not valid for the required schema:\n" + str(e)[:2000]
                                    + "\n\nReply again with ONLY a corrected JSON object matching the schema."})
            return self._parse(self._call(kw2), output_model, task, schema, kw2, retried=True)
