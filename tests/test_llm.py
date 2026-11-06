"""AnthropicLLM request construction and stop-reason handling, with a stubbed SDK client."""
import sys
import types

import pytest
from pydantic import BaseModel

from research_gap_agent.config import Settings


class Out(BaseModel):
    answer: str


class _Usage:
    input_tokens = 10
    output_tokens = 5
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _Resp:
    def __init__(self, stop_reason="end_turn", parsed=Out(answer="ok")):
        self.stop_reason = stop_reason
        self.parsed_output = parsed
        self.usage = _Usage()
        self.model = "claude-opus-5-5"
        self.stop_details = types.SimpleNamespace(explanation="declined")


class _Stream:
    def __init__(self, result):
        self.result = result

    def __enter__(self):
        if isinstance(self.result, Exception):
            raise self.result
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.result


class _Messages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.kwargs = []

    def stream(self, **kwargs):
        self.kwargs.append(kwargs)
        return _Stream(self.responses.pop(0))


@pytest.fixture
def stub_anthropic(monkeypatch):
    class BadRequestError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.message = message

    mod = types.ModuleType("anthropic")
    mod.BadRequestError = BadRequestError

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(APIConnectionError):
        pass

    class APIStatusError(Exception):
        def __init__(self, message, status_code=500):
            super().__init__(message)
            self.status_code = status_code

    mod.APIConnectionError, mod.APITimeoutError, mod.APIStatusError = APIConnectionError, APITimeoutError, APIStatusError
    mod.Timeout = lambda *a, **k: None
    holder = {}

    class Anthropic:
        def __init__(self, **kw):
            self.beta = types.SimpleNamespace(messages=holder["messages"])

    mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return holder, BadRequestError


def _llm(holder, responses, **settings_kw):
    from research_gap_agent.llm import AnthropicLLM

    holder["messages"] = _Messages(responses)
    return AnthropicLLM(Settings(model="claude-opus-5-5", use_fallbacks=True, **settings_kw))


def test_request_shape_uses_structured_output_effort_caching_and_fallbacks(stub_anthropic):
    holder, _ = stub_anthropic
    llm = _llm(holder, [_Resp()])
    out = llm.structured(task="t", system="SYS", user="hello", output_model=Out, effort="medium")
    assert out.answer == "ok"
    kw = holder["messages"].kwargs[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_format"] is Out
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"} and kw["system"][0]["text"] == "SYS"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in kw  # Opus 5.5: thinking always on; parameter omitted
    assert llm.call_log[0].task == "t" and llm.call_log[0].input_tokens == 10


def test_model_override_applies_to_that_call_only(stub_anthropic):
    holder, _ = stub_anthropic
    llm = _llm(holder, [_Resp(), _Resp()])
    llm.structured(task="screen", system="s", user="u", output_model=Out, model="claude-haiku-5-5")
    llm.structured(task="judge", system="s", user="u", output_model=Out)
    assert holder["messages"].kwargs[0]["model"] == "claude-haiku-5-5"
    assert holder["messages"].kwargs[1]["model"] == "claude-opus-5-5"


def test_fallback_rejection_is_remembered_per_model(stub_anthropic):
    holder, BadRequestError = stub_anthropic
    llm = _llm(holder, [BadRequestError("fallbacks: unsupported"), _Resp(), _Resp()])
    llm.structured(task="screen", system="s", user="u", output_model=Out, model="claude-haiku-5-5")
    llm.structured(task="judge", system="s", user="u", output_model=Out)
    kws = holder["messages"].kwargs
    assert "fallbacks" in kws[0] and "fallbacks" not in kws[1]  # haiku: rejected, retried without
    assert "fallbacks" in kws[2]  # the main model still gets fallbacks


def test_fallback_parameter_rejected_is_retried_without_it(stub_anthropic):
    holder, BadRequestError = stub_anthropic
    llm = _llm(holder, [BadRequestError("fallbacks: unsupported"), _Resp()])
    llm.structured(task="t", system="s", user="u", output_model=Out)
    assert "fallbacks" in holder["messages"].kwargs[0] and "fallbacks" not in holder["messages"].kwargs[1]


def test_refusal_and_truncation_raise(stub_anthropic):
    from research_gap_agent.llm import LLMRefusal

    holder, _ = stub_anthropic
    llm = _llm(holder, [_Resp(stop_reason="refusal"), _Resp(stop_reason="max_tokens")])
    with pytest.raises(LLMRefusal):
        llm.structured(task="t", system="s", user="u", output_model=Out)
    with pytest.raises(RuntimeError, match="max_tokens"):
        llm.structured(task="t", system="s", user="u", output_model=Out)


def test_workspace_id_is_passed_when_configured(stub_anthropic):
    holder, _ = stub_anthropic
    llm = _llm(holder, [_Resp()], workspace_id="wrkspc_123")
    llm.structured(task="t", system="s", user="u", output_model=Out)
    assert holder["messages"].kwargs[0]["workspace_id"] == "wrkspc_123"
    llm2 = _llm(holder, [_Resp()], workspace_id=None)
    llm2.structured(task="t", system="s", user="u", output_model=Out)
    assert "workspace_id" not in holder["messages"].kwargs[0]


def test_unscoped_key_error_gets_actionable_hint(stub_anthropic):
    from research_gap_agent.llm import LLMConfigError

    holder, BadRequestError = stub_anthropic
    llm = _llm(holder, [BadRequestError("This API key is not scoped to a workspace, so this request must include the anthropic-workspace-id header")])
    with pytest.raises(LLMConfigError, match="ANTHROPIC_WORKSPACE_ID"):
        llm.structured(task="t", system="s", user="u", output_model=Out)


def test_transient_stream_failures_are_retried(stub_anthropic, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    holder, _ = stub_anthropic
    import sys

    mod = sys.modules["anthropic"]
    llm = _llm(holder, [mod.APITimeoutError("read timed out"), mod.APIConnectionError("dropped"), _Resp()])
    out = llm.structured(task="t", system="s", user="u", output_model=Out)
    assert out.answer == "ok" and len(holder["messages"].kwargs) == 3


def test_transient_failures_give_up_after_retries(stub_anthropic, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    holder, _ = stub_anthropic
    import sys

    mod = sys.modules["anthropic"]
    llm = _llm(holder, [mod.APITimeoutError("t")] * 4)
    with pytest.raises(mod.APITimeoutError):
        llm.structured(task="t", system="s", user="u", output_model=Out)
    assert len(holder["messages"].kwargs) == 4


def test_stalled_call_is_abandoned_at_hard_deadline_and_retried(stub_anthropic, monkeypatch):
    """A call that never returns (wedged socket, DNS hang) is abandoned after the hard deadline and retried."""
    import threading
    from research_gap_agent.llm import LLMStalled  # noqa: F401

    monkeypatch.setattr("time.sleep", lambda s: None)
    release = threading.Event()

    class _Hang:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            release.wait(5)  # would block "forever" without the watchdog
            return _Resp()

    holder, _ = stub_anthropic
    llm = _llm(holder, [_Resp()])
    llm.hard_deadline = 0.2
    calls = holder["messages"].kwargs
    events = []
    llm.on_event = events.append
    hang = _Hang()
    real_stream = holder["messages"].stream

    def stream(**kwargs):
        if not calls:
            calls.append(kwargs)
            return hang
        return real_stream(**kwargs)

    holder["messages"].stream = stream
    try:
        out = llm.structured(task="t", system="s", user="u", output_model=Out)
    finally:
        release.set()
    assert out.answer == "ok"
    assert len(calls) == 2
    assert any("LLMStalled" in e for e in events), events


def test_stalled_call_gives_up_after_retries(stub_anthropic, monkeypatch):
    import threading
    from research_gap_agent.llm import LLMStalled

    monkeypatch.setattr("time.sleep", lambda s: None)
    release = threading.Event()

    class _Hang:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            release.wait(5)
            return _Resp()

    holder, _ = stub_anthropic
    llm = _llm(holder, [])
    llm.hard_deadline = 0.05
    llm.transient_retries = 1
    holder["messages"].stream = lambda **kw: _Hang()
    try:
        with pytest.raises(LLMStalled):
            llm.structured(task="t", system="s", user="u", output_model=Out)
    finally:
        release.set()
