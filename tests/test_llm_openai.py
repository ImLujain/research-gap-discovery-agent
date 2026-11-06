"""OpenAI-compatible backend: schema strictification, fallback chain, validation retry, provider detection."""
import types
from typing import Optional

import pytest
from pydantic import BaseModel, Field

from research_gap_agent.config import Settings
from research_gap_agent.llm_openai import OpenAICompatibleLLM, extract_json, strict_schema
from research_gap_agent.providers import PROVIDERS, detect_provider, resolve_api_key


class Inner(BaseModel):
    name: str
    score: int = Field(default=0, ge=0)


class Out(BaseModel):
    answer: str
    maybe: Optional[str] = None
    items: list[Inner] = Field(default_factory=list)


def test_strict_schema_requires_all_and_strips_unsupported():
    s = strict_schema(Out.model_json_schema())
    assert s["required"] == ["answer", "maybe", "items"] and s["additionalProperties"] is False
    inner = s["$defs"]["Inner"]
    assert inner["required"] == ["name", "score"] and inner["additionalProperties"] is False
    assert "minimum" not in inner["properties"]["score"] and "default" not in inner["properties"]["score"]


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure:\n```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('text {"a": {"b": 2}} trailing') == '{"a": {"b": 2}}'


class BadRequestError(Exception):
    pass


def _resp(content, finish="stop", refusal=None):
    msg = types.SimpleNamespace(content=content, refusal=refusal)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish)],
                                 usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=3), model="m")


class FakeClient:
    def __init__(self, script):
        self.script = list(script)  # each item: response or exception
        self.calls = []

    @property
    def chat(self):
        client = self

        class _C:
            class completions:
                @staticmethod
                def create(**kw):
                    client.calls.append(kw)
                    item = client.script.pop(0)
                    if isinstance(item, Exception):
                        raise item
                    return item
        return _C()


def _llm(script, provider="openai", **kw):
    settings = Settings(llm_provider=provider, llm_model="test-model", **kw)
    return OpenAICompatibleLLM(settings, client=FakeClient(script))


def test_strict_mode_request_shape_and_parse():
    llm = _llm([_resp('{"answer": "hi", "maybe": null, "items": []}')])
    out = llm.structured(task="t", system="SYS", user="u", output_model=Out, effort="xhigh")
    assert out.answer == "hi"
    kw = llm.client.calls[0]
    assert kw["response_format"]["type"] == "json_schema" and kw["response_format"]["json_schema"]["strict"] is True
    assert kw["messages"][0] == {"role": "system", "content": "SYS"} and kw["max_completion_tokens"] == 16000
    assert kw["reasoning_effort"] == "high"  # xhigh mapped down for OpenAI
    assert llm.call_log[0].input_tokens == 7


def test_fallback_chain_on_bad_request_sticks():
    llm = _llm([BadRequestError("response_format strict unsupported"), BadRequestError("json_schema unsupported"),
                _resp('{"answer": "ok"}'), _resp('{"answer": "again"}')])
    assert llm.structured(task="t", system="s", user="u", output_model=Out).answer == "ok"
    modes = [c.get("response_format", {}).get("type") for c in llm.client.calls]
    assert modes[:3] == ["json_schema", "json_schema", "json_object"]
    assert "JSON schema" in llm.client.calls[2]["messages"][0]["content"]  # schema moved into the prompt
    llm.structured(task="t2", system="s", user="u", output_model=Out)
    assert llm.client.calls[3]["response_format"]["type"] == "json_object"  # remembered working mode


def test_token_param_and_effort_fallbacks():
    llm = _llm([BadRequestError("Unsupported parameter: 'max_completion_tokens'"),
                BadRequestError("Unrecognized request argument: reasoning_effort"), _resp('{"answer": "x"}')])
    llm.structured(task="t", system="s", user="u", output_model=Out)
    last = llm.client.calls[-1]
    assert "max_tokens" in last and "max_completion_tokens" not in last and "reasoning_effort" not in last


def test_validation_error_triggers_one_corrective_retry():
    llm = _llm([_resp('{"answer": 5, "items": "nope"}'), _resp('```json\n{"answer": "fixed"}\n```')], provider="groq")
    out = llm.structured(task="t", system="s", user="u", output_model=Out)
    assert out.answer == "fixed"
    retry = llm.client.calls[1]["messages"]
    assert retry[-2]["role"] == "assistant" and "not valid" in retry[-1]["content"]
    assert llm.client.calls[0]["response_format"]["type"] == "json_schema" and "strict" not in llm.client.calls[0]["response_format"]["json_schema"]


def test_truncation_and_refusal_raise():
    llm = _llm([_resp('{"answer":', finish="length")])
    with pytest.raises(RuntimeError, match="truncated"):
        llm.structured(task="t", system="s", user="u", output_model=Out)
    llm2 = _llm([_resp(None, refusal="no")])
    with pytest.raises(RuntimeError, match="refused"):
        llm2.structured(task="t", system="s", user="u", output_model=Out)


def test_provider_detection_and_keys():
    assert detect_provider({}) == "anthropic"
    assert detect_provider({"OPENAI_API_KEY": "k"}) == "openai"
    assert detect_provider({"GOOGLE_API_KEY": "g"}) == "gemini"
    assert detect_provider({"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "k"}) == "anthropic"
    assert detect_provider({"RGA_LLM_PROVIDER": "ollama", "OPENAI_API_KEY": "k"}) == "ollama"
    assert detect_provider({"RGA_LLM_BASE_URL": "http://x/v1", "RGA_LLM_API_KEY": "z"}) == "openai-compatible"
    assert resolve_api_key(PROVIDERS["openai-compatible"], {"RGA_LLM_API_KEY": "z"}) == "z"
    assert resolve_api_key(PROVIDERS["mistral"], {"RGA_LLM_PROVIDER": "mistral", "RGA_LLM_API_KEY": "gen"}) == "gen"
    with pytest.raises(ValueError, match="Unknown"):
        detect_provider({"RGA_LLM_PROVIDER": "bogus"})


def test_build_llm_routes_by_provider(monkeypatch):
    from research_gap_agent.llm import build_llm

    monkeypatch.setenv("OLLAMA_API_KEY", "")
    llm = build_llm(Settings(llm_provider="ollama", llm_model="llama3.1"))
    assert isinstance(llm, OpenAICompatibleLLM) and llm.base_url == "http://localhost:11434/v1"
    with pytest.raises(ValueError, match="needs a model"):
        build_llm(Settings(llm_provider="openai-compatible", llm_model=None, llm_base_url="http://x"))
