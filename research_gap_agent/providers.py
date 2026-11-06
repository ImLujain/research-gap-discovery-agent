"""LLM provider registry: presets for Anthropic and every OpenAI-compatible API, plus auto-detection."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    kind: str  # "anthropic" | "openai" | "azure"
    key_env: str  # environment variable holding the API key
    base_url: Optional[str]  # None = SDK default
    default_model: str
    needs_key: bool = True
    strict_json_schema: bool = True  # whether to try OpenAI strict structured outputs first
    reasoning_effort: bool = False  # whether the API accepts `reasoning_effort`
    note: str = ""
    fast_model: str = ""  # cheaper model used for screening by default; blank = use the main model


PROVIDERS: dict[str, ProviderPreset] = {p.name: p for p in [
    ProviderPreset("anthropic", "anthropic", "ANTHROPIC_API_KEY", None, "claude-opus-5-5",
                   note="Native Claude API with structured outputs and server-side fallbacks.",
                   fast_model="claude-haiku-5-5"),
    ProviderPreset("openai", "openai", "OPENAI_API_KEY", None, "gpt-5", reasoning_effort=True,
                   note="OpenAI platform."),
    ProviderPreset("azure", "azure", "AZURE_OPENAI_API_KEY", None, "gpt-4o",
                   note="Azure OpenAI; set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_VERSION; model = deployment name."),
    ProviderPreset("gemini", "openai", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/",
                   "gemini-2.5-pro", strict_json_schema=False, note="Google Gemini via its OpenAI-compatible endpoint."),
    ProviderPreset("mistral", "openai", "MISTRAL_API_KEY", "https://api.mistral.ai/v1", "mistral-large-latest",
                   strict_json_schema=False),
    ProviderPreset("groq", "openai", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile",
                   strict_json_schema=False),
    ProviderPreset("together", "openai", "TOGETHER_API_KEY", "https://api.together.xyz/v1",
                   "meta-llama/Llama-3.3-70B-Instruct-Turbo", strict_json_schema=False),
    ProviderPreset("openrouter", "openai", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai/gpt-5",
                   strict_json_schema=False, note="Any model on OpenRouter; model ids look like vendor/model."),
    ProviderPreset("deepseek", "openai", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat",
                   strict_json_schema=False),
    ProviderPreset("xai", "openai", "XAI_API_KEY", "https://api.x.ai/v1", "grok-4"),
    ProviderPreset("ollama", "openai", "OLLAMA_API_KEY", "http://localhost:11434/v1", "llama3.1", needs_key=False,
                   strict_json_schema=False, note="Local Ollama server; no key needed."),
    ProviderPreset("lmstudio", "openai", "LMSTUDIO_API_KEY", "http://localhost:1234/v1", "local-model", needs_key=False,
                   strict_json_schema=False, note="Local LM Studio server; no key needed."),
    ProviderPreset("openai-compatible", "openai", "RGA_LLM_API_KEY", None, "",
                   strict_json_schema=False, note="Any other OpenAI-compatible server: set RGA_LLM_BASE_URL and RGA_LLM_MODEL."),
]}

AUTO_DETECT_ORDER = ["anthropic", "openai", "azure", "gemini", "mistral", "groq", "together", "openrouter", "deepseek", "xai"]


def resolve_api_key(preset: ProviderPreset, env: Optional[dict] = None) -> Optional[str]:
    env = os.environ if env is None else env
    generic = env.get("RGA_LLM_API_KEY")
    if preset.name == "openai-compatible":
        return generic or None
    key = env.get(preset.key_env)
    if not key and preset.name == "gemini":
        key = env.get("GOOGLE_API_KEY")
    if not key and generic and env.get("RGA_LLM_PROVIDER") == preset.name:
        key = generic
    return key or None


def detect_provider(env: Optional[dict] = None) -> str:
    """Explicit RGA_LLM_PROVIDER wins; otherwise the first provider whose key is set; a base URL implies openai-compatible."""
    env = os.environ if env is None else env
    explicit = (env.get("RGA_LLM_PROVIDER") or "").strip().lower()
    if explicit:
        if explicit not in PROVIDERS:
            raise ValueError(f"Unknown RGA_LLM_PROVIDER={explicit!r}; known: {', '.join(PROVIDERS)}")
        return explicit
    for name in AUTO_DETECT_ORDER:
        if resolve_api_key(PROVIDERS[name], env):
            return name
    if env.get("RGA_LLM_BASE_URL"):
        return "openai-compatible"
    return "anthropic"
