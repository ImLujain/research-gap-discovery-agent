"""Runtime configuration read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env_file(path: str | os.PathLike | None = None, *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE lines from a .env file into os.environ (existing variables win unless override=True).

    Looks for ``.env`` in the current working directory, then next to this package's project root.
    ``RGA_ENV_FILE`` pins a specific file instead (used by tests to stay away from the real one).
    Supports comments, blank lines, an optional ``export`` prefix and single/double quotes.
    """
    if path is None and os.environ.get("RGA_ENV_FILE"):
        path = os.environ["RGA_ENV_FILE"]
    candidates = [Path(path)] if path else [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
    loaded: dict[str, str] = {}
    for cand in candidates:
        if not cand.is_file():
            continue
        for raw in cand.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if not key or (key in os.environ and not override):
                continue
            os.environ[key] = value
            loaded[key] = value
        break
    return loaded


def default_env_path() -> Path:
    """The .env file the app reads: RGA_ENV_FILE if set, else cwd if present there, else the project root."""
    if os.environ.get("RGA_ENV_FILE"):
        return Path(os.environ["RGA_ENV_FILE"])
    cwd = Path.cwd() / ".env"
    return cwd if cwd.is_file() else Path(__file__).resolve().parent.parent / ".env"


def update_env_file(values: dict[str, str], path: str | os.PathLike | None = None) -> Path:
    """Write KEY=VALUE pairs into the .env file, replacing existing keys in place and keeping other lines.

    Also applies the values to the running process. Empty values remove the variable from both.
    """
    target = Path(path) if path else default_env_path()
    lines = target.read_text().splitlines() if target.is_file() else []
    remaining = dict(values)
    out: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        key = None
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped[len("export "):].partition("=")[0].strip() if stripped.startswith("export ") else stripped.partition("=")[0].strip()
        if key in remaining:
            val = remaining.pop(key)
            if val:
                out.append(f"{key}={val}")
            continue  # empty value: drop the line
        out.append(raw)
    for key, val in remaining.items():
        if val:
            out.append(f"{key}={val}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out) + ("\n" if out else ""))
    try:
        target.chmod(0o600)
    except OSError:
        pass
    for key, val in values.items():
        if val:
            os.environ[key] = val
        else:
            os.environ.pop(key, None)
    return target


def _detect() -> str:
    from .providers import detect_provider

    return detect_provider()


@dataclass
class Settings:
    mailto: str = field(default_factory=lambda: os.environ.get("RGA_MAILTO", ""))
    user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "RGA_USER_AGENT", "ResearchGapAgent/0.1 (https://github.com/ImLujain/research-gap-discovery-agent; mailto:{mailto})"
        )
    )
    openalex_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENALEX_API_KEY"))
    s2_api_key: str | None = field(default_factory=lambda: os.environ.get("S2_API_KEY"))
    ncbi_api_key: str | None = field(default_factory=lambda: os.environ.get("NCBI_API_KEY"))
    serpapi_key: str | None = field(default_factory=lambda: os.environ.get("SERPAPI_KEY"))
    github_token: str | None = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN"))
    workspace_id: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_WORKSPACE_ID") or None)
    cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("RGA_CACHE_DIR", ".api_cache")))
    runs_dir: Path = field(default_factory=lambda: Path(os.environ.get("RGA_RUNS_DIR", "runs")))
    model: str = field(default_factory=lambda: os.environ.get("RGA_MODEL", "claude-opus-5-5"))  # Anthropic model
    use_fallbacks: bool = field(default_factory=lambda: os.environ.get("RGA_FALLBACKS", "1") != "0")
    # generic LLM provider selection (see providers.py)
    llm_provider: str = field(default_factory=lambda: _detect())
    llm_model: str | None = field(default_factory=lambda: os.environ.get("RGA_LLM_MODEL") or None)
    llm_base_url: str | None = field(default_factory=lambda: os.environ.get("RGA_LLM_BASE_URL") or None)
    llm_reasoning_effort: bool = field(default_factory=lambda: os.environ.get("RGA_LLM_REASONING_EFFORT", "0") == "1")
    azure_endpoint: str | None = field(default_factory=lambda: os.environ.get("AZURE_OPENAI_ENDPOINT") or None)
    azure_api_version: str = field(default_factory=lambda: os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))
    http_timeout: float = 30.0
    llm_timeout: float = field(default_factory=lambda: float(os.environ.get("RGA_LLM_TIMEOUT", "1800")))  # seconds

    def resolved_user_agent(self) -> str:
        return self.user_agent.format(mailto=self.mailto or "unset")
