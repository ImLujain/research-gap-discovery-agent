"""Streamlit GUI for the Research Gap Discovery Agent.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st

from research_gap_agent.config import Settings, default_env_path, load_env_file, update_env_file
from research_gap_agent.coordinator import STAGES, Coordinator, RunCancelled
from research_gap_agent.models import RunOptions, RunState
from research_gap_agent.providers import PROVIDERS, detect_provider, resolve_api_key

st.set_page_config(page_title="Research Gap Discovery Agent", page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")
load_env_file()

CURRENT_YEAR = datetime.now().year

# ----------------------------------------------------------------------------- plain-language labels
SOURCE_INFO: dict[str, tuple[str, str]] = {
    "openalex": ("OpenAlex", "the largest open index of scholarly papers (free)"),
    "semantic_scholar": ("Semantic Scholar", "AI-curated index with citation links (free, key recommended)"),
    "crossref": ("Crossref", "the publishers' DOI registry (free)"),
    "arxiv": ("arXiv", "preprints in computing, physics and maths (free)"),
    "pubmed": ("PubMed", "biomedical and life-science literature (free)"),
    "google_scholar": ("Google Scholar", "through SerpAPI (needs a paid SerpAPI key)"),
    "github": ("GitHub", "code repositories, to find existing tools (free)"),
}
ALL_SOURCES = list(SOURCE_INFO)
DEFAULT_SOURCES = ["openalex", "semantic_scholar", "crossref", "arxiv"]

PROVIDER_LABELS = {
    "anthropic": "Anthropic (Claude)", "openai": "OpenAI (GPT)", "azure": "Azure OpenAI", "gemini": "Google Gemini",
    "mistral": "Mistral", "groq": "Groq", "together": "Together AI", "openrouter": "OpenRouter", "deepseek": "DeepSeek",
    "xai": "xAI (Grok)", "ollama": "Ollama (local)", "lmstudio": "LM Studio (local)",
    "openai-compatible": "Other OpenAI-compatible server",
}

STAGE_INFO: dict[str, tuple[str, str]] = {
    "plan": ("Planning the search", "Breaking the topic into concepts, synonyms and search queries"),
    "discover": ("Searching the databases", "Running every query on every selected source and merging duplicates"),
    "screen": ("Screening papers", "Deciding which papers are actually about the topic"),
    "fulltext": ("Reading full texts", "Downloading and reading open-access PDFs"),
    "extract": ("Extracting evidence", "Pulling the objective, method, findings and limitations out of each paper"),
    "landscape": ("Mapping the field", "Grouping papers into themes and recording what the evidence shows"),
    "generate_gaps": ("Proposing gaps", "Suggesting candidate research gaps grounded in the evidence"),
    "validate": ("Trying to disprove each gap", "Searching again with different wording to find work that already fills it"),
    "rank": ("Ranking the gaps", "Scoring the surviving gaps on nine criteria"),
    "recommend": ("Writing the recommendation", "Turning the top gap into a concrete research plan"),
    "verify": ("Verifying citations", "Checking every cited paper really exists and says what is claimed"),
    "report": ("Writing the report", "Rendering the final report and exports"),
}

STATUS_INFO: dict[str, tuple[str, str, str]] = {  # value -> (label, explanation, css class)
    "limited_evidence_found": ("Open gap", "Counter-searches found little or no work that fills this gap", "rga-ok"),
    "preliminary_evidence": ("Early work only", "Only preliminary studies exist so far", "rga-ok"),
    "conflicting_evidence": ("Conflicting findings", "Existing studies disagree with each other", "rga-warn"),
    "partially_addressed": ("Partly addressed", "Some aspects are already covered; the gap was narrowed", "rga-warn"),
    "cannot_validate": ("Could not verify", "The validation searches failed, so this is unconfirmed", "rga-grey"),
    "addressed": ("Already addressed", "Existing work already covers this", "rga-bad"),
}
CATEGORY_LABELS = {
    "knowledge": "Knowledge gap", "methodological": "Method gap", "empirical": "Missing experiments",
    "evaluation": "Evaluation gap", "generalizability": "Untested generalisation", "integration": "Integration gap",
    "contradiction": "Unresolved contradiction", "deployment": "Deployment gap",
    "interdisciplinary": "Cross-discipline gap", "temporal": "Out-of-date evidence",
}
GOALS = {"research": "A research question to study", "tool": "A tool or system to build"}
PRIORITIES = {"(none)": "No preference", "theoretical": "Theoretical", "experimental": "Experimental",
              "applied": "Applied", "interdisciplinary": "Interdisciplinary"}
DEPTHS: dict[str, dict] = {
    "Quick": {"max_papers": 30, "per_source": 15, "rounds": 1, "max_ft": 5,
              "caption": "About 30 papers, one validation round. Fastest and cheapest."},
    "Standard": {"max_papers": 60, "per_source": 25, "rounds": 2, "max_ft": 20,
                 "caption": "About 60 papers, two validation rounds. Good default."},
    "Thorough": {"max_papers": 120, "per_source": 40, "rounds": 3, "max_ft": 40,
                 "caption": "About 120 papers, three validation rounds. Slow but most complete."},
    "Custom": {"caption": "Set the exact numbers under Advanced settings."},
}
EXPORT_FILES = ["report.html", "report.md", "appendix.md", "references.bib", "references.ris", "corpus.csv", "state.json"]
EXPORT_LABELS = {
    "report.html": "Report (web page)", "report.md": "Report (Markdown)", "appendix.md": "Appendix (all evidence)",
    "references.bib": "References (BibTeX)", "references.ris": "References (RIS)", "corpus.csv": "All papers (CSV)",
    "state.json": "Raw run data (JSON)",
}

# Credential fields per provider: (env name, label, is_secret, help)
PROVIDER_FIELDS: dict[str, list[tuple[str, str, bool, str]]] = {
    "anthropic": [("ANTHROPIC_API_KEY", "Anthropic API key", True,
                   "Create one at platform.claude.com > API Keys. Max/Ultra/Team subscribers: link a Console org in "
                   "claude.ai > Settings > Billing first."),
                  ("ANTHROPIC_WORKSPACE_ID", "Anthropic workspace id (rarely needed)", False,
                   "Only if the model says the key is not scoped to a workspace: Console > Settings > Workspaces > id (wrkspc_...).")],
    "openai": [("OPENAI_API_KEY", "OpenAI API key", True, "platform.openai.com > API keys")],
    "azure": [("AZURE_OPENAI_API_KEY", "Azure OpenAI key", True, "Azure portal > your OpenAI resource > Keys"),
              ("AZURE_OPENAI_ENDPOINT", "Azure endpoint", False, "https://<resource>.openai.azure.com/"),
              ("AZURE_OPENAI_API_VERSION", "Azure API version", False, "e.g. 2024-10-21")],
    "gemini": [("GEMINI_API_KEY", "Gemini API key", True, "aistudio.google.com > Get API key")],
    "mistral": [("MISTRAL_API_KEY", "Mistral API key", True, "console.mistral.ai")],
    "groq": [("GROQ_API_KEY", "Groq API key", True, "console.groq.com")],
    "together": [("TOGETHER_API_KEY", "Together API key", True, "api.together.ai")],
    "openrouter": [("OPENROUTER_API_KEY", "OpenRouter API key", True, "openrouter.ai > Keys")],
    "deepseek": [("DEEPSEEK_API_KEY", "DeepSeek API key", True, "platform.deepseek.com")],
    "xai": [("XAI_API_KEY", "xAI API key", True, "console.x.ai")],
    "ollama": [("RGA_LLM_BASE_URL", "Server address", False, "Leave blank for http://localhost:11434/v1")],
    "lmstudio": [("RGA_LLM_BASE_URL", "Server address", False, "Leave blank for http://localhost:1234/v1")],
    "openai-compatible": [("RGA_LLM_BASE_URL", "Server address", False, "Base URL of the server, e.g. http://host:8000/v1"),
                          ("RGA_LLM_API_KEY", "API key (if the server needs one)", True, "")],
}
SOURCE_FIELDS: list[tuple[str, str, bool, str]] = [
    ("RGA_MAILTO", "Your email", False,
     "Recommended. OpenAlex, Crossref and Unpaywall give faster access to clients that identify themselves."),
    ("OPENALEX_API_KEY", "OpenAlex key", True,
     "Recommended. Without it OpenAlex counts your requests against a free daily budget shared by everyone on your network."),
    ("S2_API_KEY", "Semantic Scholar key", True, "Optional but strongly recommended; the free key removes heavy rate limits."),
    ("SERPAPI_KEY", "SerpAPI key", True, "Optional. Enables the Google Scholar source (paid service)."),
    ("NCBI_API_KEY", "PubMed (NCBI) key", True, "Optional. Higher PubMed rate limit."),
    ("GITHUB_TOKEN", "GitHub token", True, "Optional. Raises the GitHub search limit from 10 to 30 requests per minute."),
]
FORM_KEYS = ["f_topic", "f_discipline", "f_from", "f_to", "f_method", "f_priority", "f_goal", "f_existing", "f_sources",
             "f_depth", "f_max_papers", "f_per_source", "f_rounds", "f_max_ft", "f_provider", "f_model", "f_base_url",
             "f_screen_model", "f_concurrency"]

st.markdown("""
<style>
  .block-container { padding-top: 3rem; max-width: 1120px; }
  [data-testid="stMainBlockContainer"] h1 { font-size: 1.6rem !important; font-weight: 600 !important; padding-bottom: 0.2rem; }
  [data-testid="stMainBlockContainer"] h2 { font-size: 1.25rem !important; font-weight: 600 !important; }
  [data-testid="stMainBlockContainer"] h3 { font-size: 1.05rem !important; font-weight: 600 !important; }
  .rga-hero { opacity: 0.75; margin-top: 0; margin-bottom: 1rem; }
  .rga-muted { opacity: 0.65; font-size: 0.86rem; }
  .rga-status { font-weight: 600; }
  .rga-ok { color: #2E7D32; } .rga-warn { color: #B26A00; } .rga-bad { color: #B3261E; } .rga-grey { opacity: 0.6; }
  div[data-testid="stMetric"] { border-left: 2px solid rgba(127,127,127,0.35); padding-left: 0.8rem; }
  [data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0.3rem; }
  .st-key-theme_light button, .st-key-theme_dark button { min-height: 2.4rem; overflow: visible; }
  .st-key-theme_light button span, .st-key-theme_dark button span { font-size: 1.25rem; line-height: 1; overflow: visible; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def settings() -> Settings:
    return Settings()


def list_runs(runs_dir: Path) -> list[tuple[str, RunState]]:
    out = []
    if not runs_dir.exists():
        return out
    for d in sorted(runs_dir.iterdir(), reverse=True):
        sp = d / "state.json"
        if sp.exists():
            try:
                out.append((d.name, RunState.model_validate(json.loads(sp.read_text()))))
            except Exception:  # noqa: BLE001 - skip corrupt runs rather than crash the UI
                continue
    return out


def run_label(rid: str, s: RunState) -> str:
    n = len(s.stages_completed)
    status = "complete" if n == len(STAGES) else f"{n}/{len(STAGES)} stages"
    when = s.created_at.strftime("%d %b %Y")
    return f"{s.options.topic[:48]} · {when} · {status} · {rid}"


def provider_label(name: str) -> str:
    return PROVIDER_LABELS.get(name, name)


def provider_ready(name: str) -> tuple[bool, str]:
    preset = PROVIDERS[name]
    if name == "azure" and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return False, "the Azure endpoint is missing"
    if name == "openai-compatible" and not os.environ.get("RGA_LLM_BASE_URL"):
        return False, "the server address is missing"
    if preset.needs_key and not resolve_api_key(preset):
        return False, "no API key saved yet"
    return True, "ready"


def status_text(value: str) -> str:
    label, expl, css = STATUS_INFO.get(value, (value.replace("_", " "), "", "rga-grey"))
    return f'<span class="rga-status {css}" title="{expl}">{label}</span>'


def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else (f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s")


def prefill_form_from(state: RunState) -> None:
    """Copy a previous run's options into the form widgets (called before the widgets are created)."""
    o = state.options
    st.session_state.update({
        "f_topic": o.topic, "f_discipline": o.discipline or "", "f_from": o.from_year or 1900,
        "f_to": o.to_year or CURRENT_YEAR, "f_method": o.methodology_preference or "",
        "f_priority": o.priority or "(none)", "f_goal": o.goal, "f_existing": o.existing_work or "",
        "f_sources": [s for s in o.sources if s in ALL_SOURCES], "f_depth": "Custom",
        "f_max_papers": o.max_papers_to_analyze, "f_per_source": o.per_source_limit,
        "f_rounds": o.validation_rounds, "f_max_ft": o.max_full_text,
        "f_provider": o.provider if o.provider in PROVIDERS else detect_provider(), "f_model": o.model,
        "f_base_url": os.environ.get("RGA_LLM_BASE_URL", ""), "f_screen_model": o.screen_model or "",
    })


def run_in_background(options: RunOptions | None, resume_id: str | None, q: queue.Queue, provider: str | None,
                      model: str | None, base_url: str | None, stop_event: threading.Event,
                      concurrency: int | None = None) -> None:
    """Worker thread: runs the coordinator, streams progress lines into ``q``, ends with ('done'|'error'|'stopped', payload)."""
    try:
        from research_gap_agent.llm import build_llm

        s = settings()
        if provider:
            s.llm_provider = provider
        if model:
            s.llm_model = model
            s.model = model
        if base_url:
            s.llm_base_url = base_url
        if concurrency:
            s.llm_concurrency = int(concurrency)
        llm = build_llm(s)
        progress = lambda msg: q.put(("log", msg))  # noqa: E731
        if resume_id:
            coord = Coordinator.resume(resume_id, llm, settings=s, progress=progress, should_stop=stop_event.is_set)
        else:
            coord = Coordinator(options, llm, settings=s, progress=progress, should_stop=stop_event.is_set)
        q.put(("run_id", coord.state.run_id))
        q.put(("log", f"LLM provider: {s.llm_provider} ({getattr(llm, 'model', s.llm_model)})"))
        coord.run()
        q.put(("done", coord.state.run_id))
    except RunCancelled:
        q.put(("stopped", resume_id))
    except Exception as e:  # noqa: BLE001
        from research_gap_agent.llm import LLMConfigError

        if isinstance(e, LLMConfigError):
            q.put(("config_error", str(e)))
        else:
            q.put(("error", f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"))


def test_llm_connection(provider: str, model: str | None, base_url: str | None) -> tuple[bool, str]:
    """One tiny structured call to confirm credentials, workspace and model all work."""
    from pydantic import BaseModel

    from research_gap_agent.llm import LLMConfigError, build_llm

    class Ping(BaseModel):
        ok: bool
        model_seen: str

    try:
        s = settings()
        s.llm_provider = provider
        if model:
            s.llm_model = model
            s.model = model
        if base_url:
            s.llm_base_url = base_url
        llm = build_llm(s)
        out = llm.structured(task="ping", system="Reply with ok=true and the name you believe you are.", user="ping",
                             output_model=Ping, effort="low", max_tokens=200)
        used = llm.call_log[-1].model if llm.call_log else getattr(llm, "model", "?")
        return True, f"Connected. Answered by model `{used}` (it calls itself {out.model_seen})."
    except LLMConfigError as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def start_job(options: RunOptions | None, resume_id: str | None, provider: str | None, model: str | None,
              base_url: str | None, concurrency: int | None = None) -> None:
    stop_event = threading.Event()
    job = {"q": queue.Queue(), "log": [], "run_id": resume_id, "status": "running",
           "started": time.time(), "stage": "starting", "stage_started": time.time(), "stop": stop_event,
           "label": (f"resume {resume_id}" if resume_id else (options.topic[:40] if options else "run"))}
    st.session_state.setdefault("jobs", []).append(job)
    st.session_state["job"] = job
    threading.Thread(target=run_in_background,
                     args=(options, resume_id, job["q"], provider, model, base_url, stop_event, concurrency),
                     daemon=True).start()


def drain(job: dict) -> None:
    """Apply everything the worker thread has queued for this job."""
    try:
        while True:
            kind, payload = job["q"].get_nowait()
            if kind == "log":
                job["log"].append(f"{time.strftime('%H:%M:%S')} {payload}")
                if payload.startswith("[") and payload.endswith("starting"):
                    job["stage"] = payload.split("]")[0].strip("[")
                    job["stage_started"] = time.time()
            elif kind == "run_id":
                job["run_id"] = payload
            elif kind == "done":
                job["status"] = "done"
            elif kind == "stopped":
                job["status"] = "stopped"
                job["run_id"] = job["run_id"] or payload
            elif kind == "error":
                job["status"] = "error"
                job["log"].append(payload)
            elif kind == "config_error":
                job["status"] = "config_error"
                job["config_error"] = payload
    except queue.Empty:
        pass


def job_title(job: dict) -> str:
    stage = STAGE_INFO.get(job["stage"], (job["stage"], ""))[0]
    return f"{job['label']} · {job['status'].replace('_', ' ')} · {stage}"


def apply_theme(name: str) -> None:
    """Persist the chosen theme the same way Streamlit's own Settings menu does, then reload the page."""
    import streamlit.components.v1 as components

    components.html(f"""<script>
      const p = window.parent;
      p.localStorage.setItem(`stActiveTheme-${{p.location.pathname}}-v2`, JSON.stringify("{name}"));
      p.location.reload();
    </script>""", height=0)


def _select_run(rid: str) -> None:
    st.session_state["selected_run"] = rid
    st.session_state["confirm_delete"] = None


def _ask_delete(rid: str | None) -> None:
    st.session_state["confirm_delete"] = rid


def _reuse_settings(rid: str) -> None:
    st.session_state["prefill_run"] = rid


def delete_run(rid: str) -> None:
    import shutil

    shutil.rmtree(settings().runs_dir / rid, ignore_errors=True)
    if st.session_state.get("selected_run") == rid:
        st.session_state["selected_run"] = None
    st.session_state["confirm_delete"] = None


def run_status(rid: str, s: RunState) -> str:
    running = {j["run_id"] for j in st.session_state.get("jobs", []) if j["status"] == "running"}
    n = len(s.stages_completed)
    if rid in running:
        return "Running"
    if n == len(STAGES):
        gaps = len([v for v in s.validated_gaps if not v.rejected])
        return f"Complete · {gaps} gap{'s' if gaps != 1 else ''}"
    if n == 0:
        return "Not started"
    return f"{n} of {len(STAGES)} steps"


def credential_form(form_key: str, fields: list[tuple[str, str, bool, str]], button: str, extra: dict | None = None) -> None:
    with st.form(form_key):
        entered = {}
        for env, label, secret, help_ in fields:
            tip = (help_ + " " if help_ else "") + f"Stored as {env}."
            entered[env] = st.text_input(label, type="password" if secret else "default", help=tip,
                                         placeholder="already set" if os.environ.get(env) else "")
        st.caption("Leave a field blank to keep its current value. Type `clear` to remove a saved value.")
        if st.form_submit_button(button):
            changes = {k: ("" if v.strip().lower() == "clear" else v.strip()) for k, v in entered.items() if v.strip()}
            if changes or extra:
                changes.update(extra or {})
                update_env_file(changes)
                st.success("Saved. The new settings apply immediately.")
                st.rerun()
            else:
                st.info("Nothing entered.")


if st.session_state.get("prefill_run"):
    _rid = st.session_state.pop("prefill_run")
    _sp = settings().runs_dir / _rid / "state.json"
    if _sp.exists():
        prefill_form_from(RunState.model_validate(json.loads(_sp.read_text())))
        st.toast("Settings copied into the New investigation form.")

# ----------------------------------------------------------------------------- sidebar: setup
with st.sidebar:
    st.title("Research Gap Agent")
    st.caption("Finds research gaps in the literature and checks its own findings before showing them to you.")
    t1, t2 = st.columns(2)
    if t1.button(":material/light_mode:", key="theme_light", help="Light theme", width="stretch"):
        apply_theme("Light")
    if t2.button(":material/dark_mode:", key="theme_dark", help="Dark theme", width="stretch"):
        apply_theme("Dark")

    st.subheader("Setup")
    detected = detect_provider()
    ready, why = provider_ready(detected)
    st.markdown(f"**AI model** · {provider_label(detected)} · "
                f"<span class='{'rga-ok' if ready else 'rga-warn'}'>{'ready' if ready else why}</span>", unsafe_allow_html=True)
    with st.expander("AI model settings", expanded=not ready):
        chosen_provider = st.selectbox("Provider", list(PROVIDERS), index=list(PROVIDERS).index(detected), key="sb_provider",
                                       format_func=provider_label,
                                       help="Picked automatically from whichever key you have saved. Anthropic (Claude) gives the best results.")
        ready, why = provider_ready(chosen_provider)
        preset = PROVIDERS[chosen_provider]
        st.caption(f"Default model: `{preset.default_model or 'set one under Advanced settings'}`."
                   + (f" {preset.note}" if preset.note else ""))
        credential_form("provider_cred_form", PROVIDER_FIELDS[chosen_provider], "Save provider keys",
                        extra={"RGA_LLM_PROVIDER": chosen_provider})
        if st.button("Test connection", help="Makes one tiny model call with the saved key to confirm everything works."):
            with st.spinner("Calling the model…"):
                ok_conn, msg = test_llm_connection(chosen_provider, st.session_state.get("f_model") or None,
                                                   st.session_state.get("f_base_url") or None)
            (st.success if ok_conn else st.error)(msg)

    n_keys = sum(1 for env, *_ in SOURCE_FIELDS if os.environ.get(env))
    st.markdown(f"**Databases** · {n_keys} of {len(SOURCE_FIELDS)} optional keys set")
    with st.expander("Database keys"):
        st.caption("All databases work without keys. Your email and an OpenAlex key make searches faster and more reliable.")
        for env, label, _, _ in SOURCE_FIELDS:
            val = os.environ.get(env)
            st.markdown(f"<span title='Stored as {env}'>{label} · "
                        f"<span class='{'rga-ok' if val else 'rga-grey'}'>{'set' if val else 'not set'}</span></span>",
                        unsafe_allow_html=True)
        credential_form("source_cred_form", SOURCE_FIELDS, "Save source keys")
        if not os.environ.get("OPENALEX_API_KEY"):
            mail = os.environ.get("RGA_MAILTO", "")
            st.caption("No OpenAlex key yet. OpenAlex can email you a free one (this sends your email address to OpenAlex).")
            if st.button("Request a free OpenAlex key", disabled=not mail,
                         help="Save your email first" if not mail else f"Emails a confirmation link to {mail}"):
                from research_gap_agent.sources.openalex import request_api_key
                st.info(request_api_key(mail))
    st.caption("Keys are stored in the `.env` file in the project folder, on this computer only.",
               help=f"Full path: {default_env_path()}")

    if st.session_state.get("jobs"):
        st.subheader("Running now")
        for j in st.session_state["jobs"]:
            drain(j)
            stage = STAGE_INFO.get(j["stage"], (j["stage"], ""))[0]
            st.markdown(f"**{j['label']}**  \n<span class='rga-muted'>{j['status'].replace('_', ' ')} · {stage}</span>",
                        unsafe_allow_html=True)

selected_run = st.session_state.get("selected_run")
tab_new, tab_runs, tab_view = st.tabs(["New investigation", "Previous investigations", "Results"])

# ----------------------------------------------------------------------------- new run form
with tab_new:
    for k, v in {"f_provider": chosen_provider, "f_model": settings().llm_model or PROVIDERS[chosen_provider].default_model,
                 "f_from": 2020, "f_to": CURRENT_YEAR, "f_sources": DEFAULT_SOURCES, "f_depth": "Standard",
                 "f_max_papers": 60, "f_per_source": 25, "f_rounds": 2, "f_max_ft": 20, "f_priority": "(none)",
                 "f_goal": "research", "f_existing": "", "f_screen_model": "",
                 "f_concurrency": settings().llm_concurrency}.items():
        st.session_state.setdefault(k, v)

    st.title("New investigation")
    st.markdown('<p class="rga-hero">Enter a topic. The agent searches the literature, proposes research gaps, '
                'tests each one against further searches, and writes a report with verified citations.</p>', unsafe_allow_html=True)
    if not ready:
        st.warning(f"{provider_label(chosen_provider)} has {why}. "
                   "Open **AI model settings** in the sidebar, paste your key and click **Save provider keys**.")

    with st.form("run_form"):
        topic = st.text_input("Research topic", key="f_topic",
                              placeholder="e.g. Securing the Model Context Protocol (MCP) in agentic AI systems",
                              help="One sentence is enough. Be specific: a narrow topic gives sharper gaps.")
        c1, c2, c3 = st.columns([2, 1, 1])
        discipline = c1.text_input("Field or discipline (optional)", key="f_discipline", placeholder="e.g. computer security",
                                   help="Helps the agent choose the right vocabulary and databases.")
        from_year = c2.number_input("From year", min_value=1900, max_value=2100, key="f_from",
                                    help="Earliest publication year to include. 1900 means no limit.")
        to_year = c3.number_input("To year", min_value=1900, max_value=2100, key="f_to",
                                  help="Latest publication year to include. The current year means up to today.")
        st.write("")
        goal = st.radio("What are you looking for?", list(GOALS), format_func=GOALS.get, key="f_goal", horizontal=True,
                        help="Choose 'tool' if the result should be something you build. The agent then also searches GitHub for existing tools.")
        depth = st.radio("How deep should the search go?", list(DEPTHS), key="f_depth", horizontal=True,
                         captions=[d["caption"] for d in DEPTHS.values()])
        sources = st.multiselect("Databases to search", ALL_SOURCES, key="f_sources",
                                 format_func=lambda s: SOURCE_INFO[s][0],
                                 help="\n".join(f"- **{n}**: {d}" for n, d in SOURCE_INFO.values()))
        st.write("")
        with st.expander("Have you already built something, or want to be different from existing work? (optional)"):
            existing = st.text_area("Describe it", key="f_existing", height=110, label_visibility="collapsed",
                                    placeholder="Describe your tool, project or prior work. The agent looks for gaps that would make it "
                                                "unique and checks papers and GitHub for anything that already does it.")
        with st.expander("Advanced settings"):
            st.markdown("**Research preferences**")
            p1, p2 = st.columns(2)
            methodology = p1.text_input("Preferred methodology (optional)", key="f_method",
                                        placeholder="e.g. empirical evaluation, formal verification, user study",
                                        help="Nudges gap generation and the recommendation towards this kind of work.")
            priority = p2.selectbox("Kind of contribution", list(PRIORITIES), format_func=PRIORITIES.get, key="f_priority",
                                    help="Which kind of research you would rather do. Affects ranking.")
            st.markdown("**Exact numbers** (used when depth is *Custom*)")
            a1, a2, a3, a4 = st.columns(4)
            max_papers = a1.number_input("Max papers analysed", 5, 300, key="f_max_papers",
                                         help="How many papers get a full evidence extraction. The main driver of time and cost.")
            per_source = a2.number_input("Results per query per database", 5, 100, key="f_per_source")
            rounds = a3.number_input("Validation rounds", 1, 4, key="f_rounds",
                                     help="How many times a narrowed gap is re-checked against the literature.")
            max_ft = a4.number_input("Open-access full texts to fetch", 0, 100, key="f_max_ft",
                                     help="PDFs to download and read in full. 0 uses abstracts only.")
            st.markdown("**AI model for this run**")
            b1, b2, b3 = st.columns(3)
            provider = b1.selectbox("LLM provider", list(PROVIDERS), key="f_provider", format_func=provider_label)
            model = b2.text_input("Model", key="f_model", help="Leave blank for the provider's default model.")
            base_url = b3.text_input("Server address (self-hosted only)", key="f_base_url")
            st.markdown("**Speed**")
            c1_, c2_ = st.columns(2)
            screen_model = c1_.text_input("Model for screening papers", key="f_screen_model",
                                          placeholder="blank = provider's fast model",
                                          help="Deciding whether a paper is on-topic is a simple task, so a cheaper model is used "
                                               "by default (claude-haiku-5-5 for Anthropic). Type 'main' to use the main model.")
            concurrency = c2_.number_input("Model calls at the same time", 1, 8, key="f_concurrency",
                                           help="Screening batches, evidence extraction and gap validations run in parallel. "
                                                "Lower this if your API account hits rate limits.")
        st.write("")
        submitted = st.form_submit_button("Start investigation", type="primary", width="stretch")
    st.caption("A run usually takes between 30 minutes and a few hours depending on depth. You can close this page; "
               "progress is saved after every step and incomplete runs can be resumed.")

    if submitted:
        ok, why = provider_ready(provider)
        if not topic.strip():
            st.error("Please enter a research topic.")
        elif not sources:
            st.error("Select at least one database.")
        elif not ok and not base_url:
            st.error(f"{provider_label(provider)} is not ready: {why}. Add the key in the sidebar under **AI model settings**.")
        else:
            model_name = model.strip() or PROVIDERS[provider].default_model
            nums = DEPTHS[depth] if depth != "Custom" else {"max_papers": int(max_papers), "per_source": int(per_source),
                                                             "rounds": int(rounds), "max_ft": int(max_ft)}
            opts = RunOptions(topic=topic.strip(), discipline=discipline or None,
                              from_year=int(from_year) if int(from_year) > 1900 else None,
                              to_year=int(to_year) if int(to_year) < CURRENT_YEAR else None,
                              methodology_preference=methodology or None,
                              priority=None if priority == "(none)" else priority, sources=sources,
                              goal=goal, existing_work=existing.strip() or None,
                              per_source_limit=nums["per_source"], max_papers_to_analyze=nums["max_papers"],
                              validation_rounds=nums["rounds"], max_full_text=nums["max_ft"], provider=provider,
                              model=model_name, screen_model=(screen_model.strip() or None))
            start_job(opts, None, provider, model_name, base_url.strip() or None, int(concurrency))

    jobs = st.session_state.get("jobs", [])
    for j in jobs:
        drain(j)
    job = st.session_state.get("job")
    if len(jobs) > 1:
        running = [j for j in jobs if j["status"] == "running"]
        st.caption(f"{len(running)} of {len(jobs)} investigations started in this session are still running.")
        job = st.radio("Watch", options=jobs, format_func=job_title,
                       index=jobs.index(job) if job in jobs else len(jobs) - 1, key="watch_job")
        st.session_state["job"] = job

    if not jobs:
        with st.expander("About this tool"):
            st.markdown("**The twelve steps of a run**")
            for i, (sid, (name, desc)) in enumerate(STAGE_INFO.items(), 1):
                st.markdown(f"{i}. **{name}.** {desc}.")
            st.markdown("""
**What you get at the end**

- A readable report: the state of the field, the research gaps that survived validation, a ranking, and a concrete plan for the best one.
- A list of every paper found, with links, and a log of every search that was run.
- Reference files for Zotero, Mendeley or EndNote.

**What makes it trustworthy**

Every claim points at a real paper. The agent removes any citation it cannot find in its own corpus, checks DOIs,
flags retracted papers, and tries to disprove each gap before reporting it. If a search failed, the report says so.
""")

    if job:
        st.divider()
        if job["status"] == "running" and not job["stop"].is_set():
            st.subheader(f"Running: {job['label']}")
        else:
            st.subheader(job["label"])
        if job["status"] == "running" and not job["stop"].is_set():
            if st.button("Stop after the current step", help="Finishes the model call in flight, saves progress, and stops. "
                                                             "You can resume later from the Results tab."):
                job["stop"].set()
                job["log"].append(f"{time.strftime('%H:%M:%S')} stop requested; finishing the current step…")
        elif job["status"] == "running":
            st.info("Stopping after the current step. A long model call can take a few minutes to finish.")
        bar = st.empty()
        head = st.empty()
        with st.expander("Technical log", expanded=job["status"] in ("error", "config_error")):
            log_box = st.empty()
        while True:
            for j in jobs:
                drain(j)
            now = time.time()
            stage_idx = STAGES.index(job["stage"]) + 1 if job["stage"] in STAGES else 0
            name, desc = STAGE_INFO.get(job["stage"], ("Starting", "Setting up the run"))
            if job["status"] == "running":
                bar.progress(stage_idx / len(STAGES), text=f"Step {stage_idx} of {len(STAGES)}: {name}")
                latest = job["log"][-1][9:] if job["log"] else "starting…"
                head.markdown(f"{desc}.  \n<span class='rga-muted'>Running for {fmt_duration(now - job['started'])} · "
                              f"this step {fmt_duration(now - job['stage_started'])} · latest: {latest[:140]}</span>",
                              unsafe_allow_html=True)
            log_box.code("\n".join(job["log"][-80:]) or "starting…")
            if job["status"] != "running":
                break
            time.sleep(1.0)
        if job["status"] == "done":
            bar.progress(1.0, text="Finished")
            head.success(f"Report ready after {fmt_duration(time.time() - job['started'])}. Open the **Results** tab to read it.")
            selected_run = job["run_id"]
        elif job["status"] == "stopped":
            bar.empty()
            head.warning("Stopped at a checkpoint. Open the run in the **Results** tab and click **Resume** to continue later.")
        elif job["status"] == "config_error":
            bar.empty()
            head.error("**The model rejected the call because of account configuration.**\n\n" + job["config_error"]
                       + "\n\nFix it in the sidebar under **AI model settings**, click **Test connection**, then resume this run from the Results tab.")
        elif job["status"] == "error":
            bar.empty()
            head.error("The run failed. Progress was saved, so you can resume from the Results tab. Details are in the technical log.")
        others = [j for j in jobs if j is not job and j["status"] == "running"]
        if others:
            st.info("Still running in the background: " + ", ".join(j["label"] for j in others)
                    + ". Pick it under **Watch** above to follow it.")

# ----------------------------------------------------------------------------- previous investigations tab
with tab_runs:
    runs = list_runs(settings().runs_dir)
    st.title("Previous investigations")
    if not runs:
        st.info("No saved investigations yet. Every run is saved automatically and listed here.")
    else:
        complete_n = sum(1 for _, s in runs if len(s.stages_completed) == len(STAGES))
        st.markdown(f'<p class="rga-hero">{len(runs)} saved · {complete_n} complete · {len(runs) - complete_n} incomplete. '
                    'Incomplete runs continue from their last finished step with <b>Resume</b>.</p>', unsafe_allow_html=True)
        f1, f2 = st.columns([3, 1])
        query = f1.text_input("Search", key="run_filter", placeholder="Search by topic…", label_visibility="collapsed")
        show = f2.selectbox("Show", ["All", "Complete", "Incomplete"], key="run_show", label_visibility="collapsed")
        shown = [(r, s) for r, s in runs if query.lower() in s.options.topic.lower()]
        if show == "Complete":
            shown = [(r, s) for r, s in shown if len(s.stages_completed) == len(STAGES)]
        elif show == "Incomplete":
            shown = [(r, s) for r, s in shown if len(s.stages_completed) < len(STAGES)]
        if not shown:
            st.caption("No run matches.")
        limit = len(shown) if st.session_state.get("show_all_runs") or len(shown) <= 10 else 10
        for rid, s in shown[:limit]:
            is_open = selected_run == rid
            o = s.options
            with st.container(border=True):
                incomplete = len(s.stages_completed) < len(STAGES)
                running = rid in {j["run_id"] for j in st.session_state.get("jobs", []) if j["status"] == "running"}
                info, b_resume, b_open, b_again, b_del = st.columns([6, 1.3, 1.1, 1.3, 1.1])
                info.markdown(f"**{o.topic}**  \n<span class='rga-muted'>{s.created_at.strftime('%d %b %Y, %H:%M')} · "
                              f"{run_status(rid, s)} · {len(s.papers)} papers · {provider_label(o.provider)}</span>",
                              unsafe_allow_html=True)
                if incomplete and not running:
                    nxt = next((x for x in STAGES if x not in s.stages_completed), None)
                    if b_resume.button("Resume", key=f"resume_{rid}", type="primary", width="stretch",
                                       help=f"Continue from the next step: {STAGE_INFO[nxt][0] if nxt else 'report'}"):
                        _select_run(rid)
                        start_job(None, rid, st.session_state.get("f_provider"), st.session_state.get("f_model"),
                                  st.session_state.get("f_base_url") or None)
                        st.rerun()
                b_open.button("Opened" if is_open else "Open", key=f"open_{rid}", width="stretch", disabled=is_open,
                              on_click=_select_run, args=(rid,), help="Show this run in the Results tab")
                b_again.button("Run again", key=f"again_{rid}", width="stretch", on_click=_reuse_settings, args=(rid,),
                               help="Copy this run's topic and settings into the New investigation form")
                if st.session_state.get("confirm_delete") == rid:
                    st.warning("Delete this run and all its files? This cannot be undone.")
                    d1, d2, _ = st.columns([1, 1, 6])
                    if d1.button("Delete", key=f"confirm_{rid}", type="primary", width="stretch"):
                        delete_run(rid)
                        st.rerun()
                    d2.button("Keep", key=f"keep_{rid}", width="stretch", on_click=_ask_delete, args=(None,))
                else:
                    b_del.button("Delete", key=f"del_{rid}", width="stretch", on_click=_ask_delete, args=(rid,))
        if limit < len(shown):
            st.button(f"Show all {len(shown)} runs", key="show_all_runs_btn",
                      on_click=lambda: st.session_state.update(show_all_runs=True))

# ----------------------------------------------------------------------------- results tab
with tab_view:
    run_id = selected_run or (st.session_state.get("job") or {}).get("run_id")
    if not run_id:
        st.title("Results")
        st.info("Nothing to show yet. Start an investigation, or open one from **Previous investigations**.")
    else:
        run_dir = settings().runs_dir / run_id
        sp = run_dir / "state.json"
        if not sp.exists():
            st.warning("The saved data for this run could not be found.")
        else:
            state = RunState.model_validate(json.loads(sp.read_text()))
            done = set(state.stages_completed)
            complete = len(done) == len(STAGES)
            o = state.options
            st.title(o.topic)
            st.markdown(f"<span class='rga-muted'>Started {state.created_at.strftime('%d %b %Y, %H:%M')} · "
                        f"{provider_label(o.provider)} · {o.model} · "
                        f"{', '.join(SOURCE_INFO.get(s, (s, ''))[0] for s in o.sources)}"
                        + (f" · {o.from_year or 'any year'} to {o.to_year or 'today'}") + "</span>", unsafe_allow_html=True)

            if not complete:
                next_stage = next((s for s in STAGES if s not in done), None)
                st.warning(f"This run is incomplete: {len(done)} of {len(STAGES)} steps finished. "
                           + (f"Next step: **{STAGE_INFO[next_stage][0]}**." if next_stage else ""))
                st.progress(len(done) / len(STAGES))
                if st.button("Resume this run", type="primary"):
                    start_job(None, run_id, st.session_state.get("f_provider"), st.session_state.get("f_model"),
                              st.session_state.get("f_base_url") or None, st.session_state.get("f_concurrency"))
                    st.rerun()

            surviving = [v for v in state.validated_gaps if not v.rejected]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Papers found", len(state.papers), help=f"{state.n_records_retrieved} records retrieved before removing duplicates")
            m2.metric("Analysed in depth", len(state.evidence_rows))
            m3.metric("Full texts read", len(state.full_text_analyzed_ids()))
            m4.metric("Validated gaps", len(surviving), help=f"{len(state.validated_gaps) - len(surviving)} candidate(s) rejected as already addressed")

            report = run_dir / "report.md"
            html_path = run_dir / "report.html"
            if report.exists():
                d1, d2, d3 = st.columns([1, 1, 2])
                if html_path.exists():
                    d1.download_button("Download report", data=html_path.read_bytes(), file_name=f"{run_id}_report.html",
                                       type="primary", width="stretch", help="Self-contained web page; opens in any browser.")
                d2.download_button("Report as Markdown", data=report.read_bytes(), file_name=f"{run_id}_report.md",
                                   width="stretch")
                with d3.popover("More files", width="stretch"):
                    st.caption("Reference files import directly into Zotero, Mendeley and EndNote.")
                    for name in EXPORT_FILES:
                        path = run_dir / name
                        if path.exists() and name not in ("report.html", "report.md"):
                            st.download_button(EXPORT_LABELS[name], data=path.read_bytes(), file_name=f"{run_id}_{name}",
                                               width="stretch")

            if state.validated_gaps:
                sub_gaps, sub_report, sub_papers, sub_log = st.tabs(["Research gaps", "Full report", "Papers", "Search log"])
            else:
                sub_gaps = None
                sub_report, sub_papers, sub_log = st.tabs(["Report", "Papers", "Search log"])

            if sub_gaps is not None:
                with sub_gaps:
                    rank_of = {r.gap_id: r for r in state.ranking}
                    ordered = sorted(surviving, key=lambda v: rank_of[v.gap.id].rank if v.gap.id in rank_of else 999)
                    rec = state.recommendation
                    if rec:
                        with st.container(border=True):
                            st.markdown(f"**Recommended direction**  \n{rec.proposed_title}")
                            st.markdown(rec.problem_statement)
                            with st.expander("Why this one, and how to approach it"):
                                st.markdown(f"**Motivation.** {rec.motivation}\n\n**Aim.** {rec.aim}\n\n**Method.** {rec.methodology}")
                                if rec.research_questions:
                                    st.markdown("**Research questions**\n" + "\n".join(f"- {q}" for q in rec.research_questions))
                                if rec.expected_contributions:
                                    st.markdown("**Expected contributions**\n" + "\n".join(f"- {c}" for c in rec.expected_contributions))
                    st.markdown(f"### Validated gaps ({len(ordered)})")
                    st.caption("Ranked by evidence strength, novelty, feasibility and impact. Hover over a status for its meaning.")
                    for v in ordered:
                        r = rank_of.get(v.gap.id)
                        with st.container(border=True):
                            st.markdown(
                                f"**{r.rank if r else '–'}. {v.gap.title}**  \n"
                                f"<span class='rga-muted'>{status_text(v.status.value)} · "
                                f"{CATEGORY_LABELS.get(v.gap.category.value, v.gap.category.value)} · "
                                f"confidence {v.confidence} · feasibility {v.feasibility.overall}</span>",
                                unsafe_allow_html=True)
                            st.markdown(f"**What is missing.** {v.gap.what_is_missing}")
                            with st.expander("Details and evidence"):
                                st.markdown(f"**What is already established.** {v.gap.what_is_established}")
                                st.markdown(f"**Why it matters.** {v.gap.why_it_matters}")
                                st.markdown(f"**Possible contribution.** {v.gap.potential_contribution}")
                                st.markdown(f"**How confident the agent is.** {v.confidence_explanation}")
                                if r:
                                    st.markdown(f"**Main risk.** {r.main_risk}")
                                sup = [state.papers[p] for p in v.gap.supporting_paper_ids if p in state.papers]
                                if sup:
                                    st.markdown("**Supporting papers**\n" + "\n".join(
                                        f"- [{p.title}]({p.best_url()}) ({p.year})" if p.best_url() else f"- {p.title} ({p.year})"
                                        for p in sup))
                                if v.counter_evidence:
                                    st.markdown("**Closest existing work found during validation**\n" + "\n".join(
                                        f"- {state.papers[c.paper_id].title if c.paper_id in state.papers else c.paper_id} "
                                        f"({c.relationship.replace('_', ' ')}): {c.note}" for c in v.counter_evidence))
                    rejected = [v for v in state.validated_gaps if v.rejected]
                    if rejected:
                        with st.expander(f"{len(rejected)} candidate gap{'s' if len(rejected) != 1 else ''} rejected because existing work already covers them"):
                            for v in rejected:
                                st.markdown(f"**{v.gap.title}** · {status_text(v.status.value)}  \n{v.rejection_reason or ''}",
                                            unsafe_allow_html=True)
                                if v.counter_evidence:
                                    st.markdown("\n".join(
                                        f"- {state.papers[c.paper_id].title if c.paper_id in state.papers else c.paper_id}: {c.note}"
                                        for c in v.counter_evidence))
                    if state.rejected_ideas:
                        with st.expander(f"{len(state.rejected_ideas)} idea{'s' if len(state.rejected_ideas) != 1 else ''} considered and set aside before validation"):
                            for idea in state.rejected_ideas:
                                st.markdown(f"- **{idea.idea}** — {idea.why_not}")

            with sub_report:
                if html_path.exists():
                    import streamlit.components.v1 as components
                    components.html(html_path.read_text(), height=2400, scrolling=True)
                elif report.exists():
                    st.markdown(report.read_text(), unsafe_allow_html=True)
                else:
                    st.info("The report is written in the last step. Resume the run to finish it.")

            with sub_papers:
                query = st.text_input("Filter papers", placeholder="Type part of a title…", label_visibility="collapsed")
                rows = [{"Title": p.title, "Year": p.year, "Link": p.best_url(), "Type": p.publication_type.replace("_", " "),
                         "Included": bool(p.screening and p.screening.include), "Full text": p.has_full_text,
                         "Retracted": bool(p.is_retracted), "Found via": ", ".join(SOURCE_INFO.get(s, (s, ""))[0] for s in p.sources),
                         "Id": p.id}
                        for p in sorted(state.papers.values(), key=lambda p: p.id)]
                if query:
                    rows = [r for r in rows if query.lower() in (r["Title"] or "").lower()]
                st.caption(f"{len(rows)} papers")
                st.dataframe(rows, width="stretch", hide_index=True,
                             column_config={"Link": st.column_config.LinkColumn("Link", display_text="open"),
                                            "Title": st.column_config.TextColumn("Title", width="large")})

            with sub_log:
                st.caption("Every search the agent ran, in order. 'Validation' searches are the ones that tried to disprove a gap.")
                st.dataframe([{"Database": SOURCE_INFO.get(s.source, (s.source, ""))[0], "Query": s.query,
                               "Purpose": s.purpose.replace("_", " "), "Status": s.status.replace("_", " "),
                               "Results": s.n_retrieved, "New papers": s.n_new_unique}
                              for s in state.search_records], width="stretch", hide_index=True)
