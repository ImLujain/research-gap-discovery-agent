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

STATUS_INFO: dict[str, tuple[str, str, str]] = {  # value -> (label, explanation, colour)
    "limited_evidence_found": ("Open gap", "Counter-searches found little or no work that fills this gap", "#16A34A"),
    "preliminary_evidence": ("Early work only", "Only preliminary studies exist so far", "#16A34A"),
    "conflicting_evidence": ("Conflicting findings", "Existing studies disagree with each other", "#D97706"),
    "partially_addressed": ("Partly addressed", "Some aspects are already covered; the gap was narrowed", "#D97706"),
    "cannot_validate": ("Could not verify", "The validation searches failed, so this is unconfirmed", "#6B7280"),
    "addressed": ("Already addressed", "Existing work already covers this", "#DC2626"),
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
             "f_depth", "f_max_papers", "f_per_source", "f_rounds", "f_max_ft", "f_provider", "f_model", "f_base_url"]

st.markdown("""
<style>
  .block-container { padding-top: 2rem; max-width: 1180px; }
  h1 { font-weight: 700; letter-spacing: -0.01em; }
  .rga-hero { color: #4B5563; font-size: 1.05rem; margin-top: -0.4rem; margin-bottom: 1.2rem; }
  .rga-badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
               color: white; margin-right: 6px; vertical-align: middle; }
  .rga-pill { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.76rem; background: #EEF2FF;
              color: #3730A3; margin-right: 6px; vertical-align: middle; }
  .rga-muted { color: #6B7280; font-size: 0.86rem; }
  .rga-rank { font-size: 1.6rem; font-weight: 700; color: #2563EB; line-height: 1; }
  div[data-testid="stMetric"] { background: #F3F5F9; border-radius: 10px; padding: 0.8rem 1rem; }
  [data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0.3rem; }
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


def badge(text: str, colour: str) -> str:
    return f'<span class="rga-badge" style="background:{colour}">{text}</span>'


def status_badge(value: str) -> str:
    label, _, colour = STATUS_INFO.get(value, (value.replace("_", " "), "", "#6B7280"))
    return badge(label, colour)


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
        "f_base_url": os.environ.get("RGA_LLM_BASE_URL", ""),
    })


def run_in_background(options: RunOptions | None, resume_id: str | None, q: queue.Queue, provider: str | None,
                      model: str | None, base_url: str | None, stop_event: threading.Event) -> None:
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
              base_url: str | None) -> None:
    stop_event = threading.Event()
    job = {"q": queue.Queue(), "log": [], "run_id": resume_id, "status": "running",
           "started": time.time(), "stage": "starting", "stage_started": time.time(), "stop": stop_event,
           "label": (f"resume {resume_id}" if resume_id else (options.topic[:40] if options else "run"))}
    st.session_state.setdefault("jobs", []).append(job)
    st.session_state["job"] = job
    threading.Thread(target=run_in_background,
                     args=(options, resume_id, job["q"], provider, model, base_url, stop_event),
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


# ----------------------------------------------------------------------------- sidebar: setup
with st.sidebar:
    st.title("🔬 Research Gap Agent")
    st.caption("Finds research gaps in the literature and checks its own findings before showing them to you.")

    st.subheader("1. AI model")
    detected = detect_provider()
    chosen_provider = st.selectbox("Provider", list(PROVIDERS), index=list(PROVIDERS).index(detected), key="sb_provider",
                                   format_func=provider_label,
                                   help="Picked automatically from whichever key you have saved. Anthropic (Claude) gives the best results.")
    ready, why = provider_ready(chosen_provider)
    preset = PROVIDERS[chosen_provider]
    if ready:
        st.success(f"{provider_label(chosen_provider)} is ready. Model: `{preset.default_model or 'set below'}`.", icon="✅")
    else:
        st.warning(f"{provider_label(chosen_provider)}: {why}. Add it below.", icon="⚠️")
    with st.expander("Add or change the model key", expanded=not ready):
        credential_form("provider_cred_form", PROVIDER_FIELDS[chosen_provider], "Save provider keys",
                        extra={"RGA_LLM_PROVIDER": chosen_provider})
    if st.button("Test connection", help="Makes one tiny model call with the saved key to confirm everything works."):
        with st.spinner("Calling the model…"):
            ok_conn, msg = test_llm_connection(chosen_provider, st.session_state.get("f_model") or None,
                                               st.session_state.get("f_base_url") or None)
        (st.success if ok_conn else st.error)(msg)

    st.subheader("2. Literature databases")
    st.caption("All work without keys. The first two settings make searches faster and more reliable.")
    for env, label, _, _ in SOURCE_FIELDS:
        val = os.environ.get(env)
        st.markdown(f"<span title='Stored as {env}'>{'✅' if val else '▫️'} {label}"
                    f"{'' if val else ' <span class=rga-muted>(not set)</span>'}</span>", unsafe_allow_html=True)
    with st.expander("Add or change database keys"):
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
            icon = {"running": "🟢", "done": "✅", "stopped": "⏹"}.get(j["status"], "⚠️")
            stage = STAGE_INFO.get(j["stage"], (j["stage"], ""))[0]
            st.markdown(f"{icon} **{j['label']}**  \n<span class='rga-muted'>{stage}</span>", unsafe_allow_html=True)

    st.subheader("3. Previous investigations")
    runs = list_runs(settings().runs_dir)
    labels = {rid: run_label(rid, s) for rid, s in runs}
    selected_run = st.selectbox("Open a run", options=[""] + [r for r, _ in runs],
                                format_func=lambda r: labels.get(r, "Choose one…"), key="sb_run",
                                help="Opens the run in the Results tab and copies its settings into the form, so you can run it again with changes.")
    if selected_run and st.session_state.get("prefilled_from") != selected_run:
        prefill_form_from(dict(runs)[selected_run])
        st.session_state["prefilled_from"] = selected_run
        st.caption("Form filled with this run's settings.")

tab_new, tab_view = st.tabs(["🔎  New investigation", "📄  Results"])

# ----------------------------------------------------------------------------- new run form
with tab_new:
    for k, v in {"f_provider": chosen_provider, "f_model": settings().llm_model or PROVIDERS[chosen_provider].default_model,
                 "f_from": 2020, "f_to": CURRENT_YEAR, "f_sources": DEFAULT_SOURCES, "f_depth": "Standard",
                 "f_max_papers": 60, "f_per_source": 25, "f_rounds": 2, "f_max_ft": 20, "f_priority": "(none)",
                 "f_goal": "research", "f_existing": ""}.items():
        st.session_state.setdefault(k, v)

    st.title("What do you want to investigate?")
    st.markdown('<p class="rga-hero">Describe a topic. The agent reads the literature, proposes research gaps, '
                'tries to disprove each one, and writes an evidence-backed report. A run usually takes '
                'between 30 minutes and a few hours depending on depth.</p>', unsafe_allow_html=True)
    if not ready:
        st.warning(f"**One thing before your first run:** {provider_label(chosen_provider)} has {why}. "
                   "Open **1. AI model** in the sidebar, paste your key and click **Save provider keys**.", icon="🔑")

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

        goal = st.radio("What are you looking for?", list(GOALS), format_func=GOALS.get, key="f_goal", horizontal=True,
                        help="Choose 'tool' if the result should be something you build. The agent then also searches GitHub for existing tools.")
        existing = st.text_area("Anything you have already built, or want to be different from? (optional)", key="f_existing",
                                height=90, placeholder="Describe your tool, project or prior work. The agent looks for gaps that would make it "
                                                       "unique and checks papers and GitHub for anything that already does it.")

        depth = st.radio("How deep should the search go?", list(DEPTHS), key="f_depth", horizontal=True,
                         captions=[d["caption"] for d in DEPTHS.values()])
        sources = st.multiselect("Databases to search", ALL_SOURCES, key="f_sources",
                                 format_func=lambda s: SOURCE_INFO[s][0],
                                 help="\n".join(f"- **{n}**: {d}" for n, d in SOURCE_INFO.values()))

        with st.expander("Advanced settings"):
            st.caption("Research preferences")
            p1, p2 = st.columns(2)
            methodology = p1.text_input("Preferred methodology (optional)", key="f_method",
                                        placeholder="e.g. empirical evaluation, formal verification, user study",
                                        help="Nudges gap generation and the recommendation towards this kind of work.")
            priority = p2.selectbox("Kind of contribution", list(PRIORITIES), format_func=PRIORITIES.get, key="f_priority",
                                    help="Which kind of research you would rather do. Affects ranking.")
            st.caption("Exact numbers (used when depth is **Custom**)")
            a1, a2, a3, a4 = st.columns(4)
            max_papers = a1.number_input("Max papers analysed", 5, 300, key="f_max_papers",
                                         help="How many papers get a full evidence extraction. The main driver of time and cost.")
            per_source = a2.number_input("Results per query per database", 5, 100, key="f_per_source")
            rounds = a3.number_input("Validation rounds", 1, 4, key="f_rounds",
                                     help="How many times a narrowed gap is re-checked against the literature.")
            max_ft = a4.number_input("Open-access full texts to fetch", 0, 100, key="f_max_ft",
                                     help="PDFs to download and read in full. 0 uses abstracts only.")
            st.caption("AI model for this run")
            b1, b2, b3 = st.columns(3)
            provider = b1.selectbox("LLM provider", list(PROVIDERS), key="f_provider", format_func=provider_label)
            model = b2.text_input("Model", key="f_model", help="Leave blank for the provider's default model.")
            base_url = b3.text_input("Server address (self-hosted only)", key="f_base_url")
        submitted = st.form_submit_button("Start investigation", type="primary", width="stretch")

    if submitted:
        ok, why = provider_ready(provider)
        if not topic.strip():
            st.error("Please enter a research topic.")
        elif not sources:
            st.error("Select at least one database.")
        elif not ok and not base_url:
            st.error(f"{provider_label(provider)} is not ready: {why}. Add the key in the sidebar under **1. AI model**.")
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
                              model=model_name)
            start_job(opts, None, provider, model_name, base_url.strip() or None)

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
        with st.expander("How does it work, and what will I get?"):
            st.markdown("**The agent works in twelve steps, and you can watch each one:**")
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
            if st.button("⏹ Stop after the current step", help="Finishes the model call in flight, saves progress, and stops. "
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
            head.success(f"Report ready after {fmt_duration(time.time() - job['started'])}. Open the **Results** tab to read it.", icon="🎉")
            selected_run = job["run_id"]
        elif job["status"] == "stopped":
            bar.empty()
            head.warning("Stopped at a checkpoint. Open the run in the **Results** tab and click **Resume** to continue later.", icon="⏹")
        elif job["status"] == "config_error":
            bar.empty()
            head.error("**The model rejected the call because of account configuration.**\n\n" + job["config_error"]
                       + "\n\nFix it in the sidebar under **1. AI model**, click **Test connection**, then resume this run from the Results tab.")
        elif job["status"] == "error":
            bar.empty()
            head.error("The run failed. Progress was saved, so you can resume from the Results tab. Details are in the technical log.")
        others = [j for j in jobs if j is not job and j["status"] == "running"]
        if others:
            st.info("Still running in the background: " + ", ".join(j["label"] for j in others)
                    + ". Pick it under **Watch** above to follow it.")

# ----------------------------------------------------------------------------- results tab
with tab_view:
    run_id = selected_run or (st.session_state.get("job") or {}).get("run_id")
    if not run_id:
        st.title("Results")
        st.info("Nothing to show yet. Start an investigation in the first tab, or open a previous one from the sidebar "
                "under **3. Previous investigations**.", icon="💡")
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
                           + (f"Next step: **{STAGE_INFO[next_stage][0]}**." if next_stage else ""), icon="⏸")
                st.progress(len(done) / len(STAGES))
                if st.button("▶ Resume this run", type="primary"):
                    start_job(None, run_id, st.session_state.get("f_provider"), st.session_state.get("f_model"),
                              st.session_state.get("f_base_url") or None)
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
                    d1.download_button("⬇ Download report", data=html_path.read_bytes(), file_name=f"{run_id}_report.html",
                                       type="primary", width="stretch", help="Self-contained web page; opens in any browser.")
                d2.download_button("⬇ Report as Markdown", data=report.read_bytes(), file_name=f"{run_id}_report.md",
                                   width="stretch")
                with d3.popover("More files", width="stretch"):
                    st.caption("Reference files import directly into Zotero, Mendeley and EndNote.")
                    for name in EXPORT_FILES:
                        path = run_dir / name
                        if path.exists() and name not in ("report.html", "report.md"):
                            st.download_button(f"⬇ {EXPORT_LABELS[name]}", data=path.read_bytes(), file_name=f"{run_id}_{name}",
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
                            st.markdown(f"{badge('Recommended next step', '#2563EB')} **{rec.proposed_title}**", unsafe_allow_html=True)
                            st.markdown(rec.problem_statement)
                            with st.expander("Why this one, and how to approach it"):
                                st.markdown(f"**Motivation.** {rec.motivation}\n\n**Aim.** {rec.aim}\n\n**Method.** {rec.methodology}")
                                if rec.research_questions:
                                    st.markdown("**Research questions**\n" + "\n".join(f"- {q}" for q in rec.research_questions))
                                if rec.expected_contributions:
                                    st.markdown("**Expected contributions**\n" + "\n".join(f"- {c}" for c in rec.expected_contributions))
                    st.markdown(f"#### {len(ordered)} research gap{'s' if len(ordered) != 1 else ''} survived validation")
                    st.caption("Ranked by evidence strength, novelty, feasibility and impact. Hover over a status for what it means.")
                    for v in ordered:
                        r = rank_of.get(v.gap.id)
                        with st.container(border=True):
                            left, right = st.columns([1, 14])
                            left.markdown(f"<div class='rga-rank'>#{r.rank if r else '–'}</div>", unsafe_allow_html=True)
                            _, expl, _ = STATUS_INFO.get(v.status.value, ("", "", ""))
                            right.markdown(
                                f"**{v.gap.title}**  \n"
                                f"<span title='{expl}'>{status_badge(v.status.value)}</span>"
                                f"<span class='rga-pill'>{CATEGORY_LABELS.get(v.gap.category.value, v.gap.category.value)}</span>"
                                f"<span class='rga-pill'>confidence: {v.confidence}</span>"
                                f"<span class='rga-pill'>feasibility: {v.feasibility.overall}</span>",
                                unsafe_allow_html=True)
                            right.markdown(f"**What is missing.** {v.gap.what_is_missing}")
                            with right.expander("Details and evidence"):
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
                                st.markdown(f"{status_badge(v.status.value)} **{v.gap.title}**  \n{v.rejection_reason or ''}",
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
