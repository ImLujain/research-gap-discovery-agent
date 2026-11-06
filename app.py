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
from pathlib import Path

import streamlit as st

from research_gap_agent.config import Settings, default_env_path, load_env_file, update_env_file
from research_gap_agent.coordinator import STAGES, Coordinator, RunCancelled
from research_gap_agent.models import RunOptions, RunState
from research_gap_agent.providers import PROVIDERS, detect_provider, resolve_api_key

st.set_page_config(page_title="Research Gap Discovery Agent", page_icon="🔬", layout="wide")
load_env_file()

ALL_SOURCES = ["openalex", "semantic_scholar", "crossref", "arxiv", "pubmed", "google_scholar", "github"]
GOALS = {"research": "Research contribution (study, method, theory)", "tool": "Technical tool or artefact (also searches GitHub)"}
EXPORT_FILES = ["report.html", "report.md", "appendix.md", "references.bib", "references.ris", "corpus.csv", "state.json"]
PRIORITIES = ["(none)", "theoretical", "experimental", "applied", "interdisciplinary"]

# Credential fields per provider (name, is_secret, help)
PROVIDER_FIELDS: dict[str, list[tuple[str, bool, str]]] = {
    "anthropic": [("ANTHROPIC_API_KEY", True, "Console API key. Max/Ultra/Team: link a Console org in claude.ai > Settings > Billing, then create the key at platform.claude.com."),
                  ("ANTHROPIC_WORKSPACE_ID", False, "Only if the API says the key is not scoped to a workspace: Console > Settings > Workspaces > the workspace id (wrkspc_...).")],
    "openai": [("OPENAI_API_KEY", True, "platform.openai.com > API keys")],
    "azure": [("AZURE_OPENAI_API_KEY", True, "Azure portal > your OpenAI resource > Keys"),
              ("AZURE_OPENAI_ENDPOINT", False, "https://<resource>.openai.azure.com/"),
              ("AZURE_OPENAI_API_VERSION", False, "e.g. 2024-10-21")],
    "gemini": [("GEMINI_API_KEY", True, "aistudio.google.com > Get API key")],
    "mistral": [("MISTRAL_API_KEY", True, "console.mistral.ai")],
    "groq": [("GROQ_API_KEY", True, "console.groq.com")],
    "together": [("TOGETHER_API_KEY", True, "api.together.ai")],
    "openrouter": [("OPENROUTER_API_KEY", True, "openrouter.ai > Keys")],
    "deepseek": [("DEEPSEEK_API_KEY", True, "platform.deepseek.com")],
    "xai": [("XAI_API_KEY", True, "console.x.ai")],
    "ollama": [("RGA_LLM_BASE_URL", False, "Leave blank for http://localhost:11434/v1")],
    "lmstudio": [("RGA_LLM_BASE_URL", False, "Leave blank for http://localhost:1234/v1")],
    "openai-compatible": [("RGA_LLM_BASE_URL", False, "Base URL of the server, e.g. http://host:8000/v1"),
                          ("RGA_LLM_API_KEY", True, "API key if the server needs one")],
}
SOURCE_FIELDS = [
    ("RGA_MAILTO", False, "Recommended. Your email: polite-pool access for OpenAlex/Crossref and Unpaywall full text."),
    ("OPENALEX_API_KEY", True, "Recommended. Without it OpenAlex counts against a free daily budget shared by your whole network IP; when that runs out the source is skipped until midnight UTC (free key)."),
    ("S2_API_KEY", True, "Optional. Semantic Scholar is heavily rate-limited without it (free key)."),
    ("SERPAPI_KEY", True, "Optional. Enables the google_scholar source (paid)."),
    ("NCBI_API_KEY", True, "Optional. Higher PubMed rate limit."),
    ("GITHUB_TOKEN", True, "Optional. Raises the GitHub repository search limit from 10 to 30 requests per minute (github source)."),
]
FORM_KEYS = ["f_topic", "f_discipline", "f_from", "f_to", "f_method", "f_priority", "f_goal", "f_existing", "f_sources", "f_max_papers",
             "f_per_source", "f_rounds", "f_max_ft", "f_provider", "f_model", "f_base_url"]


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


def masked(value: str) -> str:
    return value if len(value) < 12 else value[:7] + "…" + value[-4:]


def provider_ready(name: str) -> tuple[bool, str]:
    preset = PROVIDERS[name]
    if name == "azure" and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return False, "AZURE_OPENAI_ENDPOINT missing"
    if name == "openai-compatible" and not os.environ.get("RGA_LLM_BASE_URL"):
        return False, "RGA_LLM_BASE_URL missing"
    if preset.needs_key and not resolve_api_key(preset):
        return False, f"{preset.key_env} missing"
    return True, "ready"


def prefill_form_from(state: RunState) -> None:
    """Copy a previous run's options into the form widgets (called before the widgets are created)."""
    o = state.options
    st.session_state.update({
        "f_topic": o.topic, "f_discipline": o.discipline or "", "f_from": o.from_year or 1900,
        "f_to": o.to_year or 0, "f_method": o.methodology_preference or "",
        "f_priority": o.priority or "(none)", "f_goal": o.goal, "f_existing": o.existing_work or "",
        "f_sources": [s for s in o.sources if s in ALL_SOURCES],
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
        return True, f"Connected. Served by model `{used}` (self-reported: {out.model_seen})."
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
    return f"{job['run_id'] or 'starting'} — {job['label']} [{job['status']}, {job['stage']}]"


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔬 Research Gap Agent")
    st.caption("Autonomous, evidence-backed research-gap discovery over academic APIs.")

    st.subheader("LLM provider")
    detected = detect_provider()
    chosen_provider = st.selectbox("Provider", list(PROVIDERS), index=list(PROVIDERS).index(detected), key="sb_provider",
                                   help="Auto-detected from which API key is set; RGA_LLM_PROVIDER overrides.")
    ready, why = provider_ready(chosen_provider)
    preset = PROVIDERS[chosen_provider]
    st.markdown(f"{'✅' if ready else '⚠️'} **{chosen_provider}** — {why}. Default model `{preset.default_model or 'set one'}`."
                + (f" {preset.note}" if preset.note else ""))
    for name, _, _ in PROVIDER_FIELDS[chosen_provider]:
        val = os.environ.get(name)
        st.markdown(f"{'✅' if val else '▫️'} `{name}`" + (f" — `{masked(val)}`" if val else " — not set"))
    if st.button("Test connection", help="Makes one tiny model call with the saved keys."):
        with st.spinner("Calling the model…"):
            ok_conn, msg = test_llm_connection(chosen_provider, st.session_state.get("f_model") or None,
                                               st.session_state.get("f_base_url") or None)
        (st.success if ok_conn else st.error)(msg)
    with st.expander("Add or change provider keys"):
        st.caption(f"Saved to `{default_env_path()}` (git-ignored, owner-readable) and applied immediately. "
                   "Blank keeps the current value; type `clear` to remove.")
        with st.form("provider_cred_form"):
            entered = {name: st.text_input(name, type="password" if secret else "default", help=why_)
                       for name, secret, why_ in PROVIDER_FIELDS[chosen_provider]}
            if st.form_submit_button("Save provider keys"):
                changes = {k: ("" if v.strip().lower() == "clear" else v.strip()) for k, v in entered.items() if v.strip()}
                changes["RGA_LLM_PROVIDER"] = chosen_provider
                update_env_file(changes)
                st.success(f"Saved {', '.join(changes)}")
                st.rerun()

    st.subheader("Academic sources")
    for name, _, _ in SOURCE_FIELDS:
        val = os.environ.get(name)
        st.markdown(f"{'✅' if val else '▫️'} `{name}`" + (f" — `{masked(val)}`" if val else " — not set"))
    with st.expander("Add or change source keys"):
        with st.form("source_cred_form"):
            entered = {name: st.text_input(name, type="password" if secret else "default", help=why_)
                       for name, secret, why_ in SOURCE_FIELDS}
            if st.form_submit_button("Save source keys"):
                changes = {k: ("" if v.strip().lower() == "clear" else v.strip()) for k, v in entered.items() if v.strip()}
                if changes:
                    update_env_file(changes)
                    st.success(f"Saved {', '.join(changes)}")
                    st.rerun()
                else:
                    st.info("Nothing entered.")
        if not os.environ.get("OPENALEX_API_KEY"):
            mail = os.environ.get("RGA_MAILTO", "")
            st.caption("No OpenAlex key yet. OpenAlex can email you a free one (sends your RGA_MAILTO address to OpenAlex).")
            if st.button("Request a free OpenAlex key", disabled=not mail,
                         help="Set RGA_MAILTO first" if not mail else f"Emails a confirmation link to {mail}"):
                from research_gap_agent.sources.openalex import request_api_key
                st.info(request_api_key(mail))

    if st.session_state.get("jobs"):
        st.subheader("Runs in this session")
        for j in st.session_state["jobs"]:
            drain(j)
            icon = {"running": "🟢", "done": "✅", "stopped": "⏹"}.get(j["status"], "⚠️")
            st.markdown(f"{icon} `{j['run_id'] or 'starting'}` — {j['stage']}  \n"
                        f"<small>{(j['log'][-1][9:] if j['log'] else 'starting…')[:90]}</small>", unsafe_allow_html=True)
    st.subheader("Past runs")
    runs = list_runs(settings().runs_dir)
    labels = {rid: f"{rid} — {s.options.topic[:50]} ({len(s.stages_completed)}/{len(STAGES)} stages)" for rid, s in runs}
    selected_run = st.selectbox("Open a run", options=[""] + [r for r, _ in runs], format_func=lambda r: labels.get(r, "—"),
                                key="sb_run")
    if selected_run and st.session_state.get("prefilled_from") != selected_run:
        prefill_form_from(dict(runs)[selected_run])
        st.session_state["prefilled_from"] = selected_run
        st.caption("Form filled with this run's settings.")

tab_new, tab_view = st.tabs(["New investigation", "Results"])

# ----------------------------------------------------------------------------- new run form
with tab_new:
    for k, v in {"f_provider": chosen_provider, "f_model": settings().llm_model or PROVIDERS[chosen_provider].default_model,
                 "f_from": 2020, "f_to": 0, "f_sources": ["openalex", "semantic_scholar", "crossref", "arxiv"],
                 "f_max_papers": 60, "f_per_source": 25, "f_rounds": 2, "f_max_ft": 20, "f_priority": "(none)", "f_goal": "research", "f_existing": ""}.items():
        st.session_state.setdefault(k, v)
    with st.form("run_form"):
        topic = st.text_input("Research topic *", key="f_topic",
                              placeholder="e.g. Securing Model Context Protocol (MCP) in agentic AI systems")
        c1, c2, c3 = st.columns(3)
        discipline = c1.text_input("Academic discipline (optional)", key="f_discipline")
        from_year = c2.number_input("From year (1900 = no limit)", min_value=1900, max_value=2100, key="f_from")
        to_year = c3.number_input("To year (0 = present)", min_value=0, max_value=2100, key="f_to")
        c4, c5 = st.columns(2)
        methodology = c4.text_input("Preferred methodology (optional)", key="f_method")
        priority = c5.selectbox("Priority", PRIORITIES, key="f_priority")
        goal = st.radio("What do you want to end up with?", list(GOALS), format_func=GOALS.get, key="f_goal", horizontal=True)
        existing = st.text_area("What you already built or want to differentiate from (optional)", key="f_existing", height=110,
                                placeholder="Paste a README summary or describe your tool. The agent looks for gaps that would make it unique "
                                            "and checks papers and GitHub for anything that already does it.")
        sources = st.multiselect("Sources", ALL_SOURCES, key="f_sources",
                                 help="github finds existing tools and code (add it for a technical-tool goal); google_scholar needs SERPAPI_KEY; pubmed is useful for biomedical topics")
        with st.expander("Advanced"):
            a1, a2, a3, a4 = st.columns(4)
            max_papers = a1.number_input("Max papers analysed", 5, 300, key="f_max_papers")
            per_source = a2.number_input("Records per query per source", 5, 100, key="f_per_source")
            rounds = a3.number_input("Validation rounds", 1, 4, key="f_rounds")
            max_ft = a4.number_input("Open-access full texts to fetch", 0, 100, key="f_max_ft")
            b1, b2, b3 = st.columns(3)
            provider = b1.selectbox("LLM provider", list(PROVIDERS), key="f_provider")
            model = b2.text_input("Model", key="f_model", help="Blank = provider default")
            base_url = b3.text_input("Base URL (self-hosted only)", key="f_base_url")
        submitted = st.form_submit_button("Run investigation", type="primary")

    if submitted:
        ok, why = provider_ready(provider)
        if not topic.strip():
            st.error("Please enter a research topic.")
        elif not sources:
            st.error("Select at least one source.")
        elif not ok and not base_url:
            st.error(f"Provider {provider} is not ready: {why}. Add the key in the sidebar.")
        else:
            model_name = model.strip() or PROVIDERS[provider].default_model
            opts = RunOptions(topic=topic.strip(), discipline=discipline or None,
                              from_year=int(from_year) if int(from_year) > 1900 else None,
                              to_year=int(to_year) or None, methodology_preference=methodology or None,
                              priority=None if priority == "(none)" else priority, sources=sources,
                              goal=goal, existing_work=existing.strip() or None,
                              per_source_limit=int(per_source), max_papers_to_analyze=int(max_papers),
                              validation_rounds=int(rounds), max_full_text=int(max_ft), provider=provider,
                              model=model_name)
            start_job(opts, None, provider, model_name, base_url.strip() or None)

    jobs = st.session_state.get("jobs", [])
    for j in jobs:
        drain(j)
    job = st.session_state.get("job")
    if len(jobs) > 1:
        running = [j for j in jobs if j["status"] == "running"]
        st.caption(f"{len(running)} of {len(jobs)} runs started in this session are still running.")
        job = st.radio("Watch run", options=jobs, format_func=job_title,
                       index=jobs.index(job) if job in jobs else len(jobs) - 1, key="watch_job")
        st.session_state["job"] = job
    if job:
        if job["status"] == "running" and not job["stop"].is_set():
            if st.button("⏹ Stop after the current step", help="Finishes the model call in flight, saves state, and stops. "
                                                             "The run can be resumed later from the Results tab."):
                job["stop"].set()
                job["log"].append(f"{time.strftime('%H:%M:%S')} stop requested; finishing the current step…")
        elif job["status"] == "running":
            st.info("Stopping after the current step… (a long model call can take several minutes to finish)")
        status = st.status("Investigation running…" if job["status"] == "running" else f"Investigation {job['status'].replace('_', ' ')}",
                           expanded=True, state="running" if job["status"] == "running" else
                           ("complete" if job["status"] == "done" else "error"))
        head = status.empty()
        log_box = status.empty()
        while True:
            for j in jobs:
                drain(j)
            now = time.time()
            stage_idx = STAGES.index(job["stage"]) + 1 if job["stage"] in STAGES else 0
            head.markdown(
                f"**Stage {stage_idx}/{len(STAGES)}: {job['stage']}** · in this stage {int(now - job['stage_started'])//60}m"
                f"{int(now - job['stage_started'])%60:02d}s · total {int(now - job['started'])//60}m{int(now - job['started'])%60:02d}s  \n"
                f"Latest: `{job['log'][-1][9:] if job['log'] else 'starting…'}`")
            log_box.code("\n".join(job["log"][-80:]) or "starting…")
            if job["status"] != "running":
                break
            time.sleep(1.0)
        if job["status"] == "done":
            status.update(label=f"Done — run {job['run_id']}", state="complete", expanded=False)
            st.success(f"Report ready. Open the **Results** tab (run `{job['run_id']}`).")
            selected_run = job["run_id"]
        elif job["status"] == "stopped":
            status.update(label=f"Stopped — run {job['run_id']} saved", state="error", expanded=False)
        others = [j for j in jobs if j is not job and j["status"] == "running"]
        if others:
            st.info("Still running in the background: " + ", ".join(j["run_id"] or "starting" for j in others)
                    + ". Pick it under **Watch run** above to follow its log.")
            st.warning("Stopped at a checkpoint. Open the run in the **Results** tab and click **Resume** to continue later.")
        elif job["status"] == "config_error":
            status.update(label="Stopped — credentials need attention", state="error", expanded=False)
            st.error("**The model call was rejected because of account configuration.**\n\n" + job["config_error"]
                     + "\n\nFix it in the sidebar (provider panel), click **Test connection**, then resume this run from the Results tab.")
        elif job["status"] == "error":
            status.update(label="Failed — state was saved; you can resume from the Results tab", state="error")
            st.error("The run failed. The traceback is in the log above.")

# ----------------------------------------------------------------------------- results tab
with tab_view:
    run_id = selected_run or (st.session_state.get("job") or {}).get("run_id")
    if not run_id:
        st.info("No run selected. Start one in **New investigation** or pick a past run in the sidebar.")
    else:
        run_dir = settings().runs_dir / run_id
        sp = run_dir / "state.json"
        if not sp.exists():
            st.warning("State file not found for this run.")
        else:
            state = RunState.model_validate(json.loads(sp.read_text()))
            done = set(state.stages_completed)
            st.markdown(f"### {state.options.topic}")
            st.caption(f"Provider {state.options.provider} · model {state.options.model} · sources {', '.join(state.options.sources)}")
            st.markdown(" → ".join(f"{'✅' if s in done else '⬜'} {s}" for s in STAGES))
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Records retrieved", state.n_records_retrieved)
            m2.metric("Unique papers", len(state.papers))
            m3.metric("Analysed", len(state.evidence_rows))
            m4.metric("Full texts", len(state.full_text_analyzed_ids()))
            surviving = [v for v in state.validated_gaps if not v.rejected]
            m5.metric("Gaps (surviving / total)", f"{len(surviving)} / {len(state.validated_gaps)}")
            if len(done) < len(STAGES):
                st.warning(f"This run is incomplete ({len(done)}/{len(STAGES)} stages).")
                if st.button("Resume this run"):
                    start_job(None, run_id, st.session_state.get("f_provider"), st.session_state.get("f_model"),
                              st.session_state.get("f_base_url") or None)
                    st.rerun()
            cols = st.columns(len(EXPORT_FILES))
            for col, name in zip(cols, EXPORT_FILES):
                path = run_dir / name
                if path.exists():
                    col.download_button(f"⬇ {name}", data=path.read_bytes(), file_name=f"{run_id}_{name}")
            report = run_dir / "report.md"
            if report.exists():
                sub_report, sub_gaps, sub_papers, sub_log = st.tabs(["Report", "Gaps", "Papers", "Search log"])
                with sub_report:
                    html_path = run_dir / "report.html"
                    if html_path.exists():
                        import streamlit.components.v1 as components
                        components.html(html_path.read_text(), height=1400, scrolling=True)
                    else:
                        st.markdown(report.read_text(), unsafe_allow_html=True)
                with sub_gaps:
                    for v in state.validated_gaps:
                        badge = "❌ rejected" if v.rejected else f"✅ {v.status.value}"
                        with st.expander(f"{v.gap.id}: {v.gap.title} — {badge} (confidence {v.confidence})"):
                            st.markdown(f"**Missing:** {v.gap.what_is_missing}\n\n**Why it matters:** {v.gap.why_it_matters}")
                            st.markdown(f"**Confidence:** {v.confidence_explanation}")
                            if v.counter_evidence:
                                st.markdown("**Counter-evidence:**\n" + "\n".join(
                                    f"- {c.paper_id} ({c.relationship}): {c.note}" for c in v.counter_evidence))
                with sub_papers:
                    rows = [{"id": p.id, "title": p.title, "year": p.year, "type": p.publication_type,
                             "included": bool(p.screening and p.screening.include), "full text": p.has_full_text,
                             "retracted": bool(p.is_retracted), "sources": ", ".join(p.sources), "url": p.best_url()}
                            for p in sorted(state.papers.values(), key=lambda p: p.id)]
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                with sub_log:
                    st.dataframe([{"id": s.id, "source": s.source, "query": s.query, "purpose": s.purpose,
                                   "status": s.status, "retrieved": s.n_retrieved, "new": s.n_new_unique}
                                  for s in state.search_records], use_container_width=True, hide_index=True)
            else:
                st.info("No report yet for this run.")
