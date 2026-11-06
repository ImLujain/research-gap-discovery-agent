"""Streamlit GUI boots headlessly, validates input, and lists past runs."""
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Never let GUI tests read or write the real project .env or inherit real keys."""
    monkeypatch.setenv("RGA_ENV_FILE", str(tmp_path / ".env"))
    monkeypatch.chdir(tmp_path)
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "RGA_LLM_PROVIDER", "RGA_LLM_MODEL", "RGA_LLM_BASE_URL", "RGA_LLM_API_KEY", "S2_API_KEY", "SERPAPI_KEY",
              "NCBI_API_KEY", "RGA_MAILTO"):
        monkeypatch.delenv(k, raising=False)


def test_app_renders_and_validates_empty_topic(tmp_path, monkeypatch):
    monkeypatch.setenv("RGA_RUNS_DIR", str(tmp_path / "runs"))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert any("Research Gap Agent" in t.value for t in at.sidebar.title)
    next(b for b in at.button if "Start investigation" in b.label).click().run()  # submit with empty topic
    assert any("research topic" in e.value.lower() for e in at.error)


def test_app_lists_past_runs_and_shows_report(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setenv("RGA_RUNS_DIR", str(runs))
    from research_gap_agent.models import RunOptions, RunState

    st = RunState(run_id="old1", options=RunOptions(topic="Old topic about agents"), stages_completed=["plan"])
    (runs / "old1").mkdir(parents=True)
    (runs / "old1" / "state.json").write_text(st.model_dump_json())
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    sb = next(s for s in at.sidebar.selectbox if s.label == "Open a run")
    label = next(o for o in sb.options if o.endswith("old1"))
    assert "1/12 stages" in label and label.startswith("Old topic about agents")
    sb.select("old1").run()  # select by raw value; options are shown with formatted labels
    assert not at.exception
    assert any("incomplete" in w.value for w in at.warning)
    assert any("Old topic about agents" in t.value for t in at.title)


def test_sidebar_saves_keys_to_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RGA_RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / ".env").write_text("# existing\n")
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    form_inputs = {t.label: t for t in at.sidebar.text_input}
    form_inputs["Semantic Scholar key"].input("s2-secret")
    save = next(b for b in at.sidebar.button if "source keys" in b.label)
    save.click().run()
    assert not at.exception
    assert "S2_API_KEY=s2-secret" in (tmp_path / ".env").read_text()
    assert any("✅ Semantic Scholar key" in m.value for m in at.sidebar.markdown)


def test_provider_selector_and_key_form(tmp_path, monkeypatch):
    monkeypatch.setenv("RGA_RUNS_DIR", str(tmp_path / "runs"))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    prov = next(s for s in at.sidebar.selectbox if s.label == "Provider")
    assert prov.value == "anthropic"
    prov.select("openai").run()
    assert any("OpenAI API key" in t.label for t in at.sidebar.text_input)
    next(t for t in at.sidebar.text_input if t.label == "OpenAI API key").input("sk-openai")
    next(b for b in at.sidebar.button if "provider keys" in b.label).click().run()
    assert not at.exception
    text = (tmp_path / ".env").read_text()
    assert "OPENAI_API_KEY=sk-openai" in text and "RGA_LLM_PROVIDER=openai" in text


def test_selecting_past_run_prefills_form(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setenv("RGA_RUNS_DIR", str(runs))
    from research_gap_agent.models import RunOptions, RunState

    opts = RunOptions(topic="XAI for IoT intrusion detection", discipline="cybersecurity", from_year=2019, to_year=2024,
                      priority="applied", sources=["openalex", "pubmed"], max_papers_to_analyze=33,
                      validation_rounds=3, max_full_text=7, provider="openai", model="gpt-5")
    (runs / "r1").mkdir(parents=True)
    (runs / "r1" / "state.json").write_text(RunState(run_id="r1", options=opts).model_dump_json())
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    next(s for s in at.sidebar.selectbox if s.label == "Open a run").select("r1").run()
    assert not at.exception
    inputs = {t.label: t.value for t in at.text_input}
    assert inputs["Research topic"] == "XAI for IoT intrusion detection"
    assert inputs["Field or discipline (optional)"] == "cybersecurity"
    assert inputs["Model"] == "gpt-5"
    nums = {n.label: n.value for n in at.number_input}
    assert nums["From year"] == 2019 and nums["To year"] == 2024
    assert nums["Max papers analysed"] == 33 and nums["Validation rounds"] == 3 and nums["Open-access full texts to fetch"] == 7
    assert next(m for m in at.multiselect if m.label == "Databases to search").value == ["openalex", "pubmed"]
    assert next(s for s in at.selectbox if s.label == "Kind of contribution").value == "applied"
    assert next(r for r in at.radio if r.label.startswith("How deep")).value == "Custom"
    assert next(s for s in at.selectbox if s.label == "LLM provider").value == "openai"


def test_test_connection_button_reports_config_error(tmp_path, monkeypatch):
    monkeypatch.setenv("RGA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    import research_gap_agent.llm as llm_mod

    class Boom:
        def __init__(self, *a, **k):
            pass

        def structured(self, **kw):
            raise llm_mod.LLMConfigError("needs a workspace id: set ANTHROPIC_WORKSPACE_ID")

        call_log = []

    monkeypatch.setattr(llm_mod, "AnthropicLLM", Boom)
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    next(b for b in at.sidebar.button if b.label == "Test connection").click().run()
    assert not at.exception
    assert any("ANTHROPIC_WORKSPACE_ID" in e.value for e in at.sidebar.error)


def test_form_defaults_do_not_trigger_session_state_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("RGA_RUNS_DIR", str(tmp_path / "runs"))
    import logging

    with caplog.at_level(logging.WARNING):
        at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert not [r for r in caplog.records if "Session State API" in r.getMessage()]
