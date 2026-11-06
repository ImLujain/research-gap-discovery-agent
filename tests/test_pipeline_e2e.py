import json

from research_gap_agent.coordinator import STAGES, Coordinator
from research_gap_agent.models import RunState

from conftest import make_scripted_llm


def test_full_pipeline_report_structure_and_counts(options, settings, fake_sources):
    llm = make_scripted_llm(judge_plan={"G01": ["addressed"], "G02": ["partially_addressed", "preliminary_evidence"]})
    coord = Coordinator(options, llm, settings=settings, sources=fake_sources, run_id="e2e")
    st = coord.run()
    assert st.stages_completed == STAGES
    md = coord.report_path.read_text()
    for h in ["## 1. Executive Summary", "## 2. Current State of the Art", "## 3. Literature Evidence Matrix",
              "## 4. Identified Research Gaps", "## 5. Research Gap Comparison", "## 6. Recommended Research Direction",
              "## 7. Important Papers to Read", "## 8. Search Methodology and Reproducibility"]:
        assert h in md
    # counts: 6 raw records over 2 queries x 2 sources -> primary query hits: alpha 'prompt injection' (3) + beta (3) + alpha 'mcp' (1) = 7 raw, 5 unique
    assert st.n_records_retrieved >= 7
    assert len([p for p in st.papers.values() if any(s.query_id.startswith("S") for s in p.source_records)]) == len(st.papers)
    assert "| Unique records after deduplication |" in md
    assert "| Full-text papers analysed (open-access PDFs) | 0" in md
    for name in ("references.bib", "references.ris", "corpus.csv"):
        assert (coord.run_dir / name).exists()
    assert "@article{" in (coord.run_dir / "references.bib").read_text()
    assert next(p for p in st.papers.values() if "soil" in p.title).screening.include is False
    # rejected gap reported as rejected; surviving gap narrowed; fake refs scrubbed
    assert "G01: Gap already solved elsewhere *(rejected after validation)*" in md
    assert "G02: Narrowed gap" in md and "narrowed 1 time(s)" in md
    assert "[unverified reference removed]" in md and "P9999" not in md.split("## References")[1]
    assert st.recommendation.gap_id == "G02"
    assert [r.paper_id for r in st.reading_list] and all(r.paper_id in st.papers for r in st.reading_list)
    # ranking only contains surviving gaps
    assert [r.gap_id for r in st.ranking] == ["G02"]
    # every search is logged with status and purpose
    assert {s.purpose.split(":")[0] for s in st.search_records} >= {"primary", "validation"}
    assert "Google Scholar" in md and "not searched" in md
    # state is persisted and valid
    saved = RunState.model_validate(json.loads(coord.state_path.read_text()))
    assert saved.run_id == "e2e" and len(saved.llm_calls) == len(llm.calls)


def test_resume_skips_completed_stages_and_reuses_state(options, settings, fake_sources):
    llm1 = make_scripted_llm()
    c1 = Coordinator(options, llm1, settings=settings, sources=fake_sources, run_id="res")
    c1.run(until="extract")
    assert c1.state.stages_completed == ["plan", "discover", "screen", "fulltext", "extract"]
    n_rows = len(c1.state.evidence_rows)
    llm2 = make_scripted_llm(judge_plan={"G01": ["limited_evidence_found"], "G02": ["limited_evidence_found"]})
    c2 = Coordinator.resume("res", llm2, settings=settings, sources=fake_sources)
    st = c2.run()
    tasks = [t for t, _, _ in llm2.calls]
    assert "plan" not in tasks and "extract" not in tasks and "landscape" in tasks
    assert len(st.evidence_rows) == n_rows and st.plan is not None
    assert c2.report_path.exists()
    # dedupe index was rebuilt from persisted papers: validation searches did not duplicate existing records
    titles = [p.title.lower() for p in st.papers.values()]
    assert len(titles) == len(set(titles))


def test_no_surviving_gaps_produces_honest_report(options, settings, fake_sources):
    llm = make_scripted_llm(judge_plan={"G01": ["addressed"], "G02": ["addressed"]})
    coord = Coordinator(options, llm, settings=settings, sources=fake_sources, run_id="none")
    st = coord.run()
    md = coord.report_path.read_text()
    assert st.ranking == [] and st.recommendation is None
    assert "no candidate survived validation" in md
    assert "Every candidate gap was rejected" in md


def test_tool_goal_and_existing_work_reach_every_agent_prompt(options, settings, fake_sources):
    """With goal=tool the planner, gap generator, validator, ranker and recommender all see the tool guidance and the
    user's own work, and the recommendation is asked for a build plan."""
    llm = make_scripted_llm(judge_plan={"G01": ["addressed"], "G02": ["partially_addressed", "preliminary_evidence"]})
    opts = options.model_copy(update={"goal": "tool", "existing_work": "mcp-trustcheck: static scanner for MCP configs"})
    coord = Coordinator(opts, llm, settings=settings, sources=fake_sources, run_id="toolgoal")
    coord.run()
    seen = {task: user for task, _, user in llm.calls}
    for task in ("plan", "generate_gaps", "counter_queries:G01", "judge:G01", "rank", "recommend"):
        assert "mcp-trustcheck" in seen[task], task
    for task in ("generate_gaps", "judge:G01", "rank", "recommend"):
        assert "TECHNICAL TOOL" in seen[task], task
    assert "technical tool or system artefact" in seen["plan"]
    # saved state round-trips the new options
    saved = RunState.model_validate(json.loads(coord.state_path.read_text()))
    assert saved.options.goal == "tool" and saved.options.existing_work.startswith("mcp-trustcheck")
