"""Parallel model calls: screening/extraction batches and gap validations run concurrently with identical results."""
from __future__ import annotations

import threading
import time

from research_gap_agent.agents.screening import screen_papers
from research_gap_agent.config import Settings
from research_gap_agent.coordinator import STAGES, Coordinator
from research_gap_agent.llm import ScriptedLLM, screen_model_for
from research_gap_agent.models import Paper, RunOptions, ScreeningItem, ScreeningOutput
from research_gap_agent.parallel import run_parallel
from tests.conftest import make_scripted_llm


def test_run_parallel_preserves_order_and_overlaps():
    active, peak, lock = [0], [0], threading.Lock()

    def slow(x):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return x * 2

    assert run_parallel(slow, range(8), workers=4) == [x * 2 for x in range(8)]
    assert peak[0] > 1
    assert run_parallel(slow, range(3), workers=1) == [0, 2, 4]


def test_screening_batches_run_in_parallel_with_ordered_decisions():
    papers = [Paper(id=f"P{i:04d}", title=f"Paper {i}", abstract="about the topic") for i in range(1, 61)]
    seen_models = []

    def handler(system, user):
        ids = [line.split("]")[0].strip("[") for line in user.splitlines() if line.startswith("[P")]
        time.sleep(0.02)
        return ScreeningOutput(decisions=[ScreeningItem(paper_id=i, include=True, reason="ok") for i in ids])

    llm = ScriptedLLM({ScreeningOutput: handler})
    out = screen_papers(llm, papers, "criterion", batch_size=10, concurrency=4, model="fast-model")
    assert [d.paper_id for d in out] == [p.id for p in papers]
    assert len(llm.calls) == 6
    assert {c.model for c in llm.call_log} == {"fast-model"}


def test_screen_model_resolution(monkeypatch):
    class Main:
        model = "claude-opus-5-5"

    s = Settings(llm_provider="anthropic", screen_model=None)
    assert screen_model_for(s, Main()) == "claude-haiku-5-5"
    assert screen_model_for(Settings(llm_provider="anthropic", screen_model="main"), Main()) is None
    assert screen_model_for(Settings(llm_provider="anthropic", screen_model="claude-sonnet-5-5"), Main()) == "claude-sonnet-5-5"
    assert screen_model_for(Settings(llm_provider="ollama", screen_model=None), Main()) is None  # no fast preset

    class Haiku:
        model = "claude-haiku-5-5"

    assert screen_model_for(s, Haiku()) is None  # main model already is the fast one


def test_pipeline_with_parallel_validation_matches_sequential(tmp_path, fake_sources):
    opts = RunOptions(topic="Securing MCP in agentic AI systems", from_year=2022, sources=["alpha", "beta"],
                      per_source_limit=10, validation_rounds=2, validation_per_source_limit=5, max_full_text=0)
    plan = {"G01": ["addressed"], "G02": ["partially_addressed", "preliminary_evidence"]}
    seq = Settings(cache_dir=tmp_path / "c1", runs_dir=tmp_path / "r1", llm_concurrency=1, screen_model="main")
    par = Settings(cache_dir=tmp_path / "c2", runs_dir=tmp_path / "r2", llm_concurrency=3, screen_model="main")
    s1 = Coordinator(opts, make_scripted_llm(judge_plan=plan), settings=seq, sources=fake_sources, run_id="seq").run()
    llm2 = make_scripted_llm(judge_plan=plan)
    c2 = Coordinator(opts, llm2, settings=par, sources=fake_sources, run_id="par")
    s2 = c2.run()
    assert s2.stages_completed == STAGES
    assert [v.gap.id for v in s2.validated_gaps] == [v.gap.id for v in s1.validated_gaps] == ["G01", "G02"]
    assert [(v.status, v.rejected) for v in s2.validated_gaps] == [(v.status, v.rejected) for v in s1.validated_gaps]
    assert [r.gap_id for r in s2.ranking] == [r.gap_id for r in s1.ranking]
    assert len(s2.papers) == len(s1.papers) and len(s2.search_records) == len(s1.search_records)
    assert len({r.id for r in s2.search_records}) == len(s2.search_records)  # ids stay unique under the lock
    assert len(s2.llm_calls) == len(llm2.calls)
    assert c2.state_path.exists()
