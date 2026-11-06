"""Progress events and the rate-limit circuit breaker in discovery."""
import httpx
import respx

from research_gap_agent.config import Settings
from research_gap_agent.discovery import LiteratureDiscovery
from research_gap_agent.models import RunOptions, RunState
from research_gap_agent.sources.semantic_scholar import SemanticScholarSource

from conftest import FakeSource, rec


def test_discovery_emits_per_query_progress(settings):
    src = FakeSource("alpha", {"q": [rec("alpha", "a1", "Paper one", doi="10.1000/one")]})
    state = RunState(run_id="t", options=RunOptions(topic="x", sources=["alpha"]))
    lines = []
    d = LiteratureDiscovery([src], state, progress=lines.append)
    d.run_queries(["q one", "q two"], purpose="primary", from_year=None, to_year=None, limit=5)
    assert any("alpha: query 1/2" in l and "…" in l for l in lines)
    assert any("alpha: query 2/2 → 1 records" in l for l in lines)
    assert lines[-1].startswith("[primary] batch done: 2 searches")


@respx.mock
def test_breaker_skips_rest_of_batch_after_consecutive_rate_limits(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    route = respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(return_value=httpx.Response(429))
    src = SemanticScholarSource(Settings(cache_dir=tmp_path / "c", s2_api_key=None))
    src.min_interval = 0
    assert src.max_attempts == 3  # fail fast without a key
    state = RunState(run_id="t", options=RunOptions(topic="x", sources=["semantic_scholar"]))
    lines = []
    d = LiteratureDiscovery([src], state, progress=lines.append)
    recs = d.run_queries(["a", "b", "c", "d"], purpose="primary", from_year=None, to_year=None, limit=5)
    assert [r.status for r in recs] == ["rate_limited"] * 4
    assert route.call_count == 2 * 3  # two queries x three attempts; the last two queries never hit the network
    assert recs[2].error.startswith("skipped") and recs[3].error.startswith("skipped")
    assert any("retrying in" in l for l in lines) and any("skipping the rest of this batch" in l for l in lines)
    assert any("rate-limited" in lim for lim in state.limitations)
    # a new batch starts with the breaker reset
    d.run_queries(["e"], purpose="validation:G01", from_year=None, to_year=None, limit=5)
    assert route.call_count == 3 * 3


@respx.mock
def test_breaker_resets_after_a_success(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    ok_body = {"total": 0, "data": []}
    route = respx.get("https://api.semanticscholar.org/graph/v1/paper/search")
    route.side_effect = [httpx.Response(429), httpx.Response(429), httpx.Response(429),  # query a: rate limited
                         httpx.Response(200, json=ok_body),  # query b ok -> counter resets
                         httpx.Response(429), httpx.Response(429), httpx.Response(429),  # query c
                         httpx.Response(200, json=ok_body)]  # query d still attempted (only 1 consecutive)
    src = SemanticScholarSource(Settings(cache_dir=tmp_path / "c", s2_api_key=None))
    src.min_interval = 0
    state = RunState(run_id="t", options=RunOptions(topic="x", sources=["semantic_scholar"]))
    recs = LiteratureDiscovery([src], state).run_queries(["a", "b", "c", "d"], purpose="primary", from_year=None,
                                                         to_year=None, limit=5)
    assert [r.status for r in recs] == ["rate_limited", "ok", "rate_limited", "ok"]
