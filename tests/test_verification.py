import httpx
import respx

from research_gap_agent.agents.verification import verify_dois
from research_gap_agent.config import Settings
from research_gap_agent.models import Paper, VerificationReport
from research_gap_agent.sources.crossref import CrossrefSource
from research_gap_agent.sources.openalex import OpenAlexSource


def _src(tmp_path, cls):
    s = cls(Settings(cache_dir=tmp_path / "c"))
    s.min_interval = 0
    s.max_attempts = 1
    return s


@respx.mock
def test_doi_resolution_title_match_and_mismatch(tmp_path):
    respx.get("https://api.crossref.org/works/10.1/good").mock(
        return_value=httpx.Response(200, json={"message": {"title": ["A Study of Agents"]}}))
    respx.get("https://api.crossref.org/works/10.1/bad").mock(
        return_value=httpx.Response(200, json={"message": {"title": ["Completely different paper"]}}))
    respx.get("https://api.crossref.org/works/10.1/missing").mock(return_value=httpx.Response(404))
    papers = [
        Paper(id="P0001", title="A study of agents", doi="10.1/good", is_retracted=False),
        Paper(id="P0002", title="A study of agents", doi="10.1/bad", is_retracted=False),
        Paper(id="P0003", title="Gone", doi="10.1/missing", is_retracted=False),
    ]
    rep = VerificationReport()
    verify_dois(papers, _src(tmp_path, CrossrefSource), None, rep)
    assert papers[0].doi_check.resolved and papers[0].doi_check.title_match is True
    assert papers[1].doi_check.title_match is False and "differs" in papers[1].doi_check.note
    assert papers[2].doi_check.resolved is False
    assert len(rep.doi_checks) == 3


@respx.mock
def test_retraction_detected_via_crossref_relation_and_openalex(tmp_path):
    respx.get("https://api.crossref.org/works/10.1/ret").mock(
        return_value=httpx.Response(200, json={"message": {"title": ["Retracted work"], "relation": {"is-retracted-by": [{}]}}}))
    respx.get("https://api.crossref.org/works/10.1/oa").mock(
        return_value=httpx.Response(200, json={"message": {"title": ["Checked in OpenAlex"]}}))
    respx.get("https://api.openalex.org/works/https://doi.org/10.1/oa").mock(
        return_value=httpx.Response(200, json={"id": "https://openalex.org/W9", "title": "Checked in OpenAlex", "is_retracted": True}))
    papers = [Paper(id="P0001", title="Retracted work", doi="10.1/ret"), Paper(id="P0002", title="Checked in OpenAlex", doi="10.1/oa"),
              Paper(id="P0003", title="Already flagged", doi=None, is_retracted=True)]
    rep = VerificationReport()
    verify_dois(papers, _src(tmp_path, CrossrefSource), _src(tmp_path, OpenAlexSource), rep)
    assert papers[0].is_retracted is True and papers[1].is_retracted is True
    assert sorted(rep.retracted_paper_ids) == ["P0001", "P0002", "P0003"]
