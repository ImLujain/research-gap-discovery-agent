import httpx
import pytest
import respx

from research_gap_agent.config import Settings
from research_gap_agent.sources.arxiv import ArxivSource, build_query
from research_gap_agent.sources.crossref import CrossrefSource
from research_gap_agent.sources.openalex import OpenAlexSource, reconstruct_abstract
from research_gap_agent.sources.semantic_scholar import SemanticScholarSource


@pytest.fixture
def settings(tmp_path):
    return Settings(cache_dir=tmp_path / "cache", mailto="t@example.org")


def test_reconstruct_abstract():
    inv = {"world": [1], "Hello": [0], "again": [2]}
    assert reconstruct_abstract(inv) == "Hello world again"
    assert reconstruct_abstract(None) is None


@respx.mock
def test_openalex_parses_fields_retraction_and_preprint(settings):
    body = {"meta": {"count": 2}, "results": [
        {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/A", "title": "T1", "publication_year": 2024,
         "authorships": [{"author": {"display_name": "A. Author"}}], "abstract_inverted_index": {"x": [0]},
         "primary_location": {"source": {"display_name": "J", "type": "journal"}, "landing_page_url": "https://j/1"},
         "type": "article", "cited_by_count": 7, "is_retracted": True, "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123"},
         "open_access": {"oa_url": "https://oa/1"}},
        {"id": "https://openalex.org/W2", "doi": None, "title": "T2", "publication_year": 2023, "authorships": [],
         "abstract_inverted_index": None,
         "primary_location": {"source": {"display_name": "arXiv", "type": "repository"},
                              "landing_page_url": "https://arxiv.org/abs/2301.00001", "version": "submittedVersion"},
         "type": "article", "cited_by_count": 0, "is_retracted": False, "ids": {}, "open_access": {}},
    ]}
    route = respx.get("https://api.openalex.org/works").mock(return_value=httpx.Response(200, json=body))
    out = OpenAlexSource(settings).search("prompt injection", from_year=2022, limit=10)
    assert out.status == "ok" and out.total_available == 2
    a, b = out.records
    assert a.doi == "https://doi.org/10.1/A" and a.pmid == "123" and a.is_retracted is True
    assert a.publication_type == "journal-article" and a.authors == ["A. Author"] and a.abstract == "x"
    assert b.publication_type == "preprint" and b.arxiv_id == "https://arxiv.org/abs/2301.00001" and b.abstract is None
    assert "from_publication_date:2022-01-01" in route.calls[0].request.url.params["filter"]
    assert route.calls[0].request.url.params["mailto"] == "t@example.org"
    # second call served from cache: no new HTTP call
    OpenAlexSource(settings).search("prompt injection", from_year=2022, limit=10)
    assert route.call_count == 1


@respx.mock
def test_semantic_scholar_rate_limit_is_reported_not_raised(settings, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(return_value=httpx.Response(429))
    src = SemanticScholarSource(settings)
    src.min_interval = 0
    out = src.search("x")
    assert out.status == "rate_limited" and out.records == []


@respx.mock
def test_semantic_scholar_parses_and_labels_preprints(settings):
    body = {"total": 1, "data": [
        {"paperId": "abc", "title": "T", "abstract": "A", "year": 2024, "authors": [{"name": "X Y"}],
         "externalIds": {"ArXiv": "2401.00001"}, "citationCount": 3, "venue": "arXiv.org", "publicationTypes": None},
        {"paperId": "def", "title": "U", "abstract": None, "year": 2023, "authors": [],
         "externalIds": {"DOI": "10.1/u", "PubMed": "99"}, "citationCount": 1, "venue": "NeurIPS",
         "publicationTypes": ["JournalArticle", "Conference"]},
    ]}
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(return_value=httpx.Response(200, json=body))
    src = SemanticScholarSource(settings)
    src.min_interval = 0
    out = src.search("x", from_year=2022, to_year=2025)
    a, b = out.records
    assert a.publication_type == "preprint" and a.arxiv_id == "2401.00001"
    assert b.publication_type == "conference-paper" and b.pmid == "99" and b.doi == "10.1/u"


@respx.mock
def test_crossref_parses_and_skips_retraction_notices(settings):
    body = {"message": {"total-results": 2, "items": [
        {"DOI": "10.1/j", "title": ["<i>Jats</i> title"], "author": [{"given": "A", "family": "B"}],
         "issued": {"date-parts": [[2024, 1]]}, "container-title": ["Journal"], "type": "journal-article",
         "abstract": "<jats:p>Abs</jats:p>", "is-referenced-by-count": 4, "URL": "https://doi.org/10.1/j"},
        {"DOI": "10.1/r", "title": ["Retraction: X"], "type": "journal-article", "update-to": [{"type": "retraction", "DOI": "10.1/j"}]},
        {"DOI": "10.1/p", "title": ["Preprint"], "type": "posted-content", "issued": {"date-parts": [[None]]}},
    ]}}
    respx.get("https://api.crossref.org/works").mock(return_value=httpx.Response(200, json=body))
    src = CrossrefSource(settings)
    src.min_interval = 0
    out = src.search("q", from_year=2020)
    assert [r.doi for r in out.records] == ["10.1/j", "10.1/p"]
    j, p = out.records
    assert j.title == "Jats title" and j.abstract == "Abs" and j.year == 2024 and j.authors == ["A B"]
    assert p.publication_type == "preprint" and p.year is None


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>42</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <published>2024-01-30T00:00:00Z</published>
    <title>A  title
    spanning lines</title>
    <summary>Summary text.</summary>
    <author><name>First Author</name></author><author><name>Second Author</name></author>
    <arxiv:doi>10.1/arx</arxiv:doi>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2" rel="related" type="application/pdf"/>
  </entry>
</feed>"""


@respx.mock
def test_arxiv_parses_atom(settings):
    respx.get("https://export.arxiv.org/api/query").mock(return_value=httpx.Response(200, text=ATOM))
    src = ArxivSource(settings)
    src.min_interval = 0
    out = src.search('"prompt injection" agents', from_year=2023)
    assert out.total_available == 42 and len(out.records) == 1
    r = out.records[0]
    assert r.title == "A title spanning lines" and r.arxiv_id == "2401.12345v2" and r.doi == "10.1/arx"
    assert r.year == 2024 and r.publication_type == "preprint" and r.oa_url.endswith(".pdf") or r.oa_url
    assert r.authors == ["First Author", "Second Author"]


def test_arxiv_query_builder():
    assert build_query('"prompt injection" LLM agents') == 'all:"prompt injection" AND all:LLM AND all:agents'
    assert build_query("a AND b") == "all:a AND all:b"


EFETCH = """<?xml version="1.0"?><PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID>
<Article><Journal><Title>J Med</Title><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Title with <i>markup</i> inside</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">Part one.</AbstractText><AbstractText>Part two.</AbstractText></Abstract>
<AuthorList><Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author></AuthorList>
<PublicationTypeList><PublicationType>Journal Article</PublicationType><PublicationType>Review</PublicationType></PublicationTypeList>
</Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1000/pm</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle></PubmedArticleSet>"""


def test_pubmed_parses_efetch_titles_with_markup():
    from research_gap_agent.sources.pubmed import PubMedSource

    recs = PubMedSource.parse_efetch(EFETCH)
    assert len(recs) == 1
    r = recs[0]
    assert r.title == "Title with markup inside" and r.pmid == "42" and r.doi == "10.1000/pm"
    assert r.abstract == "Part one. Part two." and r.authors == ["Jane Doe"] and r.year == 2025
    assert r.publication_type == "review" and r.venue == "J Med"


@respx.mock
def test_long_retry_after_skips_source_instead_of_sleeping(settings, monkeypatch):
    """OpenAlex's exhausted daily budget answers 429 with Retry-After = hours; the run must not sleep that long."""
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    route = respx.get("https://api.openalex.org/works").mock(return_value=httpx.Response(
        429, headers={"Retry-After": "46708"},
        json={"error": "Rate limit exceeded", "message": "Insufficient budget. resets at midnight UTC"}))
    src = OpenAlexSource(settings)
    src.min_interval = 0
    events = []
    src.on_event = events.append
    out = src.search("x")
    assert out.status == "rate_limited" and "13.0 h" in (out.error or "")
    assert not sleeps and route.call_count == 1
    assert any("skipping openalex" in e and "Insufficient budget" in e for e in events), events
    # later queries in the same run are skipped without another request
    out2 = src.search("y")
    assert out2.status == "rate_limited" and out2.error.startswith("skipped:") and route.call_count == 1


@respx.mock
def test_openalex_sends_api_key_and_mailto(tmp_path):
    route = respx.get("https://api.openalex.org/works").mock(return_value=httpx.Response(200, json={"results": [], "meta": {"count": 0}}))
    src = OpenAlexSource(Settings(cache_dir=tmp_path / "c", mailto="t@example.org", openalex_api_key="oak"))
    src.min_interval = 0
    assert src.search("x").status == "ok"
    q = dict(route.calls[0].request.url.params)
    assert q["api_key"] == "oak" and q["mailto"] == "t@example.org"


@respx.mock
def test_github_repositories_become_software_records(tmp_path):
    from research_gap_agent.sources.github import GitHubSource
    route = respx.get("https://api.github.com/search/repositories").mock(return_value=httpx.Response(200, json={
        "total_count": 1, "items": [{
            "id": 42, "full_name": "acme/mcp-scan", "html_url": "https://github.com/acme/mcp-scan",
            "description": "Static scanner for MCP configs", "created_at": "2025-04-01T00:00:00Z",
            "pushed_at": "2026-10-30T00:00:00Z", "stargazers_count": 1200, "forks_count": 80, "language": "Python",
            "topics": ["mcp", "security"], "owner": {"login": "acme"}, "license": {"spdx_id": "MIT"}}]}))
    src = GitHubSource(Settings(cache_dir=tmp_path / "c", github_token="ghp_test"))
    src.min_interval = 0
    out = src.search("mcp scanner", from_year=2024, limit=10)
    assert out.status == "ok" and out.total_available == 1
    r = out.records[0]
    assert r.publication_type == "software" and r.year == 2025 and r.cited_by_count == 1200
    assert r.title.startswith("acme/mcp-scan") and "not a paper" in r.abstract and "MIT" in r.abstract
    req = route.calls[0].request
    assert "created:>=2024-01-01" in dict(req.url.params)["q"] and req.headers["Authorization"] == "Bearer ghp_test"


@respx.mock
def test_github_403_is_reported_as_rate_limited(tmp_path, monkeypatch):
    from research_gap_agent.sources.github import GitHubSource
    monkeypatch.setattr("time.sleep", lambda s: None)
    respx.get("https://api.github.com/search/repositories").mock(return_value=httpx.Response(403, json={"message": "rate limit"}))
    src = GitHubSource(Settings(cache_dir=tmp_path / "c"))
    src.min_interval = 0
    assert src.search("x").status == "rate_limited"
