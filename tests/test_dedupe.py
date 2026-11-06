from research_gap_agent.dedupe import Corpus, normalize_arxiv, normalize_doi, normalize_title
from research_gap_agent.models import RawRecord


def r(**kw):
    base = dict(source="s", source_id="x", title="T")
    base.update(kw)
    return RawRecord(**base)


def test_short_doi_prefix_is_not_a_doi():
    assert normalize_doi("10.1/a") is None


def test_normalize_doi_variants():
    assert normalize_doi("https://doi.org/10.1000/ABC.1") == "10.1000/abc.1"
    assert normalize_doi("doi:10.1000/abc.1") == "10.1000/abc.1"
    assert normalize_doi("10.1000/abc.1.") == "10.1000/abc.1"
    assert normalize_doi("not a doi") is None


def test_normalize_arxiv_variants():
    assert normalize_arxiv("https://arxiv.org/abs/2301.00001v2") == "2301.00001"
    assert normalize_arxiv("arXiv:2301.00001") == "2301.00001"
    assert normalize_arxiv("2301.00001v3") == "2301.00001"
    assert normalize_arxiv("cs/0101001v1") == "cs/0101001"
    assert normalize_arxiv("W123") is None


def test_normalize_title():
    assert normalize_title("  Prompt-Injection: Attacks!  ") == "prompt injection attacks"


def test_doi_match_merges_across_sources_and_keeps_provenance():
    c = Corpus()
    p1, new1 = c.add(r(source="openalex", source_id="W1", title="A paper", doi="10.1000/X", abstract=None), "S001")
    p2, new2 = c.add(r(source="crossref", source_id="10.1000/x", title="A Paper", doi="https://doi.org/10.1000/x", abstract="full"), "S002")
    assert new1 and not new2 and p1 is p2
    assert p1.abstract == "full"
    assert sorted(p1.sources) == ["crossref", "openalex"]
    assert [s.query_id for s in p1.source_records] == ["S001", "S002"]


def test_arxiv_version_and_published_version_merge():
    c = Corpus()
    p1, _ = c.add(r(source="arxiv", source_id="2301.00001v1", title="Deep thing", arxiv_id="2301.00001v1",
                    publication_type="preprint", year=2023), "S1")
    p2, new = c.add(r(source="semantic_scholar", source_id="s2", title="Deep Thing", arxiv_id="2301.00001",
                      doi="10.5555/pub", publication_type="journal-article", year=2024), "S2")
    assert not new and p1 is p2
    assert p1.doi == "10.5555/pub"
    assert p1.publication_type == "journal-article"  # published version supersedes preprint label


def test_title_similarity_merge_with_year_tolerance():
    c = Corpus()
    c.add(r(title="Securing the model context protocol: a threat model", year=2024), "S1")
    _, new = c.add(r(source="b", title="Securing the Model Context Protocol - A Threat Model.", year=2025), "S2")
    assert not new
    _, new3 = c.add(r(source="b", title="Securing the Model Context Protocol - A Threat Model.", year=2019), "S3")
    assert new3  # too far apart in time


def test_similar_titles_with_conflicting_dois_are_not_merged():
    c = Corpus()
    c.add(r(title="Survey of X", doi="10.1000/a", year=2024), "S1")
    _, new = c.add(r(source="b", title="Survey of X", doi="10.1000/b", year=2024), "S2")
    assert new
    assert len(c.papers) == 2


def test_retraction_flag_is_sticky_and_citations_take_max():
    c = Corpus()
    p, _ = c.add(r(doi="10.1000/z", is_retracted=False, cited_by_count=3), "S1")
    c.add(r(source="b", doi="10.1000/z", is_retracted=True, cited_by_count=10), "S2")
    assert p.is_retracted is True and p.cited_by_count == 10


def test_ids_are_stable_and_sequential():
    c = Corpus()
    a, _ = c.add(r(doi="10.1000/a", title="First"), "S1")
    b, _ = c.add(r(doi="10.1000/b", title="Second"), "S1")
    assert (a.id, b.id) == ("P0001", "P0002")
    c2 = Corpus(c.papers)  # re-indexing from persisted state
    _, new = c2.add(r(source="z", doi="10.1000/B", title="Second"), "S9")
    assert not new
