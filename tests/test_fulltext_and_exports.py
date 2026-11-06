"""Full-text retrieval (Unpaywall/arXiv, PDF extraction), Google Scholar parsing, and reference exports."""
import httpx
import pytest
import respx

from research_gap_agent.config import Settings
from research_gap_agent.export import to_bibtex, to_csv, to_ris
from research_gap_agent.fulltext import FullTextFetcher, extract_pdf_text
from research_gap_agent.models import Paper, ScreeningDecision
from research_gap_agent.sources.google_scholar import GoogleScholarSource


def minimal_pdf(text: str) -> bytes:
    """Hand-built single-page PDF with one text object (enough for pypdf to extract)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_extract_pdf_text_reads_hand_built_pdf():
    assert "Hello full text" in extract_pdf_text(minimal_pdf("Hello full text"))


@respx.mock
def test_fetcher_prefers_arxiv_then_unpaywall_and_caches(tmp_path):
    long_text = "word " * 400  # > 1000 chars so it counts as extractable
    respx.get("https://arxiv.org/pdf/2301.00001").mock(
        return_value=httpx.Response(200, content=minimal_pdf(long_text), headers={"content-type": "application/pdf"}))
    respx.get("https://api.unpaywall.org/v2/10.1000/x").mock(
        return_value=httpx.Response(200, json={"best_oa_location": {"url_for_pdf": "https://repo/x.pdf"}}))
    respx.get("https://repo/x.pdf").mock(
        return_value=httpx.Response(200, content=minimal_pdf(long_text), headers={"content-type": "application/pdf"}))
    respx.get("https://doi.org/10.1000/paywalled").mock(return_value=httpx.Response(200, text="<html>", headers={"content-type": "text/html"}))
    respx.get("https://api.unpaywall.org/v2/10.1000/paywalled").mock(return_value=httpx.Response(404))
    settings = Settings(mailto="t@example.org")
    f = FullTextFetcher(settings, tmp_path / "ft")
    a = Paper(id="P0001", title="A", arxiv_id="2301.00001")
    b = Paper(id="P0002", title="B", doi="10.1000/x")
    c = Paper(id="P0003", title="C", doi="10.1000/paywalled", oa_url="https://doi.org/10.1000/paywalled")
    assert f.fetch(a) and a.full_text_source == "arxiv" and a.full_text_chars > 1000
    assert f.fetch(b) and b.full_text_source == "unpaywall"
    assert f.fetch(c) is None and c.full_text_error and "HTTP 200 text/html" in c.full_text_error
    assert (tmp_path / "ft" / "P0001.txt").exists()
    # cached: a second fetcher does not hit the network again
    calls_before = sum(r.call_count for r in respx.routes)
    assert FullTextFetcher(settings, tmp_path / "ft").fetch(a)
    assert sum(r.call_count for r in respx.routes) == calls_before


def test_google_scholar_skipped_without_key(tmp_path):
    out = GoogleScholarSource(Settings(cache_dir=tmp_path / "c", serpapi_key=None)).search("x")
    assert out.status == "skipped" and "SERPAPI_KEY" in out.error


@respx.mock
def test_google_scholar_parses_serpapi_results(tmp_path):
    body = {"search_information": {"total_results": 1234}, "organic_results": [
        {"result_id": "abc", "title": "Prompt injection in agents", "link": "https://arxiv.org/abs/2401.00001",
         "snippet": "We study...", "publication_info": {"summary": "A Smith, B Jones - arXiv preprint arXiv:2401.00001, 2024 - arxiv.org"},
         "inline_links": {"cited_by": {"total": 57}}},
        {"result_id": "def", "title": "No info", "publication_info": {}},
    ]}
    respx.get("https://serpapi.com/search.json").mock(return_value=httpx.Response(200, json=body))
    src = GoogleScholarSource(Settings(cache_dir=tmp_path / "c", serpapi_key="k"))
    src.min_interval = 0
    out = src.search("prompt injection", from_year=2023)
    assert out.status == "ok" and out.total_available == 1234 and len(out.records) == 2
    r = out.records[0]
    assert r.authors == ["A Smith", "B Jones"] and r.year == 2024 and r.cited_by_count == 57
    assert r.publication_type == "preprint" and r.abstract.startswith("[Google Scholar snippet")
    assert respx.calls[0].request.url.params["as_ylo"] == "2023"


def _papers():
    return [
        Paper(id="P0001", title="A study & more: 100% real", year=2024, authors=["Ada Lovelace", "Alan Turing"],
              doi="10.1000/a_b", venue="Journal of Things", publication_type="journal-article",
              screening=ScreeningDecision(include=True, reason="r")),
        Paper(id="P0002", title="Conference paper", year=2023, authors=["Grace Hopper"], arxiv_id="2301.00001",
              venue="NeurIPS", publication_type="conference-paper", is_retracted=True, full_text_chars=5000),
    ]


def test_bibtex_export_escapes_and_types():
    bib = to_bibtex(_papers())
    assert "@article{lovelace2024study_p0001," in bib
    assert "title = {A study \\& more: 100\\% real}" in bib
    assert "doi = {10.1000/a_b}" in bib  # DOI must not be escaped
    assert "journal = {Journal of Things}" in bib
    assert "@inproceedings{hopper2023conference_p0002," in bib and "booktitle = {NeurIPS}" in bib
    assert "eprint = {2301.00001}" in bib and "RETRACTED" in bib


def test_ris_and_csv_exports():
    ris = to_ris(_papers())
    assert "TY  - JOUR" in ris and "TY  - CONF" in ris and "AU  - Ada Lovelace" in ris and "ER  - " in ris
    csv_text = to_csv(_papers())
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("id,title,year") and len(lines) == 3
    assert "P0002" in lines[2] and ",yes," in lines[2] and lines[2].endswith(",5000")
