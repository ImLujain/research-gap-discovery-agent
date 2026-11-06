"""Semantic Scholar Academic Graph API."""
from __future__ import annotations

from typing import Optional

from ..models import RawRecord
from .base import HttpSource, SearchOutcome, _clean_text

FIELDS = "title,abstract,year,authors,externalIds,citationCount,venue,publicationTypes,openAccessPdf,publicationVenue,url"


class SemanticScholarSource(HttpSource):
    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1"
    min_interval = 1.1  # unauthenticated shared pool is ~1 req/s
    max_attempts = 5

    def __init__(self, settings=None, client=None):
        super().__init__(settings, client)
        if not self.settings.s2_api_key:
            self.max_attempts = 3  # the unauthenticated pool rarely recovers within a query; fail fast and move on

    def _headers(self) -> dict:
        return {"x-api-key": self.settings.s2_api_key} if self.settings.s2_api_key else {}

    def search(self, query: str, *, from_year: Optional[int] = None, to_year: Optional[int] = None,
               limit: int = 25) -> SearchOutcome:
        params = {"query": query, "limit": min(limit, 100), "fields": FIELDS}
        if from_year or to_year:
            params["year"] = f"{from_year or ''}-{to_year or ''}"
        try:
            data = self.get(f"{self.base_url}/paper/search", params, headers=self._headers())
        except Exception as e:  # noqa: BLE001
            return self._outcome_from_error(e)
        recs = [self.to_record(p) for p in data.get("data", [])]
        return SearchOutcome(records=[r for r in recs if r], total_available=data.get("total"))

    def citations_of(self, paper_ref: str, limit: int = 50) -> SearchOutcome:
        """Papers citing ``paper_ref`` (an S2 id, 'DOI:...' or 'ARXIV:...')."""
        params = {"fields": FIELDS, "limit": min(limit, 1000)}
        try:
            data = self.get(f"{self.base_url}/paper/{paper_ref}/citations", params, headers=self._headers())
        except Exception as e:  # noqa: BLE001
            return self._outcome_from_error(e)
        recs = [self.to_record(item.get("citingPaper") or {}) for item in data.get("data", [])]
        return SearchOutcome(records=[r for r in recs if r])

    @staticmethod
    def to_record(p: dict) -> Optional[RawRecord]:
        title = p.get("title")
        if not title or not p.get("paperId"):
            return None
        ext = p.get("externalIds") or {}
        venue = p.get("venue") or (p.get("publicationVenue") or {}).get("name")
        ptypes = [t.lower() for t in (p.get("publicationTypes") or [])]
        ptype = "unknown"
        if ext.get("ArXiv") and not ext.get("DOI") or (venue or "").lower() in ("arxiv.org", "arxiv"):
            ptype = "preprint"
        elif "review" in ptypes:
            ptype = "review"
        elif "conference" in ptypes:
            ptype = "conference-paper"
        elif "journalarticle" in ptypes:
            ptype = "journal-article"
        elif "book" in ptypes or "bookchapter" in ptypes:
            ptype = "book-chapter"
        elif ptypes:
            ptype = "other"
        return RawRecord(
            source="semantic_scholar",
            source_id=p["paperId"],
            title=title,
            abstract=_clean_text(p.get("abstract")),
            year=p.get("year"),
            authors=[a.get("name") for a in (p.get("authors") or []) if a.get("name")],
            doi=ext.get("DOI"),
            arxiv_id=ext.get("ArXiv"),
            pmid=str(ext["PubMed"]) if ext.get("PubMed") else None,
            s2_id=p["paperId"],
            venue=venue,
            publication_type=ptype,
            cited_by_count=p.get("citationCount"),
            oa_url=(p.get("openAccessPdf") or {}).get("url"),
            url=p.get("url"),
        )
