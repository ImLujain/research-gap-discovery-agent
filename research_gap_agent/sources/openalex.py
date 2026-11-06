"""OpenAlex works search."""
from __future__ import annotations

import httpx

from typing import Optional

from ..models import RawRecord
from .base import HttpSource, SearchOutcome, _clean_text

SELECT = ",".join([
    "id", "doi", "title", "publication_year", "authorships", "abstract_inverted_index", "primary_location",
    "type", "cited_by_count", "is_retracted", "ids", "open_access",
])


def reconstruct_abstract(inv: Optional[dict]) -> Optional[str]:
    if not inv:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return _clean_text(" ".join(w for _, w in positions))


def map_type(work: dict) -> str:
    t = (work.get("type") or "").lower()
    loc = work.get("primary_location") or {}
    src = (loc.get("source") or {}) or {}
    src_type = (src.get("type") or "").lower()
    version = (loc.get("version") or "").lower()
    if t == "preprint" or src_type == "repository" or version == "submittedversion":
        return "preprint"
    if t == "review":
        return "review"
    if t == "book-chapter":
        return "book-chapter"
    if t in ("article", "paratext"):
        if src_type == "conference":
            return "conference-paper"
        if src_type == "journal":
            return "journal-article"
        return "article"
    return "other" if t else "unknown"


class OpenAlexSource(HttpSource):
    name = "openalex"
    base_url = "https://api.openalex.org"
    min_interval = 0.15

    def _auth(self, params: dict) -> None:
        # OpenAlex meters unauthenticated use against a free daily budget shared by everyone behind the same IP;
        # a free personal key has its own budget. mailto still selects the polite pool.
        if self.settings.openalex_api_key:
            params["api_key"] = self.settings.openalex_api_key
        if self.settings.mailto:
            params["mailto"] = self.settings.mailto

    def search(self, query: str, *, from_year: Optional[int] = None, to_year: Optional[int] = None,
               limit: int = 25) -> SearchOutcome:
        filters = []
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")
        params = {"search": query, "per-page": min(limit, 200), "select": SELECT, "sort": "relevance_score:desc"}
        if filters:
            params["filter"] = ",".join(filters)
        self._auth(params)
        try:
            data = self.get(f"{self.base_url}/works", params)
        except Exception as e:  # noqa: BLE001
            return self._outcome_from_error(e)
        recs = [self.to_record(w) for w in data.get("results", [])]
        return SearchOutcome(records=[r for r in recs if r], total_available=(data.get("meta") or {}).get("count"))

    def lookup_doi(self, doi: str) -> Optional[RawRecord]:
        params = {"select": SELECT}
        self._auth(params)
        try:
            data = self.get(f"{self.base_url}/works/https://doi.org/{doi}", params)
        except Exception:  # noqa: BLE001
            return None
        return self.to_record(data)

    @staticmethod
    def to_record(w: dict) -> Optional[RawRecord]:
        title = w.get("title") or w.get("display_name")
        if not title:
            return None
        ids = w.get("ids") or {}
        oa_id = (w.get("id") or "").rsplit("/", 1)[-1] or None
        doi = w.get("doi")
        pmid = (ids.get("pmid") or "").rsplit("/", 1)[-1] or None
        authors = []
        for a in w.get("authorships") or []:
            name = (a.get("author") or {}).get("display_name")
            if name:
                authors.append(name)
        loc = w.get("primary_location") or {}
        src = (loc.get("source") or {}) or {}
        landing = loc.get("landing_page_url") or ""
        arxiv_id = None
        if "arxiv.org" in landing:
            arxiv_id = landing
        return RawRecord(
            source="openalex",
            source_id=oa_id or title,
            title=title,
            abstract=reconstruct_abstract(w.get("abstract_inverted_index")),
            year=w.get("publication_year"),
            authors=authors,
            doi=doi,
            arxiv_id=arxiv_id,
            pmid=pmid,
            openalex_id=oa_id,
            venue=src.get("display_name"),
            publication_type=map_type(w),
            is_retracted=w.get("is_retracted"),
            cited_by_count=w.get("cited_by_count"),
            oa_url=(w.get("open_access") or {}).get("oa_url"),
            url=landing or None,
        )


def request_api_key(email: str, *, timeout: float = 20.0) -> str:
    """Ask OpenAlex to email a free API key to `email`. Returns a human-readable status line.

    OpenAlex sends a confirmation link; the key becomes active once the user clicks it.
    """
    try:
        resp = httpx.post("https://api.openalex.org/api-keys", json={"email": email}, timeout=timeout)
    except httpx.HTTPError as e:
        return f"Could not reach OpenAlex: {e}"
    if resp.status_code >= 400:
        return f"OpenAlex refused the request (HTTP {resp.status_code}): {resp.text[:200]}"
    return f"OpenAlex has emailed a confirmation link to {email}. Click it, then paste the key here as OPENALEX_API_KEY."

