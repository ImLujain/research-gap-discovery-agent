"""Google Scholar through SerpAPI's official Google Scholar engine (requires SERPAPI_KEY).

Scraping Google Scholar directly (e.g. with the ``scholarly`` package) violates Google's terms of
service and is deliberately not used. Scholar returns no DOI or abstract; the snippet is stored as
the abstract and labelled as such.
"""
from __future__ import annotations

import re
from typing import Optional

from ..models import RawRecord
from .base import HttpSource, SearchOutcome

_YEAR = re.compile(r"\b(19|20)\d{2}\b")


class GoogleScholarSource(HttpSource):
    name = "google_scholar"
    base_url = "https://serpapi.com/search.json"
    min_interval = 1.0

    def search(self, query: str, *, from_year: Optional[int] = None, to_year: Optional[int] = None,
               limit: int = 25) -> SearchOutcome:
        if not self.settings.serpapi_key:
            return SearchOutcome(status="skipped", error="SERPAPI_KEY not set; Google Scholar not searched")
        params = {"engine": "google_scholar", "q": query, "num": min(limit, 20), "api_key": self.settings.serpapi_key,
                  "hl": "en"}
        if from_year:
            params["as_ylo"] = from_year
        if to_year:
            params["as_yhi"] = to_year
        try:
            data = self.get(self.base_url, params)
        except Exception as e:  # noqa: BLE001
            return self._outcome_from_error(e)
        if data.get("error"):
            return SearchOutcome(status="error", error=str(data["error"]))
        recs = [self.to_record(r) for r in data.get("organic_results", [])]
        total = (data.get("search_information") or {}).get("total_results")
        return SearchOutcome(records=[r for r in recs if r], total_available=total)

    @staticmethod
    def to_record(r: dict) -> Optional[RawRecord]:
        title = r.get("title")
        rid = r.get("result_id")
        if not title or not rid:
            return None
        summary = ((r.get("publication_info") or {}).get("summary") or "")
        parts = [p.strip() for p in summary.split(" - ")]
        authors = [a.strip() for a in parts[0].split(",") if a.strip()] if parts and parts[0] else []
        venue_year = parts[1] if len(parts) > 1 else ""
        m = _YEAR.search(venue_year)
        year = int(m.group(0)) if m else None
        venue = _YEAR.sub("", venue_year).strip(" ,") or None
        link = r.get("link")
        arxiv = link if link and "arxiv.org" in link else None
        cited = ((r.get("inline_links") or {}).get("cited_by") or {}).get("total")
        snippet = r.get("snippet")
        return RawRecord(
            source="google_scholar", source_id=rid, title=title,
            abstract=f"[Google Scholar snippet, not the abstract] {snippet}" if snippet else None,
            year=year, authors=authors, arxiv_id=arxiv, venue=venue,
            publication_type="preprint" if arxiv else "unknown", cited_by_count=cited, url=link,
        )
