"""Crossref REST API: bibliographic search and DOI verification."""
from __future__ import annotations

from typing import Optional

from ..models import RawRecord
from .base import HttpSource, SearchOutcome, _clean_text

SELECT = "DOI,title,author,issued,container-title,type,abstract,is-referenced-by-count,URL,update-to,relation"


class CrossrefSource(HttpSource):
    name = "crossref"
    base_url = "https://api.crossref.org"
    min_interval = 0.5

    def _params(self, extra: dict) -> dict:
        p = dict(extra)
        if self.settings.mailto:
            p["mailto"] = self.settings.mailto
        return p

    def search(self, query: str, *, from_year: Optional[int] = None, to_year: Optional[int] = None,
               limit: int = 25) -> SearchOutcome:
        flt = []
        if from_year:
            flt.append(f"from-pub-date:{from_year}")
        if to_year:
            flt.append(f"until-pub-date:{to_year}")
        params = {"query.bibliographic": query, "rows": min(limit, 100), "select": SELECT}
        if flt:
            params["filter"] = ",".join(flt)
        try:
            data = self.get(f"{self.base_url}/works", self._params(params))
        except Exception as e:  # noqa: BLE001
            return self._outcome_from_error(e)
        msg = data.get("message") or {}
        recs = [self.to_record(it) for it in msg.get("items", [])]
        return SearchOutcome(records=[r for r in recs if r], total_available=msg.get("total-results"))

    def lookup_doi(self, doi: str) -> Optional[dict]:
        try:
            data = self.get(f"{self.base_url}/works/{doi}", self._params({}))
        except Exception:  # noqa: BLE001
            return None
        return data.get("message")

    @staticmethod
    def to_record(it: dict) -> Optional[RawRecord]:
        titles = it.get("title") or []
        title = titles[0] if titles else None
        if not title or not it.get("DOI"):
            return None
        t = (it.get("type") or "").lower()
        ptype = {
            "journal-article": "journal-article",
            "proceedings-article": "conference-paper",
            "posted-content": "preprint",
            "book-chapter": "book-chapter",
        }.get(t, "other")
        year = None
        parts = (it.get("issued") or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])
        authors = []
        for a in it.get("author") or []:
            name = " ".join(x for x in [a.get("given"), a.get("family")] if x) or a.get("name")
            if name:
                authors.append(name)
        container = it.get("container-title") or []
        is_retracted = None
        for upd in it.get("update-to") or []:
            if (upd.get("type") or "").lower() == "retraction":
                # this item is a retraction notice for another DOI; it is not itself a paper
                return None
        rel = it.get("relation") or {}
        if "is-retracted-by" in rel:
            is_retracted = True
        return RawRecord(
            source="crossref",
            source_id=it["DOI"],
            title=_clean_text(title) or title,
            abstract=_clean_text(it.get("abstract")),
            year=year,
            authors=authors,
            doi=it["DOI"],
            venue=container[0] if container else None,
            publication_type=ptype,
            is_retracted=is_retracted,
            cited_by_count=it.get("is-referenced-by-count"),
            url=it.get("URL"),
        )
