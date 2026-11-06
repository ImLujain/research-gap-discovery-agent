"""arXiv Atom API."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Optional

from ..models import RawRecord
from .base import HttpSource, SearchOutcome, _clean_text

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def build_query(query: str) -> str:
    """Turn a free-text query into an arXiv ``all:`` boolean query."""
    phrases = re.findall(r'"([^"]+)"', query)
    rest = re.sub(r'"[^"]+"', " ", query)
    words = [w for w in re.split(r"\s+", rest) if w and w.upper() not in ("AND", "OR", "NOT")]
    terms = [f'all:"{p}"' for p in phrases] + [f"all:{w}" for w in words]
    return " AND ".join(terms) if terms else "all:*"


class ArxivSource(HttpSource):
    name = "arxiv"
    base_url = "https://export.arxiv.org/api/query"
    min_interval = 3.1

    def search(self, query: str, *, from_year: Optional[int] = None, to_year: Optional[int] = None,
               limit: int = 25) -> SearchOutcome:
        q = build_query(query)
        if from_year or to_year:
            q = f"({q}) AND submittedDate:[{from_year or 1991}01010000 TO {to_year or 2100}12312359]"
        params = {"search_query": q, "max_results": min(limit, 100), "sortBy": "relevance"}
        try:
            text = self.get(self.base_url, params, as_text=True)
        except Exception as e:  # noqa: BLE001
            return self._outcome_from_error(e)
        try:
            recs, total = self.parse_feed(text)
        except ET.ParseError as e:
            return SearchOutcome(status="error", error=f"XML parse error: {e}")
        return SearchOutcome(records=recs, total_available=total)

    @staticmethod
    def parse_feed(text: str) -> tuple[list[RawRecord], Optional[int]]:
        root = ET.fromstring(text)
        total_el = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
        total = int(total_el.text) if total_el is not None and total_el.text else None
        recs = []
        for entry in root.findall("a:entry", NS):
            id_url = (entry.findtext("a:id", default="", namespaces=NS) or "").strip()
            arxiv_id = id_url.rsplit("/", 1)[-1]
            title = _clean_text(entry.findtext("a:title", default="", namespaces=NS))
            if not title:
                continue
            published = entry.findtext("a:published", default="", namespaces=NS) or ""
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = [a.findtext("a:name", default="", namespaces=NS) for a in entry.findall("a:author", NS)]
            doi = entry.findtext("arxiv:doi", default=None, namespaces=NS)
            journal_ref = entry.findtext("arxiv:journal_ref", default=None, namespaces=NS)
            pdf = None
            for link in entry.findall("a:link", NS):
                if link.get("title") == "pdf":
                    pdf = link.get("href")
            recs.append(
                RawRecord(
                    source="arxiv",
                    source_id=arxiv_id,
                    title=title,
                    abstract=_clean_text(entry.findtext("a:summary", default="", namespaces=NS)),
                    year=year,
                    authors=[a for a in authors if a],
                    doi=doi,
                    arxiv_id=arxiv_id,
                    venue=journal_ref or "arXiv",
                    publication_type="preprint",
                    oa_url=pdf,
                    url=id_url or None,
                )
            )
        return recs, total
