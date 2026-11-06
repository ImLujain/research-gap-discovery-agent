"""PubMed E-utilities (esearch + efetch)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

from ..models import RawRecord
from .base import HttpSource, SearchOutcome, _clean_text


class PubMedSource(HttpSource):
    name = "pubmed"
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    min_interval = 0.4

    def _params(self, extra: dict) -> dict:
        p = {"db": "pubmed", **extra}
        if self.settings.ncbi_api_key:
            p["api_key"] = self.settings.ncbi_api_key
        if self.settings.mailto:
            p["email"] = self.settings.mailto
        p["tool"] = "ResearchGapAgent"
        return p

    def search(self, query: str, *, from_year: Optional[int] = None, to_year: Optional[int] = None,
               limit: int = 25) -> SearchOutcome:
        params = {"term": query, "retmax": min(limit, 200), "retmode": "json", "sort": "relevance"}
        if from_year or to_year:
            params.update({"datetype": "pdat", "mindate": str(from_year or 1900), "maxdate": str(to_year or 3000)})
        try:
            data = self.get(f"{self.base_url}/esearch.fcgi", self._params(params))
            res = data.get("esearchresult") or {}
            ids = res.get("idlist") or []
            total = int(res.get("count") or 0)
            if not ids:
                return SearchOutcome(total_available=total)
            xml = self.get(f"{self.base_url}/efetch.fcgi", self._params({"id": ",".join(ids), "retmode": "xml"}),
                           as_text=True)
        except Exception as e:  # noqa: BLE001
            return self._outcome_from_error(e)
        return SearchOutcome(records=self.parse_efetch(xml), total_available=total)

    @staticmethod
    def parse_efetch(xml: str) -> list[RawRecord]:
        root = ET.fromstring(xml)
        recs = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title_el = art.find(".//ArticleTitle")
            title = _clean_text("".join(title_el.itertext())) if title_el is not None else None
            if not title or not pmid:
                continue
            abstract = _clean_text(" ".join("".join(t.itertext()) for t in art.findall(".//AbstractText")))
            year = art.findtext(".//JournalIssue/PubDate/Year") or art.findtext(".//ArticleDate/Year")
            authors = []
            for a in art.findall(".//AuthorList/Author"):
                name = " ".join(x for x in [a.findtext("ForeName"), a.findtext("LastName")] if x)
                if name:
                    authors.append(name)
            doi = None
            for aid in art.findall(".//ArticleIdList/ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = aid.text
            ptypes = [pt.text or "" for pt in art.findall(".//PublicationType")]
            ptype = "review" if any("Review" in p for p in ptypes) else "journal-article"
            is_retracted = any("Retracted Publication" in p for p in ptypes) or None
            recs.append(
                RawRecord(
                    source="pubmed",
                    source_id=pmid,
                    title=title,
                    abstract=abstract,
                    year=int(year) if year and year.isdigit() else None,
                    authors=authors,
                    doi=doi,
                    pmid=pmid,
                    venue=art.findtext(".//Journal/Title"),
                    publication_type=ptype,
                    is_retracted=is_retracted,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                )
            )
        return recs
