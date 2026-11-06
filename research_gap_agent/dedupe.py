"""Identifier normalisation and cross-source deduplication."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

from .models import Paper, RawRecord, SourceRecord

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)
_ARXIV_NEW = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?", re.I)
_ARXIV_OLD = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    m = _DOI_RE.search(doi)
    if not m:
        return None
    return m.group(0).rstrip(".,;)").lower()


def normalize_arxiv(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    value = re.sub(r"^(https?://)?(www\.)?arxiv\.org/(abs|pdf)/", "", value, flags=re.I)
    value = re.sub(r"^arxiv:", "", value, flags=re.I)
    value = re.sub(r"\.pdf$", "", value, flags=re.I)
    m = _ARXIV_NEW.fullmatch(value) or _ARXIV_OLD.fullmatch(value)
    if not m:
        return None
    return m.group(1).lower()


def normalize_title(title: Optional[str]) -> str:
    if not title:
        return ""
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def titles_similar(a: str, b: str, threshold: float = 0.93) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if abs(len(a) - len(b)) > max(len(a), len(b)) * 0.25:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


_TYPE_RANK = {"unknown": 0, "other": 1, "software": 1, "preprint": 2, "review": 3, "book-chapter": 3, "article": 3, "conference-paper": 4, "journal-article": 4}


class Corpus:
    """Deduplicated paper store with stable ids and provenance."""

    def __init__(self, papers: Optional[dict[str, Paper]] = None):
        self.papers: dict[str, Paper] = papers if papers is not None else {}
        self._by_doi: dict[str, str] = {}
        self._by_arxiv: dict[str, str] = {}
        self._by_pmid: dict[str, str] = {}
        self._by_openalex: dict[str, str] = {}
        self._by_s2: dict[str, str] = {}
        self._by_title: dict[str, list[str]] = {}
        for p in self.papers.values():
            self._index(p)

    # ------------------------------------------------------------------
    def _next_id(self) -> str:
        return f"P{len(self.papers) + 1:04d}"

    def _index(self, p: Paper) -> None:
        if p.doi:
            self._by_doi[p.doi] = p.id
        if p.arxiv_id:
            self._by_arxiv[p.arxiv_id] = p.id
        if p.pmid:
            self._by_pmid[p.pmid] = p.id
        if p.openalex_id:
            self._by_openalex[p.openalex_id] = p.id
        if p.s2_id:
            self._by_s2[p.s2_id] = p.id
        nt = normalize_title(p.title)
        if nt:
            self._by_title.setdefault(nt[:12], []).append(p.id)

    def find_match(self, rec: RawRecord) -> Optional[Paper]:
        doi = normalize_doi(rec.doi)
        if doi and doi in self._by_doi:
            return self.papers[self._by_doi[doi]]
        arx = normalize_arxiv(rec.arxiv_id)
        if arx and arx in self._by_arxiv:
            return self.papers[self._by_arxiv[arx]]
        if rec.pmid and rec.pmid in self._by_pmid:
            return self.papers[self._by_pmid[rec.pmid]]
        if rec.openalex_id and rec.openalex_id in self._by_openalex:
            return self.papers[self._by_openalex[rec.openalex_id]]
        if rec.s2_id and rec.s2_id in self._by_s2:
            return self.papers[self._by_s2[rec.s2_id]]
        nt = normalize_title(rec.title)
        if nt:
            for pid in self._by_title.get(nt[:12], []):
                cand = self.papers[pid]
                if not titles_similar(nt, normalize_title(cand.title)):
                    continue
                if rec.year and cand.year and abs(rec.year - cand.year) > 1:
                    continue
                # identifier conflict means different papers despite similar titles
                if doi and cand.doi and doi != cand.doi:
                    continue
                return cand
        return None

    def add(self, rec: RawRecord, query_id: str) -> tuple[Paper, bool]:
        """Add a raw record; returns (paper, is_new)."""
        existing = self.find_match(rec)
        src = SourceRecord(source=rec.source, source_id=rec.source_id, query_id=query_id, url=rec.url)
        if existing is not None:
            self._merge(existing, rec)
            existing.source_records.append(src)
            self._index(existing)
            return existing, False
        paper = Paper(
            id=self._next_id(),
            title=rec.title.strip(),
            abstract=rec.abstract,
            year=rec.year,
            authors=list(rec.authors),
            doi=normalize_doi(rec.doi),
            arxiv_id=normalize_arxiv(rec.arxiv_id),
            pmid=rec.pmid,
            openalex_id=rec.openalex_id,
            s2_id=rec.s2_id,
            venue=rec.venue,
            publication_type=rec.publication_type,
            is_retracted=rec.is_retracted,
            cited_by_count=rec.cited_by_count,
            oa_url=rec.oa_url,
            url=rec.url,
            source_records=[src],
        )
        self.papers[paper.id] = paper
        self._index(paper)
        return paper, True

    @staticmethod
    def _merge(p: Paper, rec: RawRecord) -> None:
        if not p.abstract and rec.abstract:
            p.abstract = rec.abstract
        elif rec.abstract and p.abstract and len(rec.abstract) > len(p.abstract) * 1.5:
            p.abstract = rec.abstract
        p.doi = p.doi or normalize_doi(rec.doi)
        p.arxiv_id = p.arxiv_id or normalize_arxiv(rec.arxiv_id)
        p.pmid = p.pmid or rec.pmid
        p.openalex_id = p.openalex_id or rec.openalex_id
        p.s2_id = p.s2_id or rec.s2_id
        p.venue = p.venue or rec.venue
        p.oa_url = p.oa_url or rec.oa_url
        p.url = p.url or rec.url
        if not p.authors and rec.authors:
            p.authors = list(rec.authors)
        if p.year is None:
            p.year = rec.year
        if rec.cited_by_count is not None:
            p.cited_by_count = max(p.cited_by_count or 0, rec.cited_by_count)
        if rec.is_retracted:
            p.is_retracted = True
        elif p.is_retracted is None and rec.is_retracted is not None:
            p.is_retracted = rec.is_retracted
        # prefer the more "published" type: a journal version supersedes a preprint record
        if _TYPE_RANK.get(rec.publication_type, 0) > _TYPE_RANK.get(p.publication_type, 0):
            p.publication_type = rec.publication_type
