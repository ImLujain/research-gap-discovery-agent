"""Academic literature sources."""
from __future__ import annotations

from ..config import Settings
from .arxiv import ArxivSource
from .base import HttpSource, SearchOutcome, Source
from .crossref import CrossrefSource
from .github import GitHubSource
from .google_scholar import GoogleScholarSource
from .openalex import OpenAlexSource
from .pubmed import PubMedSource
from .semantic_scholar import SemanticScholarSource

SOURCE_CLASSES: dict[str, type[HttpSource]] = {
    "openalex": OpenAlexSource,
    "semantic_scholar": SemanticScholarSource,
    "crossref": CrossrefSource,
    "arxiv": ArxivSource,
    "pubmed": PubMedSource,
    "google_scholar": GoogleScholarSource,
    "github": GitHubSource,
}

NOT_SEARCHABLE = {
    "google_scholar": "Google Scholar was not searched (enable the google_scholar source with a SerpAPI key).",
    "dblp": "DBLP's search API is behind a bot-challenge page and could not be queried programmatically.",
    "ieee_xplore": "IEEE Xplore requires an institutional API key; it was not searched.",
    "acm_dl": "ACM Digital Library has no public search API; it was not searched.",
}


def build_sources(names: list[str], settings: Settings) -> list[Source]:
    out: list[Source] = []
    for n in names:
        cls = SOURCE_CLASSES.get(n)
        if cls is None:
            raise ValueError(f"Unknown source {n!r}; known: {sorted(SOURCE_CLASSES)}")
        out.append(cls(settings))
    return out


__all__ = ["Source", "SearchOutcome", "HttpSource", "build_sources", "SOURCE_CLASSES", "NOT_SEARCHABLE"]
