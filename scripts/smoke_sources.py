"""Run one query against every source and print counts. Usage: python scripts/smoke_sources.py "query" [sources]"""
from __future__ import annotations

import sys

from research_gap_agent.config import Settings
from research_gap_agent.sources import build_sources

query = sys.argv[1] if len(sys.argv) > 1 else "prompt injection LLM agents"
names = sys.argv[2].split(",") if len(sys.argv) > 2 else ["openalex", "semantic_scholar", "crossref", "arxiv"]
for src in build_sources(names, Settings()):
    out = src.search(query, from_year=2022, limit=5)
    print(f"{src.name:18s} status={out.status:12s} n={len(out.records):2d} total={out.total_available} err={out.error}")
    for r in out.records[:2]:
        print(f"   - [{r.publication_type}] {r.year} {r.title[:80]} doi={r.doi} abstract={'yes' if r.abstract else 'no'}")
