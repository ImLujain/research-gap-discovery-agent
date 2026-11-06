"""Literature Discovery Agent: runs queries across sources, records searches, deduplicates."""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from .dedupe import Corpus
from .models import RunState, SearchRecord
from .sources.base import SearchOutcome, Source

log = logging.getLogger(__name__)


class LiteratureDiscovery:
    def __init__(self, sources: list[Source], state: RunState, progress: Optional[Callable[[str], None]] = None):
        self.sources = sources
        self.state = state
        self.corpus = Corpus(state.papers)
        self.lock = threading.RLock()  # guards corpus/state mutation when several validations run at once
        self.progress = progress or (lambda msg: None)
        self._has_sink = progress is not None
        for src in sources:
            if hasattr(src, "on_event"):
                src.on_event = self.progress

    def _next_search_id(self) -> str:
        return f"S{len(self.state.search_records) + 1:03d}"

    def run_queries(self, queries: list[str], *, purpose: str, from_year: Optional[int], to_year: Optional[int],
                    limit: int) -> list[SearchRecord]:
        """Run every query on every source. Sources run in parallel; each source is sequential."""
        records: list[SearchRecord] = []
        stage = purpose.split(":")[0]
        n = len(queries)

        def run_source(src: Source) -> list[tuple[str, SearchOutcome]]:
            out = []
            if hasattr(src, "reset_breaker"):
                src.reset_breaker()
            for i, q in enumerate(queries, 1):
                if getattr(src, "breaker_tripped", False):
                    out.append((q, src.skipped_outcome()))
                    continue
                self.progress(f"[{stage}] {src.name}: query {i}/{n} {q!r} …")
                outcome = src.search(q, from_year=from_year, to_year=to_year, limit=limit)
                if hasattr(src, "note_outcome"):
                    src.note_outcome(outcome)
                self.progress(f"[{stage}] {src.name}: query {i}/{n} → {len(outcome.records)} records"
                              + (f" ({outcome.status}: {outcome.error})" if outcome.status != "ok" else ""))
                out.append((q, outcome))
            return out

        with ThreadPoolExecutor(max_workers=max(1, len(self.sources))) as ex:
            results = list(ex.map(run_source, self.sources))

        for src, outcomes in zip(self.sources, results):
            for q, outcome in outcomes:
                records.append(self._record(src.name, q, purpose, from_year, to_year, outcome))
        new_total = sum(r.n_new_unique for r in records)
        self.progress(f"[{stage}] batch done: {len(records)} searches, {sum(r.n_retrieved for r in records)} records, "
                      f"{new_total} new unique papers (corpus now {len(self.state.papers)})")
        return records

    def ingest_outcome(self, source_name: str, query: str, purpose: str, outcome: SearchOutcome,
                       from_year: Optional[int] = None, to_year: Optional[int] = None) -> SearchRecord:
        return self._record(source_name, query, purpose, from_year, to_year, outcome)

    def _record(self, source_name: str, query: str, purpose: str, from_year, to_year,
                outcome: SearchOutcome) -> SearchRecord:
        with self.lock:
            return self._record_locked(source_name, query, purpose, from_year, to_year, outcome)

    def _record_locked(self, source_name: str, query: str, purpose: str, from_year, to_year,
                       outcome: SearchOutcome) -> SearchRecord:
        rec = SearchRecord(
            id=self._next_search_id(),
            source=source_name,
            query=query,
            purpose=purpose,
            from_year=from_year,
            to_year=to_year,
            status=outcome.status,
            error=outcome.error,
            n_retrieved=len(outcome.records),
        )
        if outcome.status == "rate_limited":
            rec.limitation = f"{source_name} rate-limited this query; results from this source are incomplete."
        elif outcome.status == "error":
            rec.limitation = f"{source_name} returned an error for this query; results from this source are incomplete."
        new = 0
        for raw in outcome.records:
            paper, is_new = self.corpus.add(raw, rec.id)
            new += int(is_new)
            if paper.id not in rec.paper_ids:
                rec.paper_ids.append(paper.id)
        rec.n_new_unique = new
        self.state.n_records_retrieved += len(outcome.records)
        self.state.search_records.append(rec)
        if rec.limitation:
            self.state.add_limitation(rec.limitation)
        (log.debug if self._has_sink else log.info)("%s | %s | %r -> %d records (%d new)", rec.id, source_name, query,
                                                   rec.n_retrieved, new)
        return rec
