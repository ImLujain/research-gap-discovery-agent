"""Research Coordinator: stage execution, persistence, resume, and the validation loop."""
from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .agents import analysis, gap_generation, planner, ranking, recommendation, screening, validation, verification
from .agents.common import goal_block
from .config import Settings
from .discovery import LiteratureDiscovery
from .export import to_bibtex, to_csv, to_ris
from .fulltext import FullTextFetcher
from .llm import LLM, screen_model_for
from .models import (
    CandidateGap, CounterEvidence, EvidenceStatement, Paper, RunOptions, RunState, ScreeningDecision,
    ValidatedGap, ValidationRound, ValidationStatus,
)
from .parallel import run_parallel
from .report import write_report_files
from .sources import NOT_SEARCHABLE, build_sources
from .sources.base import Source
from .sources.crossref import CrossrefSource
from .sources.openalex import OpenAlexSource
from .sources.semantic_scholar import SemanticScholarSource
from .traceability import filter_ids, scrub_text

log = logging.getLogger(__name__)

STAGES = ["plan", "discover", "screen", "fulltext", "extract", "landscape", "generate_gaps", "validate", "rank",
          "recommend", "verify", "report"]


class RunCancelled(Exception):
    """Raised when a stop was requested; state has been saved and the run can be resumed later."""


class Coordinator:
    def __init__(self, options: RunOptions, llm: LLM, *, settings: Optional[Settings] = None,
                 sources: Optional[list[Source]] = None, run_id: Optional[str] = None,
                 state: Optional[RunState] = None, progress: Optional[Callable[[str], None]] = None,
                 fulltext: Optional[FullTextFetcher] = None, should_stop: Optional[Callable[[], bool]] = None):
        self.settings = settings or Settings()
        self.llm = llm
        self.sources = sources if sources is not None else build_sources(options.sources, self.settings)
        self.state = state or RunState(run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
                                       options=options)
        self.run_dir = self.settings.runs_dir / self.state.run_id
        self.progress = progress or (lambda msg: log.info(msg))
        self.discovery = LiteratureDiscovery(self.sources, self.state, progress=self.progress)
        if hasattr(self.llm, "on_event"):
            self.llm.on_event = lambda msg: self.progress(f"[llm] {msg}")
        self.fulltext = fulltext or FullTextFetcher(self.settings, self.run_dir / "fulltext")
        self.should_stop = should_stop or (lambda: False)
        self.context = goal_block(self.opts.goal, self.opts.existing_work)
        self.report_path: Optional[Path] = None
        self.export_paths: dict[str, Path] = {}
        self._state_lock = threading.RLock()  # guards save() and validated_gaps when gaps are validated in parallel
        self.concurrency = max(1, int(getattr(self.settings, "llm_concurrency", 1) or 1))
        if self.opts.screen_model is None:
            self.opts.screen_model = screen_model_for(self.settings, self.llm) or ""
        self.screen_model: Optional[str] = self.opts.screen_model or None

    # ------------------------------------------------------------------ persistence
    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    def save(self) -> None:
        with self._state_lock:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.state.llm_calls = list(self.llm.call_log)
            self.state_path.write_text(self.state.model_dump_json(indent=1))

    @classmethod
    def resume(cls, run_id: str, llm: LLM, *, settings: Optional[Settings] = None,
               sources: Optional[list[Source]] = None, progress=None, should_stop=None) -> "Coordinator":
        settings = settings or Settings()
        path = settings.runs_dir / run_id / "state.json"
        state = RunState.model_validate(json.loads(path.read_text()))
        return cls(state.options, llm, settings=settings, sources=sources, state=state, progress=progress,
                   should_stop=should_stop)

    def _check_stop(self, where: str) -> None:
        if self.should_stop():
            self.save()
            self.progress(f"[stop] stopped before {where}; state saved — resume later to continue")
            raise RunCancelled(where)

    # ------------------------------------------------------------------ helpers
    @property
    def opts(self) -> RunOptions:
        return self.state.options

    @property
    def period(self) -> str:
        o = self.opts
        if o.from_year or o.to_year:
            return f"{o.from_year or 'earliest'}–{o.to_year or 'present'}"
        return "no date restriction"

    def _source(self, cls):
        return next((s for s in self.sources if isinstance(s, cls)), None)

    def _known(self) -> set[str]:
        with self.discovery.lock:
            return set(self.state.papers)

    # ------------------------------------------------------------------ run
    def run(self, until: Optional[str] = None) -> RunState:
        for stage in STAGES:
            if stage in self.state.stages_completed:
                self.progress(f"[{stage}] already completed — skipping")
                if stage == until:
                    break
                continue
            self._check_stop(stage)
            self.progress(f"[{stage}] starting")
            try:
                getattr(self, f"stage_{stage}")()
            except RunCancelled:
                raise
            except BaseException:  # includes KeyboardInterrupt: always leave a resumable state behind
                self.save()
                log.exception("Stage %s failed; state saved to %s", stage, self.state_path)
                raise
            self.state.stages_completed.append(stage)
            self.save()
            self.progress(f"[{stage}] done")
            if stage == until:
                break
        return self.state

    # ------------------------------------------------------------------ stages
    def stage_plan(self) -> None:
        self.state.plan = planner.plan_research(self.llm, self.opts, self.period)
        for a in self.state.plan.assumptions:
            self.state.add_limitation(f"Assumption: {a}")

    def stage_discover(self) -> None:
        assert self.state.plan
        queries = [q.query for q in self.state.plan.search_queries]
        self.progress(f"[discover] {len(queries)} queries × {len(self.sources)} sources "
                      f"({', '.join(s.name for s in self.sources)}); sources run in parallel, queries sequentially")
        self.discovery.run_queries(queries, purpose="primary", from_year=self.opts.from_year,
                                   to_year=self.opts.to_year, limit=self.opts.per_source_limit)
        for name, why in NOT_SEARCHABLE.items():
            self.state.add_limitation(why)
        self.state.add_limitation("Analysis relies on titles, abstracts and metadata, plus open-access full texts where they could be retrieved.")
        if not self.state.papers:
            raise RuntimeError("No records retrieved from any source; cannot continue")

    def stage_screen(self) -> None:
        assert self.state.plan
        plan = self.state.plan
        criterion = (
            f"Topic: {plan.topic}\nCore concepts: {', '.join(plan.core_concepts)}\n"
            f"In-scope related areas: {', '.join(plan.related_areas_in_scope)}\n"
            f"Out of scope: {', '.join(plan.related_areas_excluded)}"
        )
        papers = [p for p in self.state.papers.values() if p.screening is None]
        self.progress(f"[screen] screening {len(papers)} records in batches of 25"
                      + (f", {self.concurrency} batches at a time" if self.concurrency > 1 else "")
                      + (f", model {self.screen_model}" if self.screen_model else ""))
        decisions = screening.screen_papers(self.llm, papers, criterion, progress=self.progress,
                                            model=self.screen_model, concurrency=self.concurrency)
        for d in decisions:
            p = self.state.papers.get(d.paper_id)
            if p is not None:
                p.screening = ScreeningDecision(include=d.include, reason=d.reason, stage="primary")

    def _papers_to_analyze(self) -> list[Paper]:
        inc = self.state.included_papers()
        inc.sort(key=lambda p: (p.abstract is None, -(p.cited_by_count or 0), -(p.year or 0)))
        return inc[: self.opts.max_papers_to_analyze]

    def stage_fulltext(self) -> None:
        """Fetch open-access full texts for the papers that will be analysed (bounded by --max-full-text)."""
        if self.opts.max_full_text <= 0:
            self.state.add_limitation("Full-text retrieval was disabled; analysis used abstracts only.")
            return
        papers = self._papers_to_analyze()[: self.opts.max_full_text]
        got = 0
        for i, p in enumerate(papers, 1):
            self.progress(f"[fulltext] {i}/{len(papers)} {p.id} {p.title[:60]!r} …")
            if self.fulltext.fetch(p):
                got += 1
                self.progress(f"[fulltext] {p.id}: {p.full_text_chars} chars from {p.full_text_source}")
            else:
                self.progress(f"[fulltext] {p.id}: no open-access text ({p.full_text_error})")
        self.progress(f"[fulltext] open-access full text obtained for {got}/{len(papers)} papers")
        n = len(self._papers_to_analyze())
        if got < n:
            self.state.add_limitation(
                f"Full text was available and analysed for {got} of {n} analysed papers; the rest were analysed from "
                f"title/abstract only (paywalled, no OA copy, or not attempted beyond the first {self.opts.max_full_text}).")

    def stage_extract(self) -> None:
        papers = self._papers_to_analyze()
        n_inc = len(self.state.included_papers())
        if n_inc > len(papers):
            self.state.add_limitation(
                f"{n_inc} records passed screening but only the {len(papers)} with the most citations/abstracts were analysed "
                f"(--max-papers)."
            )
        texts = {p.id: t for p in papers if p.has_full_text and (t := self.fulltext.load(p))}
        self.progress(f"[extract] extracting evidence from {len(papers)} papers ({len(texts)} with full text)")
        self.state.evidence_rows = analysis.extract_evidence(self.llm, papers, self.opts.topic, full_texts=texts,
                                                             progress=self.progress, concurrency=self.concurrency)

    def stage_landscape(self) -> None:
        land = analysis.build_landscape(self.llm, self.state.evidence_rows, self.state.papers, self.opts.topic)
        known = self._known()
        rep = self.state.verification
        for t in land.themes:
            t.paper_ids = filter_ids(t.paper_ids, known, f"theme {t.id}", rep)
        kept: list[EvidenceStatement] = []
        for e in land.evidence_statements:
            e.paper_ids = filter_ids(e.paper_ids, known, f"evidence {e.id}", rep)
            if e.paper_ids:
                kept.append(e)
            else:
                rep.notes.append(f"Evidence statement {e.id} dropped: no verifiable supporting papers")
        land.evidence_statements = kept
        for c in land.conflicting_findings:
            c.paper_ids_side_a = filter_ids(c.paper_ids_side_a, known, "conflict", rep)
            c.paper_ids_side_b = filter_ids(c.paper_ids_side_b, known, "conflict", rep)
        land.foundational_paper_ids = filter_ids(land.foundational_paper_ids, known, "foundational", rep)
        land.recent_key_paper_ids = filter_ids(land.recent_key_paper_ids, known, "recent_key", rep)
        self.state.landscape = land

    def stage_generate_gaps(self) -> None:
        assert self.state.landscape
        out = gap_generation.generate_gaps(self.llm, self.state.landscape, self.state.evidence_rows, self.state.papers,
                                           self.opts.topic, self.opts.priority, context=self.context)
        known = self._known()
        ev_known = {e.id for e in self.state.landscape.evidence_statements}
        rep = self.state.verification
        gaps: list[CandidateGap] = []
        for i, g in enumerate(out.gaps, 1):
            gid = f"G{i:02d}"
            cg = CandidateGap(id=gid, **g.model_dump())
            cg.supporting_paper_ids = filter_ids(cg.supporting_paper_ids, known, f"gap {gid}", rep)
            cg.supporting_evidence_ids = [e for e in cg.supporting_evidence_ids if e in ev_known]
            # a gap must trace to at least one verifiable paper, directly or via an evidence statement
            via_ev = {pid for e in self.state.landscape.evidence_statements if e.id in cg.supporting_evidence_ids
                      for pid in e.paper_ids}
            if not cg.supporting_paper_ids and not via_ev:
                rep.notes.append(f"Candidate gap '{g.title}' dropped: no verifiable supporting evidence")
                continue
            if not cg.supporting_paper_ids:
                cg.supporting_paper_ids = sorted(via_ev)
            gaps.append(cg)
        self.state.candidate_gaps = gaps
        self.state.rejected_ideas = out.considered_and_rejected
        if not gaps:
            self.state.add_limitation("The gap generation stage produced no candidate with verifiable supporting evidence.")

    # ------------------------------------------------------------------ validation loop
    def stage_validate(self) -> None:
        """Validate each candidate gap; progress is checkpointed per gap so an interrupted stage resumes mid-way.

        Gaps are independent, so up to ``concurrency`` are validated at once. Each gap's searches go through the
        discovery lock, and each finished gap is appended to the state and saved under the state lock.
        """
        done = {v.gap.id for v in self.state.validated_gaps}
        n = len(self.state.candidate_gaps)
        pending: list[tuple[int, CandidateGap]] = []
        for i, gap in enumerate(self.state.candidate_gaps, 1):
            if gap.id in done:
                self.progress(f"[validate] gap {i}/{n} {gap.id} already validated — skipping")
            else:
                pending.append((i, gap))
        workers = min(self.concurrency, len(pending)) or 1
        self._validation_inner_concurrency = max(1, self.concurrency // workers)
        if workers > 1:
            self.progress(f"[validate] validating {len(pending)} gaps, {workers} at a time")

        def work(item: tuple[int, CandidateGap]) -> ValidatedGap:
            i, gap = item
            self._check_stop(f"validation of {gap.id}")
            self.progress(f"[validate] gap {i}/{n} {gap.id}: {gap.title}")
            result = self._validate_gap(gap)
            with self._state_lock:
                self.state.validated_gaps.append(result)
                self.save()
            self.progress(f"[validate] {gap.id} → {result.status.value}" + (" (rejected)" if result.rejected else "")
                          + " (checkpoint saved)")
            return result

        try:
            run_parallel(work, pending, workers)
        finally:
            order = {g.id: k for k, g in enumerate(self.state.candidate_gaps)}
            self.state.validated_gaps.sort(key=lambda v: order.get(v.gap.id, len(order)))

    def _validate_gap(self, original: CandidateGap) -> ValidatedGap:
        gap = original
        rounds: list[ValidationRound] = []
        found_all: dict[str, Paper] = {}
        counter: dict[str, CounterEvidence] = {}
        known = self._known
        rep = self.state.verification
        s2 = self._source(SemanticScholarSource)

        for r in range(1, self.opts.validation_rounds + 1):
            self.progress(f"[validate] {gap.id} round {r}: writing counter-queries")
            cq = validation.propose_counter_queries(self.llm, gap, self.opts.topic, context=self.context)
            purpose = f"validation:{gap.id}" + (f"r{r}" if r > 1 else "")
            recs = self.discovery.run_queries([q.query for q in cq.queries], purpose=purpose,
                                              from_year=self.opts.from_year, to_year=self.opts.to_year,
                                              limit=self.opts.validation_per_source_limit)
            # citation chasing on the papers that suggested the gap (round 1 only)
            if r == 1 and s2 is not None:
                for pid in gap.supporting_paper_ids[:3]:
                    p = self.state.papers.get(pid)
                    ref = p.s2_id if p and p.s2_id else (f"DOI:{p.doi}" if p and p.doi else (f"ARXIV:{p.arxiv_id}" if p and p.arxiv_id else None))
                    if not ref:
                        continue
                    outcome = s2.citations_of(ref, limit=25)
                    recs.append(self.discovery.ingest_outcome(s2.name, f"citations of {pid}", f"citations:{pid}", outcome))
            n_ok = sum(1 for x in recs if x.status == "ok")
            n_fail = len(recs) - n_ok
            found_ids = [pid for x in recs for pid in x.paper_ids]
            found_ids = [pid for pid in dict.fromkeys(found_ids) if pid not in gap.supporting_paper_ids]
            new_ids = [pid for pid in found_ids if pid not in found_all]
            # screen the newly found papers for relevance to this specific gap
            relevant_ids: list[str] = []
            # reuse decisions already made for this gap (e.g. by an interrupted earlier attempt) instead of re-screening
            to_screen = []
            for pid in new_ids:
                p = self.state.papers[pid]
                if p.screening and p.screening.stage == "validation" and p.screening.gap_id == gap.id:
                    if p.screening.include:
                        relevant_ids.append(pid)
                        found_all[pid] = p
                else:
                    to_screen.append(pid)
            if to_screen:
                crit = (f"Relevant to this candidate research gap (include papers that address, partially address, "
                        f"contradict, or provide preliminary evidence about it):\n{gap.title}\n{gap.what_is_missing}")
                decisions = screening.screen_papers(self.llm, [self.state.papers[i] for i in to_screen], crit,
                                                    model=self.screen_model,
                                                    concurrency=getattr(self, "_validation_inner_concurrency", 1))
                for d in decisions:
                    p = self.state.papers.get(d.paper_id)
                    if p is None:
                        continue
                    if p.screening is None:
                        p.screening = ScreeningDecision(include=d.include, reason=d.reason, stage="validation", gap_id=gap.id)
                    if d.include:
                        relevant_ids.append(d.paper_id)
                        found_all[d.paper_id] = p
            # reassess against everything relevant found so far for this gap (earlier rounds included)
            relevant_now = list(found_all.values())
            summary = (f"{len(recs)} searches: {n_ok} ok, {n_fail} failed/rate-limited. "
                       f"{len(found_ids)} papers returned, {len(new_ids)} new, {len(relevant_ids)} screened relevant this round; "
                       f"{len(relevant_now)} relevant in total across rounds.")
            supporting = [self.state.papers[i] for i in gap.supporting_paper_ids if i in self.state.papers]
            self.progress(f"[validate] {gap.id} round {r}: judging against {len(relevant_now)} relevant papers")
            judgement = validation.judge_gap(self.llm, gap, self.opts.topic, relevant_now, supporting, summary,
                                             context=self.context)
            # traceability on the judgement
            judgement.counter_evidence = [c for c in judgement.counter_evidence
                                          if c.paper_id in filter_ids([c.paper_id], known(), f"validation {gap.id}", rep)]
            if judgement.revised_gap is not None:
                judgement.revised_gap.supporting_paper_ids = filter_ids(
                    judgement.revised_gap.supporting_paper_ids, known(), f"revised {gap.id}", rep)
            if n_ok == 0:
                judgement.status = ValidationStatus.cannot_validate
                judgement.reasoning = "All validation searches failed or were rate-limited; no judgement possible. " + judgement.reasoning
            elif judgement.status == ValidationStatus.limited_evidence_found and not relevant_now and n_fail:
                judgement.confidence = "low"
                judgement.confidence_explanation += " Some validation searches failed, so coverage is incomplete."
            for c in judgement.counter_evidence:
                counter[c.paper_id] = c
            rounds.append(ValidationRound(
                round_number=r, gap_snapshot=gap, counter_queries=cq.queries, search_record_ids=[x.id for x in recs],
                n_searches_ok=n_ok, n_searches_failed=n_fail, new_paper_ids=new_ids, relevant_paper_ids=relevant_ids,
                judgement=judgement,
            ))
            # loop control
            if judgement.status == ValidationStatus.partially_addressed and judgement.revised_gap is not None \
                    and r < self.opts.validation_rounds:
                rev = judgement.revised_gap
                gap = CandidateGap(id=gap.id, revision_of=gap.id, revision_number=gap.revision_number + 1,
                                   **rev.model_dump())
                ev_known = {e.id for e in (self.state.landscape.evidence_statements if self.state.landscape else [])}
                gap.supporting_evidence_ids = [e for e in gap.supporting_evidence_ids if e in ev_known]
                if not gap.supporting_paper_ids:
                    gap.supporting_paper_ids = original.supporting_paper_ids
                self.progress(f"[validate] {gap.id} partially addressed; re-validating narrowed revision (round {r + 1})")
                continue
            break

        final = rounds[-1].judgement
        rejected = final.status == ValidationStatus.addressed
        # if the last round still proposed a revision but no rounds remain, surface it rather than silently dropping it
        if final.status == ValidationStatus.partially_addressed and final.revised_gap is not None and gap.revision_number == rounds[-1].gap_snapshot.revision_number:
            rep.notes.append(f"{gap.id}: a further narrowed revision was proposed after the last validation round; see report.")
        return ValidatedGap(
            gap=gap, original_gap=original, status=final.status, rounds=rounds,
            counter_evidence=list(counter.values()), confidence=final.confidence,
            confidence_explanation=final.confidence_explanation, feasibility=final.feasibility,
            rejected=rejected,
            rejection_reason=("Existing work already addresses this gap: " + final.reasoning) if rejected else None,
        )

    # ------------------------------------------------------------------ ranking / recommendation
    def _surviving(self) -> list[ValidatedGap]:
        return [v for v in self.state.validated_gaps if not v.rejected]

    def stage_rank(self) -> None:
        surviving = self._surviving()
        if not surviving:
            self.state.add_limitation("Every candidate gap was rejected during validation; no ranking was produced.")
            return
        self.state.ranking, self.state.ranking_trade_offs = ranking.rank_gaps(self.llm, surviving, self.opts.topic,
                                                                               context=self.context)

    def stage_recommend(self) -> None:
        assert self.state.landscape
        self.state.state_of_art = analysis.write_state_of_art(self.llm, self.state.landscape, self.state.papers,
                                                              self.opts.topic)
        if not self.state.ranking:
            return
        top_rank = self.state.ranking[0]
        top = next(v for v in self.state.validated_gaps if v.gap.id == top_rank.gap_id)
        self.state.recommendation = recommendation.recommend(self.llm, top, top_rank, self.state.papers, self.opts.topic,
                                                             context=self.context)
        rl = recommendation.reading_list(self.llm, self.state.landscape, top, self.state.papers, self.opts.topic)
        known = self._known()
        self.state.reading_list = [it for it in rl.items
                                   if it.paper_id in filter_ids([it.paper_id], known, "reading_list", self.state.verification)]

    # ------------------------------------------------------------------ verification
    def _cited_paper_ids(self) -> list[str]:
        ids: list[str] = []
        for v in self.state.validated_gaps:
            ids += v.gap.supporting_paper_ids + [c.paper_id for c in v.counter_evidence]
        ids += [r.paper_id for r in self.state.reading_list]
        if self.state.landscape:
            ids += self.state.landscape.foundational_paper_ids + self.state.landscape.recent_key_paper_ids
        return list(dict.fromkeys(i for i in ids if i in self.state.papers))

    def stage_verify(self) -> None:
        known = self._known()
        rep = self.state.verification
        if self.state.state_of_art:
            self.state.state_of_art.overview = scrub_text(self.state.state_of_art.overview, known, "state_of_art", rep)
            for t in self.state.state_of_art.themes:
                t.prose = scrub_text(t.prose, known, f"state_of_art {t.theme_id}", rep)
        if self.state.recommendation:
            rec = self.state.recommendation
            rec.evidence_unresolved = scrub_text(rec.evidence_unresolved, known, "recommendation", rep)
            rec.motivation = scrub_text(rec.motivation, known, "recommendation", rep)
        cited = [self.state.papers[i] for i in self._cited_paper_ids()]
        verification.verify_dois(cited, self._source(CrossrefSource), self._source(OpenAlexSource), rep)
        if self.state.landscape:
            used_ev = {e for v in self._surviving() for e in v.gap.supporting_evidence_ids}
            statements = [e for e in self.state.landscape.evidence_statements if e.id in used_ev]
            if statements:
                rep.claim_checks = verification.check_claims(self.llm, statements, self.state.papers)
        if rep.retracted_paper_ids:
            self.state.add_limitation(
                "Retracted papers were found among cited records: " + ", ".join(rep.retracted_paper_ids))

    def stage_report(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = write_report_files(self.state, self.run_dir)
        self.write_exports()

    def write_exports(self) -> dict[str, Path]:
        """BibTeX / RIS / CSV of the corpus (all unique records; screening status is in the CSV)."""
        papers = sorted(self.state.papers.values(), key=lambda p: p.id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name, fn in (("references.bib", to_bibtex), ("references.ris", to_ris), ("corpus.csv", to_csv)):
            path = self.run_dir / name
            path.write_text(fn(papers))
            self.export_paths[name] = path
        return self.export_paths
