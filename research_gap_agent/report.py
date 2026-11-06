"""Render the Research Gap Discovery Report from run state.

Three artefacts are produced from the same state:

* ``report.html`` – an interactive dashboard: summary on top, everything else behind tabs and collapsible cards.
* ``report.md``   – the eight required sections as Markdown; each gap and every long evidence trail is folded into a
                    collapsible ``<details>`` block (rendered by GitHub, VS Code and Streamlit).
* ``appendix.md`` – the full evidence matrix, the complete search log, the verification log and every record.

Markdown inside ``<details>`` blocks is separated by blank lines so common viewers render it; ``markdown="1"``
lets Python-Markdown do the same for the HTML export.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .models import Paper, RunState, ValidatedGap

PAPER_REF = re.compile(r"\[(P\d{4})\]")
BARE_REF = re.compile(r"\[?\b(P\d{4})\b\]?(?!:)")  # model prose cites ids bare or in brackets; never re-link rendered cites

STATUS_LABEL = {
    "addressed": "addressed by existing work",
    "partially_addressed": "partially addressed",
    "conflicting_evidence": "conflicting evidence",
    "preliminary_evidence": "preliminary evidence only",
    "limited_evidence_found": "limited evidence found",
    "cannot_validate": "could not be validated",
}
RELATIONSHIP_LABEL = {
    "addresses_fully": "addresses fully",
    "addresses_partially": "addresses partially",
    "conflicts": "conflicts",
    "preliminary": "preliminary",
    "tangential": "tangential",
}
MAX_VISIBLE_COUNTER = 6
MAX_KEY_PAPERS = 25
TEASER_CHARS = 220


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _md(s: str | None) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def _cite(state: RunState, pid: str) -> str:
    p = state.papers.get(pid)
    if p is None:
        return f"[{pid}]"
    url = p.best_url()
    label = f"{pid}: {p.short_cite()}"
    return f"[{label}]({url})" if url else f"[{label}]"


def _prose(text: str) -> str:
    """Make model prose render cleanly: a list that follows a sentence needs a blank line before it in Markdown."""
    lines = (text or "").split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        is_item = line.lstrip().startswith(("- ", "* ")) or bool(re.match(r"\s*\d+[.)] ", line))
        prev = lines[i - 1] if i else ""
        prev_item = prev.lstrip().startswith(("- ", "* ")) or bool(re.match(r"\s*\d+[.)] ", prev))
        if is_item and prev.strip() and not prev_item:
            out.append("")
        elif not is_item and prev_item and line.strip():
            out.append("")
        out.append(line)
    return "\n".join(out)


def _link_refs(state: RunState, text: str) -> str:
    return BARE_REF.sub(lambda m: _cite(state, m.group(1)), _prose(text))


def _cites(state: RunState, ids: list[str]) -> str:
    return ", ".join(_cite(state, i) for i in ids) or "—"


def _badge(kind: str, text: str) -> str:
    return f'<span class="badge b-{kind}">{html.escape(text)}</span>'


def _details(summary: str, body: str, open_: bool = False, cls: str = "") -> str:
    attrs = ' markdown="1"' + (" open" if open_ else "") + (f' class="{cls}"' if cls else "")
    return f"<details{attrs}>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items) if items else "- —"


def _teaser(text: str, n: int = TEASER_CHARS) -> str:
    t = " ".join((text or "").split())
    t = re.sub(r"\[?\b(P\d{4})\b\]?", r"\1", t)
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + "…"


def _paper_line(p: Paper) -> str:
    bits = [f"**{p.id}** {', '.join(p.authors[:3])}{' et al.' if len(p.authors) > 3 else ''} ({p.year or 'n.d.'}). *{p.title}*."]
    if p.venue:
        bits.append(p.venue + ".")
    bits.append(f"[{p.publication_type}{'; PREPRINT, not peer reviewed' if p.is_preprint else ''}]")
    if p.is_retracted:
        bits.append("**RETRACTED.**")
    url = p.best_url()
    if url:
        bits.append(f"<{url}>")
    if p.doi_check and p.doi_check.resolved is False:
        bits.append("(DOI did not resolve at Crossref)")
    elif p.doi_check and p.doi_check.title_match is False:
        bits.append("(Crossref title differs; verify)")
    bits.append(f"Sources: {', '.join(p.sources)}.")
    return " ".join(bits)


def _paper_flags(p: Paper) -> str:
    flags = []
    if p.is_preprint:
        flags.append("preprint")
    if p.has_full_text:
        flags.append("full text")
    if p.is_retracted:
        flags.append("**RETRACTED**")
    return f" ({', '.join(flags)})" if flags else ""


def _status(v: ValidatedGap) -> str:
    return STATUS_LABEL.get(v.status.value, v.status.value.replace("_", " "))


def _gap_badges(v: ValidatedGap, rank: int | None, with_rank: bool = True) -> str:
    g = v.gap
    parts = []
    if rank and with_rank:
        parts.append(_badge("rank", f"#{rank}"))
    parts.append(_badge("cat", g.category.value.replace("_", " ")))
    parts.append(_badge("rej" if v.rejected else f"st-{v.status.value}", ("rejected: " if v.rejected else "") + _status(v)))
    parts.append(_badge(f"conf-{v.confidence}", f"confidence {v.confidence}"))
    parts.append(_badge(f"feas-{v.feasibility.overall}", f"feasibility {v.feasibility.overall}"))
    if g.revision_number:
        parts.append(_badge("rev", f"narrowed {g.revision_number}×"))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# gap card (body only; heading/summary are added by the caller)
# ---------------------------------------------------------------------------
def _gap_body(state: RunState, v: ValidatedGap, compact: bool = False) -> str:
    g = v.gap
    out = []
    if g.revision_number:
        out.append(f"*Revision: narrowed {g.revision_number} time(s) during validation. Original title: "
                   f"“{v.original_gap.title}”.*")
    if v.rejected and v.rejection_reason:
        out.append(f"**Why it was rejected:** {_link_refs(state, v.rejection_reason)}")
    out.append("**What is missing**\n\n" + "\n".join("> " + ln for ln in _link_refs(state, g.what_is_missing).split("\n")))
    out.append(f"**What existing research has established.** {_link_refs(state, g.what_is_established)}")
    if not compact:
        out.append(f"**Why it matters.** {_link_refs(state, g.why_it_matters)}")
        out.append(f"**Potential contribution.** {_link_refs(state, g.potential_contribution)}")
    out.append(f"**Confidence ({v.confidence}).** {_link_refs(state, v.confidence_explanation)}")

    ev = {e.id: e for e in (state.landscape.evidence_statements if state.landscape else [])}
    sup = [f"- **{eid}** ({ev[eid].kind}): {ev[eid].statement} — {_cites(state, ev[eid].paper_ids)}"
           for eid in g.supporting_evidence_ids if eid in ev]
    strong = [c for c in v.counter_evidence if c.relationship in ("addresses_fully", "addresses_partially", "conflicts")]
    other = [c for c in v.counter_evidence if c not in strong]
    visible = strong[:MAX_VISIBLE_COUNTER]
    hidden = strong[MAX_VISIBLE_COUNTER:] + other
    if visible:
        out.append(_details(f"Closest existing work found by the counter-searches ({len(visible)})", "\n".join(
            f"- {_cite(state, c.paper_id)} — *{RELATIONSHIP_LABEL.get(c.relationship, c.relationship)}*: {c.note}"
            for c in visible)))
    elif not v.counter_evidence:
        out.append("**Closest existing work found by the counter-searches:** none identified "
                   "(this is not proof that none exists).")
    body = ["**Supporting evidence statements**\n\n" + ("\n".join(sup) if sup else "- (no evidence statements)"),
            f"**Supporting papers:** {_cites(state, g.supporting_paper_ids)}"]
    if hidden:
        body.append(f"**Further related work ({len(hidden)})**\n\n" + "\n".join(
            f"- {_cite(state, c.paper_id)} — *{RELATIONSHIP_LABEL.get(c.relationship, c.relationship)}*: {c.note}"
            for c in hidden))
    body.append(f"**Direct evidence vs. agent interpretation.** {_link_refs(state, g.direct_vs_interpretation)}")
    out.append(_details(f"Evidence detail — {len(sup)} supporting statements, {len(g.supporting_paper_ids)} supporting "
                        f"papers, {len(v.counter_evidence)} related records examined", "\n\n".join(body)))

    trail = []
    for r in v.rounds:
        recs = [s for s in state.search_records if s.id in r.search_record_ids]
        trail.append(
            f"**Round {r.round_number}** → status *{STATUS_LABEL.get(r.judgement.status.value, r.judgement.status.value)}*. "
            f"{len(r.counter_queries)} counter-queries, {len(recs)} searches ({r.n_searches_ok} ok, "
            f"{r.n_searches_failed} failed or rate-limited), {len(r.new_paper_ids)} new records, "
            f"{len(r.relevant_paper_ids)} judged relevant.\n\n"
            + "\n".join(f"- `{q.query}`" for q in r.counter_queries)
            + f"\n\n*Judgement.* {_link_refs(state, r.judgement.reasoning)}")
    last = v.rounds[-1].judgement if v.rounds else None
    if last and last.what_could_not_be_established:
        trail.append(f"**What could not be established.** {_link_refs(state, last.what_could_not_be_established)}")
    f = v.feasibility
    trail.append(f"**Feasibility ({f.overall}).** Data: {f.data_availability} Resources and tools: {f.resources_and_tools} "
                 f"Technical: {f.technical_requirements} Evaluation challenges: {f.evaluation_challenges}")
    out.append(_details(f"Validation trail — {len(v.rounds)} round(s), final status: {_status(v)}", "\n\n".join(trail)))
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# report parts (shared by the Markdown and HTML renderers)
# ---------------------------------------------------------------------------
@dataclass
class GapPart:
    id: str
    title: str
    rank: int | None
    rejected: bool
    status_key: str
    status: str
    confidence: str
    feasibility: str
    category: str
    teaser: str
    badges: str
    body_md: str


@dataclass
class ReportParts:
    topic: str
    run_id: str
    today: str
    metrics: list[tuple[str, str]]
    recommended_md: str = ""
    ranked_table_md: str = ""
    ranked_notes_md: str = ""
    themes_md: str = ""
    scope_counts_md: str = ""
    limitations: list[str] = field(default_factory=list)
    sota_md: str = ""
    key_papers_md: str = ""
    key_papers_note: str = ""
    gaps_intro: str = ""
    gaps: list[GapPart] = field(default_factory=list)
    rejected: list[GapPart] = field(default_factory=list)
    rejected_table_md: str = ""
    ideas_md: str = ""
    comparison_md: str = ""
    recommendation_md: str = ""
    reading_md: str = ""
    methodology_md: str = ""
    references_md: str = ""
    n_references: int = 0


def build_parts(state: RunState) -> ReportParts:
    o = state.options
    plan = state.plan
    land = state.landscape
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    searched = sorted({s.source for s in state.search_records if s.status == "ok"})
    attempted = sorted({s.source for s in state.search_records})
    failed_only = [s for s in attempted if s not in searched]
    n_unique = len(state.papers)
    n_screened = len(state.primary_screened())
    n_val_screened = len(state.validation_screened())
    n_included = len(state.included_papers())
    n_analyzed = len(state.evidence_rows)
    n_ft = len(state.full_text_analyzed_ids())
    dates = sorted({s.executed_at.strftime("%Y-%m-%d") for s in state.search_records})
    period = f"{o.from_year or 'earliest'}–{o.to_year or 'present'}" if (o.from_year or o.to_year) else "no date restriction"
    surviving = [v for v in state.validated_gaps if not v.rejected]
    rejected = [v for v in state.validated_gaps if v.rejected]
    by_id = {v.gap.id: v for v in state.validated_gaps}
    rank_of = {r.gap_id: r.rank for r in state.ranking}
    ranked_vs = [by_id[r.gap_id] for r in state.ranking if r.gap_id in by_id]
    unranked = [v for v in surviving if v.gap.id not in rank_of]

    P = ReportParts(topic=o.topic, run_id=state.run_id, today=today, metrics=[
        ("Unique records", str(n_unique)), ("Screened", str(n_screened + n_val_screened)), ("Analysed", str(n_analyzed)),
        ("Full texts", str(n_ft)), ("Searches run", str(len(state.search_records))),
        ("Candidate gaps", str(len(state.candidate_gaps))), ("Surviving gaps", str(len(surviving)))])

    # ---- summary pieces
    if state.ranking and state.recommendation and state.recommendation.gap_id in by_id:
        top = by_id[state.recommendation.gap_id]
        P.recommended_md = (f"**{top.gap.id} — {top.gap.title}** ({_status(top)}; confidence {top.confidence}; "
                            f"feasibility {top.feasibility.overall}). Proposed study: *{state.recommendation.proposed_title}*.")
    if state.ranking:
        P.ranked_table_md = ("| # | Gap | Category | Validation | Confidence | Feasibility |\n|:--:|---|---|---|---|---|\n"
                             + "\n".join(
                                 f"| {r.rank} | **{r.gap_id}** {_md(by_id[r.gap_id].gap.title)} | {by_id[r.gap_id].gap.category.value} "
                                 f"| {_status(by_id[r.gap_id])} | {by_id[r.gap_id].confidence} | {by_id[r.gap_id].feasibility.overall} |"
                                 for r in state.ranking if r.gap_id in by_id))
        P.ranked_notes_md = "\n".join(f"{r.rank}. **{r.gap_id}** — {_link_refs(state, r.overall_rationale)}" for r in state.ranking)
    else:
        P.ranked_notes_md = "**Most promising research opportunities:** none — no candidate survived validation."
    if land:
        P.themes_md = "\n".join(f"- **{t.id} {t.name}** — {t.description}" for t in land.themes)
    rows = [
        ("Research topic", o.topic),
        ("Scope of investigation", (f"Discipline: {o.discipline or 'not specified'}; core concepts: "
                                    + ", ".join(plan.core_concepts) if plan else "—")),
        ("Databases actually searched", ", ".join(searched) or "none"),
        ("Databases attempted but failing/rate-limited on every query", ", ".join(failed_only) or "none"),
        ("Databases not searched", "Google Scholar (no API), IEEE Xplore and ACM DL (no authorised access)"
                                   + ("" if "pubmed" in attempted else ", PubMed (not enabled)")),
        ("Search dates", ", ".join(dates) or "—"),
        ("Publication period", period),
        ("Records retrieved (all sources, before deduplication)", str(state.n_records_retrieved)),
        ("Unique records after deduplication", str(n_unique)),
        ("Records screened at title/abstract level (primary search)", str(n_screened)),
        ("Records included after primary screening", str(n_included)),
        ("Additional records found and screened by validation searches", str(n_val_screened)),
        ("Papers analysed (abstract-level structured extraction)", str(n_analyzed)),
        ("Full-text papers analysed (open-access PDFs)", str(n_ft)),
        ("Candidate gaps generated / rejected in validation / surviving",
         f"{len(state.candidate_gaps)} / {len(rejected)} / {len(surviving)}"),
    ]
    P.scope_counts_md = "| Item | Value |\n|---|---|\n" + "\n".join(f"| {k} | {_md(v)} |" for k, v in rows)
    P.limitations = list(state.limitations)

    # ---- state of the art
    S: list[str] = []
    if state.state_of_art:
        S.append(_link_refs(state, state.state_of_art.overview))
    if land:
        prose = {t.theme_id: t.prose for t in (state.state_of_art.themes if state.state_of_art else [])}
        for t in land.themes:
            block = [f"*{t.description}*"]
            if t.id in prose:
                block.append(_link_refs(state, prose[t.id]))
            block.append(f"**Representative publications:** {_cites(state, t.paper_ids)}")
            block.append(f"**Established approaches**\n\n{_bullets(t.established_approaches)}\n\n"
                         f"**Important contributions**\n\n{_bullets(t.important_contributions)}\n\n"
                         f"**Strengths**\n\n{_bullets(t.strengths)}\n\n**Weaknesses**\n\n{_bullets(t.weaknesses)}\n\n"
                         f"**Unresolved questions**\n\n{_bullets(t.unresolved_questions)}")
            S.append(_details(f"<b>{t.id}.</b> {html.escape(t.name)}", "\n\n".join(block), cls="theme"))
        if land.conflicting_findings:
            S.append("**Conflicting findings**\n\n" + "\n".join(
                f"- {c.description} — side A: {_cites(state, c.paper_ids_side_a)}; side B: {_cites(state, c.paper_ids_side_b)}"
                for c in land.conflicting_findings))
        S.append(_details(f"Cross-paper evidence statements ({len(land.evidence_statements)})", "\n".join(
            f"- **{e.id}** ({e.kind}): {e.statement} — {_cites(state, e.paper_ids)}" for e in land.evidence_statements)))
    P.sota_md = "\n\n".join(S)

    # ---- key papers
    key_ids: list[str] = []
    for v in ranked_vs + unranked:
        key_ids += v.gap.supporting_paper_ids
        key_ids += [c.paper_id for c in v.counter_evidence if c.relationship in ("addresses_fully", "addresses_partially")]
    key_ids += [r.paper_id for r in state.reading_list]
    analysed = {r.paper_id: r for r in state.evidence_rows}
    seen: list[str] = []
    for pid in key_ids:
        if pid in analysed and pid in state.papers and pid not in seen:
            seen.append(pid)
    key_rows = [analysed[p] for p in seen[:MAX_KEY_PAPERS]] or state.evidence_rows[:MAX_KEY_PAPERS]
    P.key_papers_note = (f"The {len(key_rows)} analysed papers most relevant to the gaps. The complete matrix for all "
                         f"{n_analyzed} analysed papers, with methodology and limitations columns, is in `appendix.md`, Section A.")
    tbl = "| Paper | Year | Main findings | Relevance to potential gaps |\n|---|:--:|---|---|"
    for r in key_rows:
        p = state.papers.get(r.paper_id)
        if p:
            tbl += (f"\n| {_cite(state, p.id)}{_paper_flags(p)} | {p.year or 'n.d.'} | {_md(r.main_findings)} "
                    f"| {_md(r.relevance_to_potential_gaps)} |")
    P.key_papers_md = tbl

    # ---- gaps
    if not state.validated_gaps:
        P.gaps_intro = "No candidate gaps with verifiable supporting evidence were generated."
    else:
        P.gaps_intro = (f"{len(surviving)} of {len(state.validated_gaps)} candidate gaps survived adversarial validation. "
                        "Open a card for what is missing, the closest existing work, the evidence and the validation trail.")

    def part(v: ValidatedGap, rank: int | None, compact: bool = False) -> GapPart:
        return GapPart(id=v.gap.id, title=v.gap.title, rank=rank, rejected=v.rejected, status_key=v.status.value,
                       status=_status(v), confidence=v.confidence, feasibility=v.feasibility.overall,
                       category=v.gap.category.value.replace("_", " "), teaser=_teaser(v.gap.what_is_missing),
                       badges=_gap_badges(v, rank), body_md=_gap_body(state, v, compact=compact))

    P.gaps = [part(v, rank_of.get(v.gap.id)) for v in ranked_vs + unranked]
    P.rejected = [part(v, None, compact=True) for v in rejected]
    if rejected:
        P.rejected_table_md = "| Gap | Final status | Reason |\n|---|---|---|\n" + "\n".join(
            f"| **{v.gap.id}** {_md(v.gap.title)} | {_status(v)} | {_md(_link_refs(state, v.rejection_reason or ''))} |"
            for v in rejected)
    if state.rejected_ideas:
        P.ideas_md = "\n".join(f"- {i.idea} — {i.why_not}" for i in state.rejected_ideas)

    # ---- comparison
    if state.ranking:
        C = ["| Research gap | Evidence the problem is unresolved | Potential contribution | Feasibility | Main risk |\n|---|---|---|---|---|"]
        for r in state.ranking:
            v = by_id[r.gap_id]
            ev = next((x.rationale for x in r.ratings if x.criterion == "evidence_problem_unresolved"), "")
            pc = next((x.rationale for x in r.ratings if x.criterion == "potential_contribution"), "")
            C[-1] += (f"\n| **{r.gap_id}** {_md(v.gap.title)} ({_status(v)}) | {_md(ev)} | {_md(pc)} "
                      f"| {v.feasibility.overall} — {_md(v.feasibility.evaluation_challenges)} | {_md(r.main_risk)} |")
        crit = _criteria(state)
        C.append("**Rubric ratings**\n\n| Gap | Rank | " + " | ".join(c.replace('_', ' ') for c in crit) + " |\n|---|:--:|"
                 + ":--:|" * len(crit))
        for r in state.ranking:
            rt = {x.criterion: x.rating for x in r.ratings}
            C[-1] += f"\n| {r.gap_id} | {r.rank} | " + " | ".join(rt.get(c, "—") for c in crit) + " |"
        C.append("Ratings are qualitative judgements by the ranking agent (high = 2, medium = 1, low = 0 for ordering only); "
                 "they are not statistically validated measurements.")
        C.append("**Trade-offs.** " + _link_refs(state, state.ranking_trade_offs))
        P.comparison_md = "\n\n".join(C)
    else:
        P.comparison_md = "No gaps survived validation, so no comparison is possible."

    # ---- recommendation
    rec = state.recommendation
    if rec:
        v = by_id.get(rec.gap_id)
        R = [f"Based on gap **{rec.gap_id}** (validation status: {_status(v) if v else 'n/a'}; "
             f"confidence: {v.confidence if v else 'n/a'}).",
             f"### {rec.proposed_title}",
             f"**Problem statement.** {_link_refs(state, rec.problem_statement)}",
             f"**Aim.** {_link_refs(state, rec.aim)}",
             "**Research questions**\n\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(rec.research_questions, 1)),
             _details("Motivation and evidence that the problem remains unresolved",
                      f"**Motivation.** {_link_refs(state, rec.motivation)}\n\n"
                      f"**Evidence that the problem remains unresolved.** {_link_refs(state, rec.evidence_unresolved)}"),
             "**Hypotheses**\n\n" + (_bullets(rec.hypotheses) if rec.hypotheses else "- none proposed"),
             f"**Proposed methodology.** {_link_refs(state, rec.methodology)}",
             "| | |\n|---|---|\n"
             f"| **Datasets, benchmarks, resources** | {_md('; '.join(rec.datasets_benchmarks_resources))} |\n"
             f"| **Baselines and comparison methods** | {_md('; '.join(rec.baselines))} |\n"
             f"| **Evaluation metrics** | {_md('; '.join(rec.evaluation_metrics))} |",
             "**Expected contributions**\n\n" + _bullets(rec.expected_contributions),
             _details("Potential limitations and the plan to validate novelty before committing",
                      "**Potential limitations**\n\n" + _bullets(rec.potential_limitations)
                      + "\n\n**Plan to validate novelty before committing**\n\n" + _bullets(rec.novelty_validation_plan)),
             "*No experimental results are claimed; the proposal's performance relative to existing methods is unknown until tested.*"]
        P.recommendation_md = "\n\n".join(R)
    else:
        P.recommendation_md = "No recommendation: no candidate gap survived validation."

    # ---- reading list
    if state.reading_list:
        order = ["foundational", "survey", "recent_key", "directly_related", "challenges_direction"]
        blocks = []
        for role in order:
            items = [r for r in state.reading_list if r.role == role and r.paper_id in state.papers]
            if items:
                blocks.append(f"**{role.replace('_', ' ').title()}**\n\n" + "\n".join(
                    f"- {_cite(state, r.paper_id)} — *{state.papers[r.paper_id].title}*. {r.why_it_matters}" for r in items))
        P.reading_md = "\n\n".join(blocks)
    else:
        P.reading_md = "No reading list was produced."

    # ---- methodology
    per_source: dict[str, dict[str, int]] = {}
    for s in state.search_records:
        d = per_source.setdefault(s.source, {"n": 0, "ok": 0, "records": 0})
        d["n"] += 1
        d["ok"] += s.status == "ok"
        d["records"] += s.n_retrieved
    M = [f"**Publication period:** {period}. **Sources enabled:** {', '.join(o.sources)}. "
         f"**Search dates:** {', '.join(dates) or '—'}."]
    if per_source:
        M.append("| Source | Searches | Succeeded | Failed or rate-limited | Records retrieved |\n|---|:--:|:--:|:--:|:--:|\n"
                 + "\n".join(f"| {src} | {d['n']} | {d['ok']} | {d['n'] - d['ok']} | {d['records']} |"
                             for src, d in sorted(per_source.items())))
    M.append("**Inclusion criteria:** records addressing the topic, closely related problems, or methods clearly applicable "
             "to it, as judged at title/abstract level. **Exclusion criteria:** off-topic records, non-research items, "
             "records with no usable title/abstract information, retraction notices.")
    M.append("**Screening procedure:** LLM-assisted title/abstract screening with a recorded reason per record; validation "
             "searches were screened for relevance to the specific candidate gap. **Deduplication:** by DOI, arXiv id, "
             "PubMed id, OpenAlex id, Semantic Scholar id, then normalised title similarity with ±1 year tolerance; merged "
             "records keep every source as provenance.")
    ft_ids = state.full_text_analyzed_ids()
    ft_fail = [p for p in state.papers.values() if p.full_text_error]
    M.append(f"**Full-text availability:** open-access full text was retrieved and used for {len(ft_ids)} analysed papers; "
             f"{len(ft_fail)} attempted retrievals failed (paywalled or no extractable text). Other papers were analysed "
             f"from title/abstract; {sum(1 for r in state.evidence_rows if not state.papers[r.paper_id].abstract)} analysed "
             f"records had no abstract.")
    M.append("**Known coverage limitations:** " + (" ".join(
        x for x in state.limitations if 'not searched' in x or 'rate-limited' in x or 'error' in x) or "none recorded."))
    if plan:
        M.append(_details("Planning assumptions, topic decomposition and research questions",
                          "**Assumptions**\n\n" + _bullets(plan.assumptions)
                          + "\n\n**Topic decomposition.** Core concepts: " + ", ".join(plan.core_concepts)
                          + ". Alternative terminology: " + ", ".join(plan.alternative_terminology)
                          + ". In-scope related areas: " + ", ".join(plan.related_areas_in_scope)
                          + ". Excluded: " + ", ".join(plan.related_areas_excluded) + "."
                          + "\n\n**Research questions**\n\n" + _bullets([q.question for q in plan.research_questions])))
    vr = state.verification
    M.append(f"**Verification:** {len(vr.removed_unknown_ids)} unverifiable paper ids removed from model output; "
             f"{len(vr.doi_checks)} DOIs checked against Crossref ({sum(1 for d in vr.doi_checks if not d.resolved)} unresolved, "
             f"{sum(1 for d in vr.doi_checks if d.title_match is False)} title mismatches); retracted papers among cited "
             f"records: {', '.join(vr.retracted_paper_ids) or 'none detected'}.")
    if vr.claim_checks:
        weak = [c for c in vr.claim_checks if c.verdict in ("not_supported", "partially_supported", "cannot_assess")]
        if weak:
            M.append("**Evidence statements whose cited abstracts do not fully support them**\n\n" + "\n".join(
                f"- {c.evidence_id} ({c.verdict}): {c.note}" for c in weak))
    calls = state.llm_calls
    if calls:
        M.append(f"**LLM usage:** {len(calls)} calls; {sum(c.input_tokens for c in calls):,} input tokens, "
                 f"{sum(c.output_tokens for c in calls):,} output tokens; models: {', '.join(sorted({c.model for c in calls}))}.")
    M.append(f"**Exports:** `references.bib`, `references.ris` and `corpus.csv` next to this report hold every unique record "
             f"for import into Zotero/Mendeley/EndNote. The complete search log ({len(state.search_records)} searches) and "
             f"verification log are in `appendix.md`.")
    P.methodology_md = "\n\n".join(M)

    # ---- references: every id cited anywhere above
    everything = "\n".join([P.recommended_md, P.ranked_notes_md, P.sota_md, P.key_papers_md, P.comparison_md,
                            P.recommendation_md, P.reading_md, P.methodology_md]
                           + [g.body_md for g in P.gaps + P.rejected] + [P.rejected_table_md])
    cited_ids = set(PAPER_REF.findall(everything)) | set(re.findall(r"\[(P\d{4}): ", everything))
    refs = [_paper_line(state.papers[pid]) for pid in sorted(cited_ids) if pid in state.papers]
    P.references_md = "\n".join(f"- {r}" for r in refs)
    P.n_references = len(refs)
    return P


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def render_report(state: RunState) -> str:
    P = build_parts(state)
    L: list[str] = []
    L.append(f"# Research Gap Discovery Report\n\n**Topic:** {P.topic}\n\n"
             f"*Generated {P.today} · run `{P.run_id}` · the full evidence matrix, search log and reference list are "
             f"in `appendix.md` next to this file. `report.html` is the interactive version.*")
    L.append("> This report reflects the literature retrieved and analysed within the documented search scope (Section 8). "
             "It is based on titles, abstracts and metadata, plus open-access full texts where retrievable. Absence of a "
             "result is never treated as proof that no prior work exists.")

    L.append("## 1. Executive Summary")
    L.append("| " + " | ".join(k for k, _ in P.metrics) + " |\n|" + ":---:|" * len(P.metrics) + "\n| "
             + " | ".join(v for _, v in P.metrics) + " |")
    if P.recommended_md:
        L.append("**Recommended direction:** " + P.recommended_md + " See Section 6.")
    if P.ranked_table_md:
        L.append("**Research opportunities, ranked**\n\n" + P.ranked_table_md)
    L.append(P.ranked_notes_md)
    if P.themes_md:
        L.append("**Main research themes**\n\n" + P.themes_md)
    L.append(_details("Scope and counts", P.scope_counts_md))
    L.append("**Important limitations of the investigation**\n\n" + _bullets(P.limitations))

    L.append("## 2. Current State of the Art\n\n" + P.sota_md)

    L.append("## 3. Literature Evidence Matrix\n\n" + P.key_papers_note + "\n\n" + P.key_papers_md)

    L.append("## 4. Identified Research Gaps\n\n" + P.gaps_intro)
    for g in P.gaps:
        head = f"### {g.id}: {g.title}"
        L.append(_details(f"<b>{g.id}</b> · {html.escape(g.title)} · <i>{g.status}</i>",
                          head + "\n\n" + g.badges + "\n\n" + g.body_md, cls="gap"))
    if P.rejected:
        L.append("### Candidates rejected during validation\n\n" + P.rejected_table_md)
        for g in P.rejected:
            head = f"### {g.id}: {g.title} *(rejected after validation)*"
            L.append(_details(f"<b>{g.id}</b> · {html.escape(g.title)} · <i>rejected: {g.status}</i>",
                              head + "\n\n" + g.badges + "\n\n" + g.body_md, cls="gap"))
    if P.ideas_md:
        L.append(_details(f"Ideas considered and not proposed as gaps ({P.ideas_md.count(chr(10)) + 1})", P.ideas_md))

    L.append("## 5. Research Gap Comparison\n\n" + P.comparison_md)
    L.append("## 6. Recommended Research Direction\n\n" + P.recommendation_md)
    L.append("## 7. Important Papers to Read\n\n" + P.reading_md)
    L.append("## 8. Search Methodology and Reproducibility\n\n" + P.methodology_md)
    L.append("## References (all cited records)\n\n" + _details(f"{P.n_references} records cited in this report", P.references_md))
    return "\n\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# appendix
# ---------------------------------------------------------------------------
def render_appendix(state: RunState) -> str:
    o = state.options
    L: list[str] = [f"# Appendix — Research Gap Discovery Report\n\n**Topic:** {o.topic} · run `{state.run_id}`"]

    L.append("## A. Full literature evidence matrix")
    L.append("| Paper | Year | Research objective | Methodology | Main findings | Limitations | Relevance to potential gaps |\n"
             "|---|:--:|---|---|---|---|---|")
    for r in state.evidence_rows:
        p = state.papers.get(r.paper_id)
        if p:
            L[-1] += (f"\n| {_cite(state, p.id)}{_paper_flags(p)} | {p.year or 'n.d.'} | {_md(r.research_objective)} "
                      f"| {_md(r.methodology)} | {_md(r.main_findings)} | {_md('; '.join(r.limitations))} "
                      f"| {_md(r.relevance_to_potential_gaps)} |")

    L.append("## B. Search log")
    L.append("| Id | Source | Query | Purpose | Date | Period | Retrieved | New unique | Status |\n|---|---|---|---|---|---|:--:|:--:|---|")
    for s in state.search_records:
        L[-1] += (f"\n| {s.id} | {s.source} | {_md(s.query)} | {s.purpose} | {s.executed_at.strftime('%Y-%m-%d')} "
                  f"| {s.from_year or ''}–{s.to_year or ''} | {s.n_retrieved} | {s.n_new_unique} "
                  f"| {s.status}{(' (' + _md(s.error) + ')') if s.error else ''} |")

    vr = state.verification
    L.append("## C. Verification log")
    L.append(f"- Unknown paper ids removed from LLM output: {len(vr.removed_unknown_ids)}"
             + (" (" + ", ".join(f"{r.where}: {r.paper_id}" for r in vr.removed_unknown_ids) + ")" if vr.removed_unknown_ids else "")
             + f"\n- DOIs checked against Crossref: {len(vr.doi_checks)}; unresolved: "
             f"{sum(1 for d in vr.doi_checks if not d.resolved)}; title mismatches: {sum(1 for d in vr.doi_checks if d.title_match is False)}"
             + f"\n- Retracted papers among cited records: {', '.join(vr.retracted_paper_ids) or 'none detected'}"
             + (f"\n- Claim-support checks on evidence statements used by surviving gaps: "
                + ", ".join(f"{c.evidence_id}={c.verdict}" for c in vr.claim_checks) if vr.claim_checks else "")
             + ("\n- Notes: " + "; ".join(vr.notes) if vr.notes else ""))
    if vr.doi_checks:
        L.append("| DOI | Resolved | Title matches | Note |\n|---|:--:|:--:|---|\n" + "\n".join(
            f"| {d.doi} | {'yes' if d.resolved else 'no'} | "
            f"{'—' if d.title_match is None else ('yes' if d.title_match else 'no')} | {_md(d.note)} |" for d in vr.doi_checks))

    L.append("## D. All unique records in the corpus")
    L.append(f"{len(state.papers)} records. Screening decisions and provenance are in `corpus.csv`; BibTeX/RIS exports are next to this file.")
    L.append("\n".join("- " + _paper_line(p) for p in sorted(state.papers.values(), key=lambda p: p.id)))
    return "\n\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------
_CSS = """
:root{--ink:#1f2937;--muted:#6b7280;--line:#e5e7eb;--bg:#f6f7fb;--card:#ffffff;--soft:#f8fafc;--accent:#2563eb;
 --ok:#15803d;--warn:#b45309;--bad:#b91c1c;--info:#1d4ed8;--grey:#4b5563}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
 color:var(--ink);background:var(--bg);line-height:1.55;font-size:15px}
.wrap{max-width:1080px;margin:0 auto;padding:28px 24px 80px}
header.hero{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:26px 30px;margin-bottom:18px}
.hero h1{font-size:1.9rem;line-height:1.2;margin:0 0 .25em}
.hero .topic{font-size:1.15rem;color:var(--grey);margin:0 0 .6em}
.meta{color:var(--muted);font-size:.9em}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:10px;margin:18px 0 6px}
.metric{background:var(--soft);border:1px solid var(--line);border-radius:12px;padding:10px 12px;text-align:center}
.metric b{display:block;font-size:1.5rem;line-height:1.2;color:var(--accent)}
.metric span{font-size:.78em;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.callout{border-left:4px solid var(--accent);background:#eff6ff;border-radius:0 10px 10px 0;padding:.7em 1em;margin:.8em 0}
.callout.rec{border-color:var(--ok);background:#f0fdf4}
.callout p{margin:.25em 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 24px;margin:14px 0}
.card h2{margin:.1em 0 .6em;font-size:1.2rem}
h2{font-size:1.3rem;margin:1.6em 0 .6em}h3{font-size:1.08rem;margin:1.4em 0 .5em}h4{margin:1.1em 0 .4em}
p,li{max-width:80ch}
a{color:var(--accent)}
blockquote{margin:.6em 0 1em;padding:.7em 1em;border-left:4px solid var(--accent);background:#eff6ff;border-radius:0 8px 8px 0}
blockquote p{margin:.2em 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:.6em 0 1.2em;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:7px 10px;vertical-align:top;text-align:left}
th{background:var(--soft)}tbody tr:nth-child(even){background:#fcfcfd}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:.9em}
details{margin:.7em 0 1em;border:1px solid var(--line);border-radius:10px;background:var(--soft)}
details>summary{cursor:pointer;padding:.55em .9em;font-weight:600;color:var(--grey);list-style:none}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"▸ ";color:var(--accent)}details[open]>summary::before{content:"▾ "}
details>summary:hover{color:var(--accent)}
details>*:not(summary){margin-left:1em;margin-right:1em}details>:last-child{padding-bottom:.7em}
details.gap{background:var(--card);border-left:5px solid var(--line)}
details.gap>summary{padding:.8em 1em;font-weight:500;color:var(--ink);line-height:1.45}
details.gap>summary::before{content:""}
details.gap .gtitle{font-weight:650;font-size:1.02em}
details.gap .teaser{display:block;color:var(--muted);font-size:.92em;font-weight:400;margin-top:.25em}
details.gap.s-limited_evidence_found,details.gap.s-preliminary_evidence{border-left-color:var(--ok)}
details.gap.s-partially_addressed,details.gap.s-conflicting_evidence{border-left-color:var(--warn)}
details.gap.s-addressed,details.gap.rejected{border-left-color:var(--bad)}
details.gap h3{margin-top:.6em}
.badge{display:inline-block;font-size:11.5px;font-weight:600;padding:2px 9px;border-radius:999px;margin:0 5px 4px 0;
 background:#e5e7eb;color:var(--grey);letter-spacing:.01em;vertical-align:middle}
.b-cat{background:#ede9fe;color:#5b21b6}.b-rank{background:#1d4ed8;color:#fff}
.b-rev{background:#fef3c7;color:var(--warn)}
.b-rej,.b-st-addressed{background:#fee2e2;color:var(--bad)}
.b-st-partially_addressed,.b-st-conflicting_evidence{background:#fef3c7;color:var(--warn)}
.b-st-limited_evidence_found,.b-st-preliminary_evidence{background:#dcfce7;color:var(--ok)}
.b-st-cannot_validate{background:#e5e7eb;color:var(--grey)}
.b-conf-high,.b-feas-high{background:#dcfce7;color:var(--ok)}
.b-conf-medium,.b-feas-medium{background:#fef3c7;color:var(--warn)}
.b-conf-low,.b-feas-low{background:#fee2e2;color:var(--bad)}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:22px 0 10px;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5}
.tabs button{border:1px solid var(--line);background:var(--card);color:var(--grey);padding:7px 14px;border-radius:999px;
 cursor:pointer;font:inherit;font-size:.92em;font-weight:600}
.tabs button:hover{border-color:var(--accent);color:var(--accent)}
.tabs button.active{background:var(--accent);border-color:var(--accent);color:#fff}
.panel{display:none}.panel.active{display:block}
.toolbar{display:flex;gap:8px;justify-content:flex-end;margin:-4px 0 8px}
.toolbar button{border:1px solid var(--line);background:var(--card);border-radius:8px;padding:4px 10px;font:inherit;font-size:.85em;cursor:pointer;color:var(--grey)}
@media print{.tabs,.toolbar{display:none}.panel{display:block}details{break-inside:avoid}}
"""

_JS = """
(function(){
  const tabs=[...document.querySelectorAll('.tabs button')], panels=[...document.querySelectorAll('.panel')];
  function show(id,remember){tabs.forEach(b=>b.classList.toggle('active',b.dataset.tab===id));
    panels.forEach(p=>p.classList.toggle('active',p.id===id));
    if(remember&&history.replaceState)history.replaceState(null,'','#'+id);}
  tabs.forEach(b=>b.addEventListener('click',()=>show(b.dataset.tab,true)));
  const h=location.hash.slice(1); show(panels.some(p=>p.id===h)?h:panels[0].id,false);
  document.querySelectorAll('[data-expand]').forEach(b=>b.addEventListener('click',()=>{
    const open=b.dataset.expand==='open';
    b.closest('.panel').querySelectorAll('details').forEach(d=>d.open=open);}));
  document.querySelectorAll('a[href^="#gap-"]').forEach(a=>a.addEventListener('click',e=>{
    const d=document.getElementById(a.getAttribute('href').slice(1)); if(!d)return; e.preventDefault();
    show('gaps'); d.open=true; d.scrollIntoView({behavior:'smooth',block:'start'});}));
})();
"""


def _md2html(converter, text: str) -> str:
    converter.reset()
    return converter.convert(text)


def render_html(state: RunState) -> str:
    """Interactive dashboard: summary first, the rest behind tabs; every gap and long trail folded."""
    P = build_parts(state)
    title = f"Research gaps: {P.topic}"
    try:
        import markdown
    except ImportError:  # pragma: no cover - optional dependency
        return f"<!DOCTYPE html><html><body><pre>{html.escape(render_report(state))}</pre></body></html>"
    conv = markdown.Markdown(extensions=["tables", "md_in_html", "sane_lists", "attr_list"])
    H = lambda text: _md2html(conv, text)  # noqa: E731

    def gap_card(g: GapPart) -> str:
        cls = "gap rejected" if g.rejected else f"gap s-{g.status_key}"
        rank = f'<span class="badge b-rank">#{g.rank}</span> ' if g.rank else ""
        summary = (f'{rank}<span class="gtitle">{html.escape(g.id)}: {html.escape(g.title)}</span> '
                   f'{_badge("rej" if g.rejected else "st-" + g.status_key, ("rejected: " if g.rejected else "") + g.status)}'
                   f'<span class="teaser">{html.escape(g.teaser)}</span>')
        inner = re.sub(r'<span class="badge b-rank">[^<]*</span>\s*', "", g.badges)  # rank already in the summary line
        body = H(inner + "\n\n" + g.body_md)
        return f'<details class="{cls}" id="gap-{g.id}"><summary>{summary}</summary><div class="gbody">{body}</div></details>'

    metrics = "".join(f'<div class="metric"><b>{html.escape(v)}</b><span>{html.escape(k)}</span></div>' for k, v in P.metrics)
    ranked_links = re.sub(r"\*\*(G\d+)\*\*", r"**[\1](#gap-\1)**", P.ranked_table_md)
    hero = [f'<header class="hero"><h1>Research Gap Discovery Report</h1><p class="topic">{html.escape(P.topic)}</p>',
            f'<p class="meta">Generated {P.today} · run <code>{html.escape(P.run_id)}</code> · based on titles, abstracts, '
            f'metadata and open-access full texts within the documented search scope; absence of a result is never '
            f'treated as proof that no prior work exists.</p>',
            f'<div class="metrics">{metrics}</div>']
    if P.recommended_md:
        hero.append(f'<div class="callout rec"><p><b>Recommended direction</b></p>{H(P.recommended_md)}</div>')
    if ranked_links:
        hero.append("<h2>Research opportunities, ranked</h2>" + H(ranked_links))
    else:
        hero.append(H(P.ranked_notes_md))
    lim = P.limitations
    if lim:
        top, rest = lim[:3], lim[3:]
        hero.append("<p><b>Key limitations</b></p>" + H(_bullets(top)))
        if rest:
            hero.append(H(_details(f"All limitations ({len(lim)})", _bullets(lim))))
    hero.append("</header>")

    tabs = [("gaps", f"Research gaps ({len(P.gaps)})"), ("summary", "Summary & themes"), ("sota", "State of the art"),
            ("compare", "Comparison"), ("recommend", "Recommendation"), ("reading", "Reading list"),
            ("papers", "Key papers"), ("method", "Methodology"), ("refs", f"References ({P.n_references})")]
    nav = '<nav class="tabs">' + "".join(f'<button data-tab="{i}">{html.escape(t)}</button>' for i, t in tabs) + "</nav>"
    toolbar = ('<div class="toolbar"><button data-expand="open">Expand all</button>'
               '<button data-expand="close">Collapse all</button></div>')

    panels = []
    gaps_html = [f'<div class="panel" id="gaps">{toolbar}<p>{html.escape(P.gaps_intro)}</p>']
    gaps_html += [gap_card(g) for g in P.gaps]
    if P.rejected:
        gaps_html.append("<h3>Candidates rejected during validation</h3>" + H(P.rejected_table_md))
        gaps_html += [gap_card(g) for g in P.rejected]
    if P.ideas_md:
        gaps_html.append(H(_details("Ideas considered and not proposed as gaps", P.ideas_md)))
    gaps_html.append("</div>")
    panels.append("".join(gaps_html))

    summary = ['<div class="panel" id="summary"><div class="card"><h2>Why these gaps rank as they do</h2>' + H(P.ranked_notes_md) + "</div>"]
    if P.themes_md:
        summary.append('<div class="card"><h2>Main research themes</h2>' + H(P.themes_md) + "</div>")
    summary.append('<div class="card"><h2>Scope and counts</h2>' + H(P.scope_counts_md) + "</div>")
    summary.append('<div class="card"><h2>Limitations of the investigation</h2>' + H(_bullets(P.limitations)) + "</div></div>")
    panels.append("".join(summary))

    panels.append(f'<div class="panel" id="sota">{toolbar}<div class="card">{H(P.sota_md)}</div></div>')
    panels.append(f'<div class="panel" id="compare"><div class="card">{H(P.comparison_md)}</div></div>')
    panels.append(f'<div class="panel" id="recommend">{toolbar}<div class="card">{H(P.recommendation_md)}</div></div>')
    panels.append(f'<div class="panel" id="reading"><div class="card">{H(P.reading_md)}</div></div>')
    panels.append(f'<div class="panel" id="papers"><div class="card">{H(P.key_papers_note)}{H(P.key_papers_md)}</div></div>')
    panels.append(f'<div class="panel" id="method">{toolbar}<div class="card">{H(P.methodology_md)}</div></div>')
    panels.append(f'<div class="panel" id="refs"><div class="card">{H(P.references_md)}</div></div>')

    return (f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{_CSS}</style></head>'
            f'<body><div class="wrap">{"".join(hero)}{nav}{"".join(panels)}</div><script>{_JS}</script></body></html>\n')


def write_report_files(state: RunState, run_dir) -> Path:
    """Write report.md, appendix.md and report.html into run_dir; return the path of report.md."""
    run_dir = Path(run_dir)
    (run_dir / "report.md").write_text(render_report(state))
    (run_dir / "appendix.md").write_text(render_appendix(state))
    (run_dir / "report.html").write_text(render_html(state))
    return run_dir / "report.md"


def _criteria(state: RunState) -> list[str]:
    from .models import RANKING_CRITERIA
    return RANKING_CRITERIA
