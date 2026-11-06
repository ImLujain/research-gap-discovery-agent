"""Shared prompt fragments and formatting helpers."""
from __future__ import annotations

from typing import Iterable

from ..models import CandidateGap, EvidenceRow, EvidenceStatement, Paper, Theme

INTEGRITY_RULES = """
Integrity rules (mandatory):
- Refer to papers ONLY by the ids given to you (format P0001). Never invent papers, authors, DOIs, datasets, venues or findings. If you are unsure, say so.
- Distinguish what a paper directly states from your own interpretation across papers.
- A limitation acknowledged by one paper is NOT a field-wide gap unless several independent papers show the same pattern.
- Absence of a matching paper in the material you were given is NOT evidence that no such work exists.
- Prefer defensible, specific claims over impressive-sounding ones. Say what cannot be established.
- Preprints are not peer reviewed; weigh them accordingly and say when a claim rests only on preprints.
""".strip()


def format_paper(p: Paper, max_abstract: int = 1500) -> str:
    bits = [f"[{p.id}] {p.title} ({p.year or 'n.d.'})"]
    if p.authors:
        bits.append("  Authors: " + ", ".join(p.authors[:6]) + (" et al." if len(p.authors) > 6 else ""))
    meta = [f"type={p.publication_type}"]
    if p.venue:
        meta.append(f"venue={p.venue}")
    if p.cited_by_count is not None:
        meta.append(f"cited_by={p.cited_by_count}")
    if p.is_retracted:
        meta.append("RETRACTED")
    if p.doi:
        meta.append(f"doi={p.doi}")
    bits.append("  " + "; ".join(meta))
    if p.abstract:
        a = p.abstract if len(p.abstract) <= max_abstract else p.abstract[:max_abstract] + " […]"
        bits.append("  Abstract: " + a)
    else:
        bits.append("  Abstract: (not available — title and metadata only)")
    return "\n".join(bits)


def format_papers(papers: Iterable[Paper], max_abstract: int = 1500) -> str:
    return "\n\n".join(format_paper(p, max_abstract) for p in papers)


def format_rows(rows: Iterable[EvidenceRow]) -> str:
    out = []
    for r in rows:
        out.append(
            f"[{r.paper_id}] objective: {r.research_objective}\n"
            f"  methodology: {r.methodology}\n  findings: {r.main_findings}\n"
            f"  limitations (author-acknowledged): {'; '.join(r.limitations)}\n"
            f"  datasets/benchmarks: {', '.join(r.datasets_or_benchmarks) or 'none stated'}\n"
            f"  metrics: {', '.join(r.evaluation_metrics) or 'none stated'}\n"
            f"  assumptions: {'; '.join(r.key_assumptions) or 'none stated'}\n  environment: {r.environment}"
        )
    return "\n\n".join(out)


def format_themes(themes: Iterable[Theme]) -> str:
    out = []
    for t in themes:
        out.append(
            f"[{t.id}] {t.name}: {t.description}\n  papers: {', '.join(t.paper_ids)}\n"
            f"  approaches: {'; '.join(t.established_approaches)}\n  weaknesses: {'; '.join(t.weaknesses)}\n"
            f"  unresolved: {'; '.join(t.unresolved_questions)}"
        )
    return "\n\n".join(out)


def format_evidence(statements: Iterable[EvidenceStatement]) -> str:
    return "\n".join(f"[{e.id}] ({e.kind}) {e.statement}  — papers: {', '.join(e.paper_ids)}" for e in statements)


def format_gap(g: CandidateGap) -> str:
    return (
        f"[{g.id}] {g.title} (category: {g.category.value})\n"
        f"  Established: {g.what_is_established}\n  Missing: {g.what_is_missing}\n"
        f"  Supporting evidence: {', '.join(g.supporting_evidence_ids)}; papers: {', '.join(g.supporting_paper_ids)}\n"
        f"  Why it matters: {g.why_it_matters}\n  Potential contribution: {g.potential_contribution}\n"
        f"  Direct vs interpretation: {g.direct_vs_interpretation}"
    )


def topic_block(plan_topic: str, discipline: str | None, period: str, methodology: str | None, priority: str | None,
                goal: str = "research", existing_work: str | None = None) -> str:
    lines = [f"Research topic: {plan_topic}"]
    lines.append(f"Discipline: {discipline or 'not specified'}")
    lines.append(f"Publication period: {period}")
    lines.append(f"Preferred methodology: {methodology or 'not specified'}")
    lines.append(f"Priority: {priority or 'not specified'}")
    lines.append(f"Desired outcome: {'a technical tool or system artefact' if goal == 'tool' else 'a research contribution'}")
    if existing_work and existing_work.strip():
        lines.append("What the user already built / wants to differentiate from:\n" + existing_work.strip()[:6000])
    return "\n".join(lines)


TOOL_GOAL_GUIDANCE = """
The user's goal is a TECHNICAL TOOL or system artefact, not a theoretical, survey or governance contribution.
- Prefer gaps that a buildable artefact would fill: a scanner, analyser, library, benchmark, dataset, test harness,
  protocol extension, runtime guard, CLI or developer workflow. Frame each gap's potential contribution as the artefact:
  what it takes as input, what it produces, and what it does that no existing tool does.
- Treat existing software (code repositories, released tools, products) as prior work on equal footing with papers.
  A gap is not open if a maintained tool already does it; say which tool and what it covers.
- Avoid gaps whose answer is a framework, taxonomy, policy, user study or literature review unless the user asked for one.
- When the user describes work they already built, look for gaps that would make THAT work unique: capabilities nobody
  ships, measurements nobody publishes, or combinations nobody has put in one tool. Say explicitly how each candidate
  differs from the user's own work and from the closest existing tools.
""".strip()


def goal_block(goal: str, existing_work: str | None) -> str:
    """Context appended to agent prompts: what kind of outcome the user wants and what they already have."""
    lines = []
    if goal == "tool":
        lines.append(TOOL_GOAL_GUIDANCE)
    else:
        lines.append("The user's goal is a research contribution (study, method, analysis or theory).")
    if existing_work and existing_work.strip():
        lines.append("What the user already built or wants to differentiate from (their own description; treat as context, "
                     "not as a verified publication):\n" + existing_work.strip()[:6000])
    return "\n\n".join(lines)
