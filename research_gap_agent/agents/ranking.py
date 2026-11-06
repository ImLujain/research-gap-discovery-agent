"""Research Ranking Agent: transparent qualitative rubric."""
from __future__ import annotations

from ..llm import LLM
from ..models import RANKING_CRITERIA, RankedGap, RankingOutput, ValidatedGap
from .common import INTEGRITY_RULES, format_gap

SYSTEM = f"""You are the Research Ranking Agent. Rate each validated research gap on every criterion below as high, medium or low
with a one- or two-sentence rationale, name its main risk, and explain the overall reasoning. Then describe the trade-offs
between the candidates.

Criteria (use these exact names): {', '.join(RANKING_CRITERIA)}.

Rules: importance is not novelty — a problem can be important but well studied; a topic can be novel but insignificant.
Weigh the validation status: 'limited_evidence_found' is weaker support than 'preliminary_evidence' with named papers,
because absence of results is not evidence of absence. Ratings are qualitative judgements, not measurements.
When the goal is a technical tool, weigh 'potential_contribution' and 'feasibility' by whether a concrete, buildable
artefact would fill the gap and by how clearly it differs from existing tools and the user's own work.
{INTEGRITY_RULES}"""

RATING_POINTS = {"high": 2, "medium": 1, "low": 0}


def _format_validated(v: ValidatedGap) -> str:
    ce = "; ".join(f"{c.paper_id} ({c.relationship}): {c.note}" for c in v.counter_evidence) or "none"
    return (
        f"{format_gap(v.gap)}\n  Validation status: {v.status.value}\n  Confidence: {v.confidence} — {v.confidence_explanation}\n"
        f"  Counter-evidence: {ce}\n  Feasibility: {v.feasibility.overall} (data: {v.feasibility.data_availability}; "
        f"evaluation: {v.feasibility.evaluation_challenges})"
    )


def rank_gaps(llm: LLM, validated: list[ValidatedGap], topic: str, effort: str = "high",
              context: str = "") -> tuple[list[RankedGap], str]:
    user = (f"Topic: {topic}\n\n" + (f"{context}\n\n" if context else "") + "Validated gaps:\n\n"
            + "\n\n".join(_format_validated(v) for v in validated))
    out = llm.structured(task="rank", system=SYSTEM, user=user, output_model=RankingOutput, effort=effort)
    known = {v.gap.id for v in validated}
    ranked: list[RankedGap] = []
    for item in out.items:
        if item.gap_id not in known:
            continue
        ratings = [r for r in item.ratings if r.criterion in RANKING_CRITERIA]
        score = sum(RATING_POINTS.get(r.rating, 0) for r in ratings)
        ranked.append(RankedGap(gap_id=item.gap_id, rank=0, score=score, ratings=ratings, main_risk=item.main_risk,
                                overall_rationale=item.overall_rationale))
    # deterministic order: score, then evidence-strength rating, then id
    def key(r: RankedGap):
        ev = next((RATING_POINTS[x.rating] for x in r.ratings if x.criterion == "evidence_problem_unresolved"), 0)
        return (-r.score, -ev, r.gap_id)

    ranked.sort(key=key)
    for i, r in enumerate(ranked, 1):
        r.rank = i
    return ranked, out.trade_offs
