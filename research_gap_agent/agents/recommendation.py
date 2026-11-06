"""Recommended research direction and prioritised reading list."""
from __future__ import annotations

from ..llm import LLM
from ..models import LandscapeOutput, Paper, RankedGap, ReadingListOutput, RecommendationOutput, ValidatedGap
from .common import INTEGRITY_RULES, format_gap, format_papers

REC_SYSTEM = f"""You are writing the 'Recommended Research Direction' section of a research-gap report for the top-ranked gap.
Provide: proposed title; precise problem statement; motivation; evidence that the problem remains unresolved (cite ids);
aim; research questions; hypotheses where appropriate; proposed methodology; suggested datasets/benchmarks/resources
(only ones named in the papers you were given, or clearly standard ones — if none, say what would have to be built);
baselines; evaluation metrics; expected contributions; potential limitations; and a concrete plan to validate novelty before
committing (specific searches, venues, people/communities to check, citation chasing).
When the goal is a technical tool, write the proposal as a build plan: the artefact's name and one-line purpose,
inputs and outputs, core components, what it does that the named existing tools (including the user's own work)
do not, how it would be evaluated, and the minimum viable version.
Never invent experimental results or claim the approach will outperform existing methods.
{INTEGRITY_RULES}"""

READ_SYSTEM = f"""Build a prioritised reading list from the papers provided: foundational papers, important recent studies, directly
related work, papers that challenge the proposed research direction, and surveys/reviews. Explain in one or two sentences why
each matters. 8 to 15 items. Use only ids you were given.
{INTEGRITY_RULES}"""


def recommend(llm: LLM, top: ValidatedGap, ranked: RankedGap, papers: dict[str, Paper], topic: str,
              effort: str = "high", context: str = "") -> RecommendationOutput:
    ids = list(dict.fromkeys(top.gap.supporting_paper_ids + [c.paper_id for c in top.counter_evidence]))
    cited = [papers[i] for i in ids if i in papers]
    user = (
        f"Topic: {topic}\n\n" + (f"{context}\n\n" if context else "")
        + f"Top-ranked gap:\n{format_gap(top.gap)}\n\nValidation status: {top.status.value}\n"
        f"Confidence: {top.confidence} — {top.confidence_explanation}\nMain risk: {ranked.main_risk}\n"
        f"Counter-evidence: " + ("; ".join(f"{c.paper_id} ({c.relationship}): {c.note}" for c in top.counter_evidence) or "none")
        + f"\n\nRelevant papers:\n\n{format_papers(cited, max_abstract=800)}"
    )
    return llm.structured(task="recommend", system=REC_SYSTEM, user=user, output_model=RecommendationOutput,
                          effort=effort)


def reading_list(llm: LLM, landscape: LandscapeOutput, top: ValidatedGap, papers: dict[str, Paper], topic: str,
                 effort: str = "medium") -> ReadingListOutput:
    ids = list(dict.fromkeys(
        landscape.foundational_paper_ids + landscape.recent_key_paper_ids + top.gap.supporting_paper_ids
        + [c.paper_id for c in top.counter_evidence] + [pid for t in landscape.themes for pid in t.paper_ids]
    ))
    cands = [papers[i] for i in ids if i in papers][:80]
    user = (f"Topic: {topic}\n\nProposed direction: {top.gap.title} — {top.gap.what_is_missing}\n\n"
            f"Candidate papers:\n\n{format_papers(cands, max_abstract=400)}")
    return llm.structured(task="reading_list", system=READ_SYSTEM, user=user, output_model=ReadingListOutput,
                          effort=effort)
