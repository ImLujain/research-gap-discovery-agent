"""Gap Validation Agent: counter-queries and adversarial judgement."""
from __future__ import annotations

from ..llm import LLM
from ..models import CandidateGap, CounterQueryOutput, Paper, ValidationJudgementOutput
from .common import INTEGRITY_RULES, format_gap, format_papers

QUERY_SYSTEM = f"""You are the Gap Validation Agent. Your job is to DISPROVE a candidate research gap by finding existing work that
already addresses it. Write 3 to 6 keyword search queries for academic APIs (2–6 terms each, quotes for phrases) that use
alternative terminology, related concepts, synonyms/abbreviations, different methodological framings, and recent-work
angles. Do not reuse the exact wording of the gap; think about how researchers who solved this problem would have named
it, possibly in a different community. When the goal is a technical tool, also write queries that would find existing
tools and code repositories (tool, scanner, CLI, library, framework, open-source).
{INTEGRITY_RULES}"""

JUDGE_SYSTEM = f"""You are the Gap Validation Agent. You previously proposed counter-queries to find work that addresses a candidate
research gap. You now receive the papers those searches found (plus papers citing the gap's supporting papers, if available).
Judge the gap honestly and adversarially.
Records of type 'software' are code repositories or released tools; they count as existing work that can address
a gap, especially when the goal is a technical tool.

Classify into exactly one status:
- addressed: existing work already answers the gap substantially; the gap should be rejected.
- partially_addressed: important parts are solved; a narrower or reframed gap may remain (provide revised_gap).
- conflicting_evidence: studies disagree on whether/how the problem is solved.
- preliminary_evidence: some work exists but it is preliminary (small scale, preprints only, no empirical validation).
- limited_evidence_found: the targeted searches found little directly relevant work. This is NOT proof of novelty; say so.
- cannot_validate: the searches failed or returned nothing usable, so no judgement is possible.

Also list each relevant paper as counter-evidence with its relationship (addresses_fully / addresses_partially / conflicts /
preliminary / tangential) and a note, give a confidence level with an explicit explanation of the uncertainty, assess
feasibility, and state what could not be established. If a revised (narrowed) gap is warranted, provide it with the same
structure as the original, citing only ids you were given.
{INTEGRITY_RULES}"""


def propose_counter_queries(llm: LLM, gap: CandidateGap, topic: str, effort: str = "medium",
                            context: str = "") -> CounterQueryOutput:
    user = f"Topic: {topic}\n\nCandidate gap:\n{format_gap(gap)}" + (f"\n\n{context}" if context else "")
    return llm.structured(task=f"counter_queries:{gap.id}", system=QUERY_SYSTEM, user=user,
                          output_model=CounterQueryOutput, effort=effort)


def judge_gap(llm: LLM, gap: CandidateGap, topic: str, found: list[Paper], supporting: list[Paper],
              search_summary: str, effort: str = "high", context: str = "") -> ValidationJudgementOutput:
    user = (
        f"Topic: {topic}\n\n" + (f"{context}\n\n" if context else "") + f"Candidate gap:\n{format_gap(gap)}\n\n"
        f"Search execution summary:\n{search_summary}\n\n"
        f"Papers found by the counter-searches (screened as relevant):\n\n"
        f"{format_papers(found) if found else '(none)'}\n\n"
        f"Papers that originally suggested the gap (for reference):\n\n{format_papers(supporting, max_abstract=500)}"
    )
    return llm.structured(task=f"judge:{gap.id}", system=JUDGE_SYSTEM, user=user,
                          output_model=ValidationJudgementOutput, effort=effort)
