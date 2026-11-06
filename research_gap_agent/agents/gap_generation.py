"""Gap Generation Agent: candidate gaps from cross-paper analysis."""
from __future__ import annotations

from ..llm import LLM
from ..models import EvidenceRow, GapGenerationOutput, LandscapeOutput, Paper
from .common import INTEGRITY_RULES, format_evidence, format_rows, format_themes

SYSTEM = f"""You are the Gap Generation Agent. From the research landscape, evidence statements and extraction rows, propose
candidate research gaps. Consider these categories, but treat them as possibilities, not a checklist:
knowledge, methodological, empirical, evaluation, generalizability, integration, contradiction, deployment,
interdisciplinary, temporal.

Rules:
- Propose between 3 and 8 candidates. Do NOT force a gap the evidence does not support. Fewer, well-supported candidates
  beat many speculative ones. Also list ideas you considered and rejected, with the reason.
- Each candidate must cite the evidence statement ids (E###) and paper ids (P####) that suggest it, state what the
  literature has established, what specifically is missing, why it matters, and the potential contribution.
- Say explicitly which parts are directly stated by the papers and which are your interpretation.
- These are CANDIDATES that will be adversarially validated by targeted searches; you are not being asked to prove novelty.
{INTEGRITY_RULES}"""


def generate_gaps(llm: LLM, landscape: LandscapeOutput, rows: list[EvidenceRow], papers: dict[str, Paper],
                  topic: str, priority: str | None, effort: str = "high", context: str = "") -> GapGenerationOutput:
    user = (
        f"Topic: {topic}\nUser priority: {priority or 'not specified'}\n"
        + (f"\n{context}\n" if context else "") + "\n"
        f"Themes:\n\n{format_themes(landscape.themes)}\n\n"
        f"Evidence statements:\n{format_evidence(landscape.evidence_statements)}\n\n"
        f"Conflicting findings:\n" + "\n".join(
            f"- {c.description} (A: {', '.join(c.paper_ids_side_a)}; B: {', '.join(c.paper_ids_side_b)})"
            for c in landscape.conflicting_findings) + "\n\n"
        f"Established knowledge:\n" + "\n".join(f"- {s}" for s in landscape.established_knowledge) + "\n\n"
        f"Extraction rows:\n\n{format_rows(rows)}"
    )
    return llm.structured(task="generate_gaps", system=SYSTEM, user=user, output_model=GapGenerationOutput,
                          effort=effort, max_tokens=32000)
