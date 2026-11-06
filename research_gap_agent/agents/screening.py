"""Title/abstract screening for relevance (primary and gap-specific)."""
from __future__ import annotations

from ..llm import LLM
from ..models import Paper, ScreeningItem, ScreeningOutput
from .common import INTEGRITY_RULES, format_papers

SYSTEM = f"""You screen academic records for relevance, like the title/abstract screening stage of a systematic review.
Decide include/exclude for every record given, with a one-sentence reason. Be inclusive at the margins: include records
that address the topic, closely related problems, or methods clearly applicable to it. Exclude off-topic records, non-research
items (editorials, tables of contents, news), and records whose title and abstract give no usable information.
Return exactly one decision per paper id provided.
{INTEGRITY_RULES}"""


def screen_papers(llm: LLM, papers: list[Paper], criterion: str, *, batch_size: int = 25,
                  effort: str = "low", progress=None) -> list[ScreeningItem]:
    decisions: list[ScreeningItem] = []
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        if progress:
            progress(f"[screen] batch {i // batch_size + 1}/{(len(papers) + batch_size - 1) // batch_size} ({len(batch)} records) → model")
        user = f"Screening criterion:\n{criterion}\n\nRecords:\n\n{format_papers(batch, max_abstract=900)}"
        out = llm.structured(task="screen", system=SYSTEM, user=user, output_model=ScreeningOutput, effort=effort)
        wanted = {p.id for p in batch}
        got = {d.paper_id for d in out.decisions}
        decisions.extend(d for d in out.decisions if d.paper_id in wanted)
        for pid in wanted - got:  # a missing decision is treated as excluded, visibly
            decisions.append(ScreeningItem(paper_id=pid, include=False, reason="no screening decision returned"))
    return decisions
