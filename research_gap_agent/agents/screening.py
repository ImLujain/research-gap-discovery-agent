"""Title/abstract screening for relevance (primary and gap-specific)."""
from __future__ import annotations

from ..llm import LLM
from ..models import Paper, ScreeningItem, ScreeningOutput
from ..parallel import run_parallel
from .common import INTEGRITY_RULES, format_papers

SYSTEM = f"""You screen academic records for relevance, like the title/abstract screening stage of a systematic review.
Decide include/exclude for every record given, with a one-sentence reason. Be inclusive at the margins: include records
that address the topic, closely related problems, or methods clearly applicable to it. Exclude off-topic records, non-research
items (editorials, tables of contents, news), and records whose title and abstract give no usable information.
Return exactly one decision per paper id provided.
{INTEGRITY_RULES}"""


def screen_papers(llm: LLM, papers: list[Paper], criterion: str, *, batch_size: int = 25,
                  effort: str = "low", progress=None, model: str | None = None, concurrency: int = 1) -> list[ScreeningItem]:
    """Screen papers in batches. Batches are independent, so up to ``concurrency`` run at once; ``model`` picks a
    cheaper model for this task (None = the main model). Decisions come back in paper order either way."""
    batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]

    def one(item: tuple[int, list[Paper]]) -> list[ScreeningItem]:
        bi, batch = item
        if progress:
            progress(f"[screen] batch {bi}/{len(batches)} ({len(batch)} records) → model")
        user = f"Screening criterion:\n{criterion}\n\nRecords:\n\n{format_papers(batch, max_abstract=900)}"
        out = llm.structured(task="screen", system=SYSTEM, user=user, output_model=ScreeningOutput, effort=effort,
                             model=model)
        wanted = {p.id for p in batch}
        got = {d.paper_id for d in out.decisions}
        decisions = [d for d in out.decisions if d.paper_id in wanted]
        for pid in wanted - got:  # a missing decision is treated as excluded, visibly
            decisions.append(ScreeningItem(paper_id=pid, include=False, reason="no screening decision returned"))
        return decisions

    results = run_parallel(one, list(enumerate(batches, 1)), concurrency)
    return [d for batch in results for d in batch]
