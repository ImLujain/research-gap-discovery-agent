"""Literature Analysis Agent: per-paper extraction, landscape synthesis, and state-of-the-art prose."""
from __future__ import annotations

from ..llm import LLM
from ..models import EvidenceRow, ExtractionOutput, LandscapeOutput, Paper, StateOfArtOutput
from .common import INTEGRITY_RULES, format_evidence, format_paper, format_papers, format_rows, format_themes

EXTRACT_SYSTEM = f"""You extract structured evidence from academic records. Some papers come with (possibly truncated) full
text; others only with title, abstract and metadata — the record says which. For each paper produce one row: research objective, methodology, main findings, limitations the authors
acknowledge (write 'none stated' if the abstract states none), datasets/benchmarks, evaluation metrics, key assumptions,
environment (theoretical / simulation / laboratory / real-world / mixed / unknown) and relevance to potential gaps.
Do not infer limitations the authors did not state; put inferred weaknesses under relevance_to_potential_gaps and label
them as your inference. If the abstract is missing, say so in each field rather than guessing.
Return exactly one row per paper id provided.
{INTEGRITY_RULES}"""

LANDSCAPE_SYSTEM = f"""You are the Literature Analysis Agent. From the structured evidence rows of a screened corpus you build the
research landscape for a topic:
1. Group papers into 3–8 meaningful research themes (each with approaches, important contributions, strengths, weaknesses,
   unresolved questions and the ids of its papers). Every analysed paper should belong to at least one theme where sensible.
2. Write cross-paper evidence statements with ids E001, E002, …: shared assumptions, reuse of the same datasets without
   generalisability tests, evaluation only in controlled settings, consistently excluded scenarios/populations, incomplete
   metrics, repeated author-acknowledged limitations, conflicting results, trends receiving increasing attention, emerging
   areas with little empirical work. Each statement lists the supporting paper ids and is marked 'direct' (stated by the
   papers) or 'interpretation' (your inference across papers). Only make a statement when at least one paper supports it;
   patterns claimed to be field-wide need several independent papers.
3. List conflicting findings with the papers on each side.
4. Summarise established knowledge, areas of increasing attention, foundational papers and recent key papers.
Do not assume the newest or most-cited papers are the most reliable.
{INTEGRITY_RULES}"""

SOTA_SYSTEM = f"""You write the 'Current State of the Art' section of a research-gap report for an expert reader.
Write one well-structured prose block per theme (established approaches, important contributions with representative
publications, strengths and weaknesses, unresolved questions), plus a short overview. Cite papers inline using their ids in
square brackets, e.g. [P0012]; a claim about the literature must carry at least one id. Do not cite ids you were not given.
Full paragraphs, no bullet lists, no headings inside the prose.
{INTEGRITY_RULES}"""


def extract_evidence(llm: LLM, papers: list[Paper], topic: str, *, batch_size: int = 8,
                     effort: str = "medium", full_texts: dict[str, str] | None = None, progress=None) -> list[EvidenceRow]:
    """Extract one evidence row per paper. Papers with full text are sent one or two per call; the rest in batches."""
    full_texts = full_texts or {}
    rows: list[EvidenceRow] = []
    with_ft = [p for p in papers if p.id in full_texts]
    without = [p for p in papers if p.id not in full_texts]
    batches = [with_ft[i:i + 2] for i in range(0, len(with_ft), 2)] + \
              [without[i:i + batch_size] for i in range(0, len(without), batch_size)]
    for bi, batch in enumerate(batches, 1):
        if progress:
            progress(f"[extract] batch {bi}/{len(batches)}: {', '.join(p.id for p in batch)} → model")
        blocks = []
        for p in batch:
            block = format_paper(p)
            if p.id in full_texts:
                block += f"\n  FULL TEXT (open access, may be truncated):\n{full_texts[p.id]}\n  END OF FULL TEXT [{p.id}]"
            blocks.append(block)
        user = f"Topic under investigation: {topic}\n\nPapers:\n\n" + "\n\n".join(blocks)
        out = llm.structured(task="extract", system=EXTRACT_SYSTEM, user=user, output_model=ExtractionOutput,
                             effort=effort)
        wanted = {p.id for p in batch}
        rows.extend(r for r in out.rows if r.paper_id in wanted)
    return rows


def build_landscape(llm: LLM, rows: list[EvidenceRow], papers: dict[str, Paper], topic: str,
                    effort: str = "high") -> LandscapeOutput:
    meta = "\n".join(
        f"[{p.id}] {p.title} ({p.year or 'n.d.'}; {p.publication_type}; cited_by={p.cited_by_count if p.cited_by_count is not None else 'unknown'})"
        for p in (papers[r.paper_id] for r in rows if r.paper_id in papers)
    )
    user = (f"Topic: {topic}\n\nPaper metadata:\n{meta}\n\nEvidence rows:\n\n{format_rows(rows)}")
    return llm.structured(task="landscape", system=LANDSCAPE_SYSTEM, user=user, output_model=LandscapeOutput,
                          effort=effort, max_tokens=32000)


def write_state_of_art(llm: LLM, landscape: LandscapeOutput, papers: dict[str, Paper], topic: str,
                       effort: str = "medium") -> StateOfArtOutput:
    ids = {pid for t in landscape.themes for pid in t.paper_ids}
    cited = [papers[p] for p in ids if p in papers]
    user = (
        f"Topic: {topic}\n\nThemes:\n\n{format_themes(landscape.themes)}\n\n"
        f"Evidence statements:\n{format_evidence(landscape.evidence_statements)}\n\n"
        f"Papers you may cite:\n\n{format_papers(cited, max_abstract=600)}"
    )
    return llm.structured(task="state_of_art", system=SOTA_SYSTEM, user=user, output_model=StateOfArtOutput,
                          effort=effort, max_tokens=32000)
