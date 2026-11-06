"""Citation Verification Agent: DOI resolution, retraction flags, claim-support checks."""
from __future__ import annotations

import logging
from typing import Optional

from ..dedupe import normalize_title, titles_similar
from ..llm import LLM
from ..models import ClaimCheckItem, ClaimCheckOutput, DoiCheck, EvidenceStatement, Paper, VerificationReport
from ..sources.crossref import CrossrefSource
from ..sources.openalex import OpenAlexSource
from .common import INTEGRITY_RULES, format_papers

log = logging.getLogger(__name__)

CLAIM_SYSTEM = f"""You check whether cited papers actually support evidence statements. For each statement you get the statement
text and the title/abstract of every paper it cites. Verdicts: supported (the cited material clearly supports it),
partially_supported (supports part of it, or only some cited papers support it), not_supported (the cited material does not
say this), cannot_assess (abstracts missing or too thin). Be strict: an abstract that is merely on-topic does not support a
specific claim. Give a one-sentence note naming which paper ids support or fail to support the statement.
{INTEGRITY_RULES}"""


def verify_dois(papers: list[Paper], crossref: Optional[CrossrefSource], openalex: Optional[OpenAlexSource],
                report: VerificationReport) -> None:
    """Resolve each cited DOI through Crossref and compare titles; pull retraction flags from OpenAlex if missing."""
    for p in papers:
        if p.is_retracted:
            if p.id not in report.retracted_paper_ids:
                report.retracted_paper_ids.append(p.id)
        if not p.doi or p.doi_check is not None:
            continue
        if crossref is None:
            continue
        msg = crossref.lookup_doi(p.doi)
        if msg is None:
            chk = DoiCheck(doi=p.doi, resolved=False, note="DOI did not resolve at Crossref")
        else:
            titles = msg.get("title") or []
            ct = titles[0] if titles else None
            match = titles_similar(normalize_title(ct), normalize_title(p.title), threshold=0.85) if ct else None
            chk = DoiCheck(doi=p.doi, resolved=True, title_match=match, crossref_title=ct,
                           note=None if match else "Crossref title differs from retrieved title; check manually")
            for upd in msg.get("update-to") or []:
                if (upd.get("type") or "").lower() == "retraction":
                    chk.note = "Crossref lists this DOI as a retraction notice"
            if "is-retracted-by" in (msg.get("relation") or {}):
                p.is_retracted = True
                if p.id not in report.retracted_paper_ids:
                    report.retracted_paper_ids.append(p.id)
        p.doi_check = chk
        report.doi_checks.append(chk)
        if openalex is not None and p.is_retracted is None:
            rec = openalex.lookup_doi(p.doi)
            if rec is not None:
                p.is_retracted = rec.is_retracted
                if rec.is_retracted and p.id not in report.retracted_paper_ids:
                    report.retracted_paper_ids.append(p.id)


def check_claims(llm: LLM, statements: list[EvidenceStatement], papers: dict[str, Paper],
                 effort: str = "medium", batch_size: int = 10) -> list[ClaimCheckItem]:
    items: list[ClaimCheckItem] = []
    for i in range(0, len(statements), batch_size):
        batch = statements[i:i + batch_size]
        blocks = []
        for s in batch:
            cited = [papers[p] for p in s.paper_ids if p in papers]
            blocks.append(f"Statement [{s.id}] ({s.kind}): {s.statement}\nCited papers:\n{format_papers(cited, max_abstract=900)}")
        out = llm.structured(task="claim_check", system=CLAIM_SYSTEM, user="\n\n---\n\n".join(blocks),
                             output_model=ClaimCheckOutput, effort=effort)
        wanted = {s.id for s in batch}
        items.extend(it for it in out.items if it.evidence_id in wanted)
    return items
