"""Enforce that every paper/evidence reference points at something that exists."""
from __future__ import annotations

import re
from typing import Iterable

from .models import RemovedReference, VerificationReport

PAPER_ID_RE = re.compile(r"\bP\d{4}\b")
EVIDENCE_ID_RE = re.compile(r"\bE\d{3}\b")


def filter_ids(ids: Iterable[str], known: set[str], where: str, report: VerificationReport) -> list[str]:
    """Keep ids that exist; log and drop the rest; de-duplicate preserving order."""
    out: list[str] = []
    for pid in ids:
        pid = pid.strip()
        if pid in known:
            if pid not in out:
                out.append(pid)
        else:
            report.removed_unknown_ids.append(RemovedReference(where=where, paper_id=pid))
    return out


def unknown_ids_in_text(text: str, known: set[str]) -> list[str]:
    return sorted({m for m in PAPER_ID_RE.findall(text or "") if m not in known})


def scrub_text(text: str, known: set[str], where: str, report: VerificationReport) -> str:
    """Replace unknown [P####] references in prose with an explicit marker."""
    unknown = unknown_ids_in_text(text, known)
    for pid in unknown:
        report.removed_unknown_ids.append(RemovedReference(where=where, paper_id=pid))
        text = re.sub(rf"\[?\b{pid}\b\]?", "[unverified reference removed]", text)
    return text
