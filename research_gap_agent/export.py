"""Export the cited/analysed corpus as BibTeX, RIS and CSV (importable into Zotero, Mendeley, EndNote)."""
from __future__ import annotations

import csv
import io
import re

from .models import Paper

_BIB_ESCAPE = str.maketrans({"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"})


def _bib_key(p: Paper) -> str:
    first = re.sub(r"[^a-z]", "", (p.authors[0].split()[-1] if p.authors else "anon").lower()) or "anon"
    word = next((w for w in re.findall(r"[a-z]{4,}", p.title.lower()) if w not in ("with", "from", "that", "this")), "paper")
    return f"{first}{p.year or 'nd'}{word}_{p.id.lower()}"


def _bib_type(p: Paper) -> str:
    return {"journal-article": "article", "review": "article", "conference-paper": "inproceedings",
            "book-chapter": "incollection", "preprint": "misc"}.get(p.publication_type, "misc")


def to_bibtex(papers: list[Paper]) -> str:
    entries = []
    for p in papers:
        fields = {"title": p.title, "author": " and ".join(p.authors) if p.authors else None, "year": p.year,
                  "doi": p.doi, "url": p.best_url(), "note": f"Internal id {p.id}; type {p.publication_type}"
                  + ("; PREPRINT" if p.is_preprint else "") + ("; RETRACTED" if p.is_retracted else "")}
        t = _bib_type(p)
        if t == "article" and p.venue:
            fields["journal"] = p.venue
        elif t == "inproceedings" and p.venue:
            fields["booktitle"] = p.venue
        elif p.venue:
            fields["howpublished"] = p.venue
        if p.arxiv_id:
            fields["eprint"] = p.arxiv_id
            fields["archiveprefix"] = "arXiv"
        body = ",\n".join(f"  {k} = {{{str(v).translate(_BIB_ESCAPE) if k not in ('doi', 'url') else v}}}"
                          for k, v in fields.items() if v not in (None, ""))
        entries.append(f"@{t}{{{_bib_key(p)},\n{body}\n}}")
    return "\n\n".join(entries) + "\n"


def to_ris(papers: list[Paper]) -> str:
    out = []
    for p in papers:
        ty = {"journal-article": "JOUR", "review": "JOUR", "conference-paper": "CONF", "book-chapter": "CHAP",
              "preprint": "UNPB"}.get(p.publication_type, "GEN")
        lines = [f"TY  - {ty}", f"TI  - {p.title}"] + [f"AU  - {a}" for a in p.authors]
        if p.year:
            lines.append(f"PY  - {p.year}")
        if p.venue:
            lines.append(f"T2  - {p.venue}")
        if p.doi:
            lines.append(f"DO  - {p.doi}")
        if p.best_url():
            lines.append(f"UR  - {p.best_url()}")
        if p.abstract:
            lines.append(f"AB  - {p.abstract}")
        lines.append(f"N1  - Internal id {p.id}" + ("; RETRACTED" if p.is_retracted else ""))
        lines.append("ER  - ")
        out.append("\n".join(lines))
    return "\n".join(out) + "\n"


def to_csv(papers: list[Paper]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "title", "year", "authors", "venue", "type", "doi", "arxiv_id", "url", "cited_by", "retracted",
                "sources", "screening", "full_text_chars"])
    for p in papers:
        w.writerow([p.id, p.title, p.year or "", "; ".join(p.authors), p.venue or "", p.publication_type, p.doi or "",
                    p.arxiv_id or "", p.best_url() or "", p.cited_by_count if p.cited_by_count is not None else "",
                    "yes" if p.is_retracted else "", ", ".join(p.sources),
                    ("include" if p.screening.include else "exclude") if p.screening else "", p.full_text_chars])
    return buf.getvalue()
