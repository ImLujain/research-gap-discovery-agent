"""Open-access full-text retrieval: Unpaywall + arXiv + OA links, PDF text extraction, on-disk cache."""
from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx

from .config import Settings
from .models import Paper

log = logging.getLogger(__name__)
logging.getLogger("pypdf").setLevel(logging.ERROR)  # font/encoding chatter is harmless for text extraction

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_CHARS = 60_000  # ~15k tokens per paper fed to extraction


def extract_pdf_text(data: bytes, max_pages: int = 40) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    chunks = []
    for page in reader.pages[:max_pages]:
        try:
            chunks.append(page.extract_text() or "")
        except Exception as e:  # noqa: BLE001 - a single bad page should not kill the document
            log.debug("page extraction failed: %s", e)
    text = "\n".join(chunks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class FullTextFetcher:
    def __init__(self, settings: Settings, cache_dir: Path, client: Optional[httpx.Client] = None):
        self.settings = settings
        self.cache_dir = cache_dir
        self.client = client or httpx.Client(timeout=60.0, follow_redirects=True,
                                             headers={"User-Agent": settings.resolved_user_agent()})

    # ------------------------------------------------------------------
    def candidate_urls(self, p: Paper) -> list[tuple[str, str]]:
        urls: list[tuple[str, str]] = []
        if p.arxiv_id:
            urls.append(("arxiv", f"https://arxiv.org/pdf/{p.arxiv_id}"))
        if p.oa_url:
            urls.append(("oa_url", p.oa_url))
        if p.doi and self.settings.mailto:
            pdf = self.unpaywall_pdf(p.doi)
            if pdf:
                urls.append(("unpaywall", pdf))
        seen, out = set(), []
        for src, u in urls:
            if u not in seen:
                seen.add(u)
                out.append((src, u))
        return out

    def unpaywall_pdf(self, doi: str) -> Optional[str]:
        try:
            r = self.client.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": self.settings.mailto})
            if r.status_code != 200:
                return None
            data = r.json()
        except Exception as e:  # noqa: BLE001
            log.debug("unpaywall failed for %s: %s", doi, e)
            return None
        best = data.get("best_oa_location") or {}
        if best.get("url_for_pdf"):
            return best["url_for_pdf"]
        for loc in data.get("oa_locations") or []:
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
        return None

    def text_path(self, p: Paper) -> Path:
        return self.cache_dir / f"{p.id}.txt"

    def load(self, p: Paper) -> Optional[str]:
        path = self.text_path(p)
        return path.read_text() if path.exists() else None

    def fetch(self, p: Paper) -> Optional[str]:
        """Fetch and cache the full text of one paper; records outcome on the Paper. Returns text or None."""
        cached = self.load(p)
        if cached:
            p.full_text_chars = len(cached)
            return cached
        errors = []
        for src, url in self.candidate_urls(p):
            try:
                r = self.client.get(url)
                ctype = r.headers.get("content-type", "")
                if r.status_code != 200 or "pdf" not in ctype.lower():
                    errors.append(f"{src}: HTTP {r.status_code} {ctype[:30]}")
                    continue
                if len(r.content) > MAX_PDF_BYTES:
                    errors.append(f"{src}: PDF too large")
                    continue
                text = extract_pdf_text(r.content)
                if len(text) < 1000:
                    errors.append(f"{src}: no extractable text (scanned or image PDF?)")
                    continue
                text = text[:MAX_CHARS]
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.text_path(p).write_text(text)
                p.full_text_chars = len(text)
                p.full_text_source = src
                p.full_text_error = None
                return text
            except Exception as e:  # noqa: BLE001
                errors.append(f"{src}: {type(e).__name__}")
        p.full_text_error = "; ".join(errors) if errors else "no open-access full-text location found"
        return None
