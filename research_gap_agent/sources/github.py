"""GitHub repository search: existing tools and implementations as prior work.

Records are typed ``software``: a repository is not a publication, but when the user's goal is a technical tool, a
maintained repository that already does the job is exactly the evidence the validation stage must see.
Unauthenticated search allows 10 requests/minute; ``GITHUB_TOKEN`` raises that to 30.
"""
from __future__ import annotations

from typing import Optional

import httpx

from ..models import RawRecord
from .base import HttpSource, SearchOutcome, SourceRateLimited


class GitHubSource(HttpSource):
    name = "github"
    base_url = "https://api.github.com"
    min_interval = 6.5  # 10 requests/minute unauthenticated

    def __init__(self, settings=None, client=None):
        super().__init__(settings, client)
        if self.settings.github_token:
            self.min_interval = 2.1  # 30 requests/minute with a token

    def _headers(self) -> dict:
        h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.settings.github_token:
            h["Authorization"] = f"Bearer {self.settings.github_token}"
        return h

    def search(self, query: str, *, from_year: Optional[int] = None, to_year: Optional[int] = None,
               limit: int = 25) -> SearchOutcome:
        q = query.replace('"', "")  # GitHub's qualifier syntax treats quotes differently; keywords are enough
        if from_year:
            q += f" created:>={from_year}-01-01"
        if to_year:
            q += f" created:<={to_year}-12-31"
        params = {"q": q, "sort": "stars", "order": "desc", "per_page": min(limit, 50)}
        try:
            data = self.get(f"{self.base_url}/search/repositories", params, headers=self._headers())
        except Exception as e:  # noqa: BLE001
            return self._outcome_from_error(e)
        recs = [self.to_record(r) for r in data.get("items", [])]
        return SearchOutcome(records=[r for r in recs if r], total_available=data.get("total_count"))

    def _outcome_from_error(self, e: Exception) -> SearchOutcome:
        # GitHub signals secondary rate limits with 403, not 429
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 403:
            return SearchOutcome(status="rate_limited", error="HTTP 403 (GitHub rate limit)")
        return super()._outcome_from_error(e)

    @staticmethod
    def to_record(r: dict) -> Optional[RawRecord]:
        name = r.get("full_name")
        if not name or not r.get("html_url"):
            return None
        created = (r.get("created_at") or "")[:4]
        pushed = (r.get("pushed_at") or "")[:10]
        desc = (r.get("description") or "").strip()
        topics = ", ".join(r.get("topics") or [])
        bits = [f"[GitHub repository, not a paper] {desc}" if desc else "[GitHub repository, not a paper; no description]"]
        meta = [f"{r.get('stargazers_count', 0)} stars", f"{r.get('forks_count', 0)} forks"]
        if r.get("language"):
            meta.append(f"language: {r['language']}")
        if pushed:
            meta.append(f"last push: {pushed}")
        if r.get("archived"):
            meta.append("ARCHIVED")
        if r.get("license") and r["license"].get("spdx_id"):
            meta.append(f"license: {r['license']['spdx_id']}")
        bits.append("Repository metadata: " + "; ".join(meta) + (f". Topics: {topics}" if topics else ""))
        owner = (r.get("owner") or {}).get("login")
        return RawRecord(
            source="github", source_id=str(r.get("id") or name), title=f"{name} (GitHub repository)",
            abstract=" ".join(bits), year=int(created) if created.isdigit() else None,
            authors=[owner] if owner else [], venue="GitHub", publication_type="software",
            cited_by_count=r.get("stargazers_count"), url=r.get("html_url"),
        )
