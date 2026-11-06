"""Base class for HTTP-backed academic sources: rate limiting, retries, disk cache."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import httpx

from ..config import Settings
from ..models import RawRecord, SearchStatus

log = logging.getLogger(__name__)


@dataclass
class SearchOutcome:
    records: list[RawRecord] = field(default_factory=list)
    status: SearchStatus = "ok"
    error: Optional[str] = None
    total_available: Optional[int] = None


class Source(Protocol):
    name: str

    def search(self, query: str, *, from_year: Optional[int] = None, to_year: Optional[int] = None,
               limit: int = 25) -> SearchOutcome: ...


class SourceRateLimited(Exception):
    pass


class HttpSource:
    name: str = "base"
    base_url: str = ""
    min_interval: float = 0.0  # seconds between requests to this source
    max_attempts: int = 4
    rate_limit_trip: int = 2  # consecutive rate-limited queries before the rest of the batch is skipped
    max_retry_wait: float = 60.0  # a Retry-After longer than this means "come back tomorrow", not "wait"

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or Settings()
        self._client = client or httpx.Client(
            timeout=self.settings.http_timeout,
            headers={"User-Agent": self.settings.resolved_user_agent(), "Accept": "application/json"},
            follow_redirects=True,
        )
        self._lock = threading.Lock()
        self._last_request = 0.0
        self.cache_dir: Optional[Path] = self.settings.cache_dir / self.name if self.settings.cache_dir else None
        self.on_event: Optional[Callable[[str], None]] = None  # progress sink for retries/skips
        self._consecutive_rate_limits = 0
        self._exhausted_until = 0.0  # monotonic time; set when the server asks us to wait longer than max_retry_wait
        self.exhausted_reason: Optional[str] = None

    def _emit(self, msg: str) -> None:
        if self.on_event:
            try:
                self.on_event(msg)
            except Exception:  # noqa: BLE001 - never let a UI sink break a search
                pass
            log.debug(msg)  # the progress sink already shows it; avoid printing every event twice
        else:
            log.info(msg)

    def reset_breaker(self) -> None:
        self._consecutive_rate_limits = 0

    @property
    def breaker_tripped(self) -> bool:
        return self._consecutive_rate_limits >= self.rate_limit_trip

    def note_outcome(self, outcome: "SearchOutcome") -> None:
        if outcome.status == "rate_limited":
            self._consecutive_rate_limits += 1
            if self.breaker_tripped:
                self._emit(f"{self.name}: rate-limited on {self._consecutive_rate_limits} consecutive queries; "
                           f"skipping the rest of this batch (results from {self.name} will be incomplete)")
        elif outcome.status == "ok":
            self._consecutive_rate_limits = 0

    def skipped_outcome(self) -> "SearchOutcome":
        return SearchOutcome(status="rate_limited", error="skipped: source rate-limited on previous queries in this batch")

    # ------------------------------------------------------------------
    def _cache_path(self, url: str, params: dict) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256((url + "?" + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:32]
        return self.cache_dir / f"{key}.json"

    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def get(self, url: str, params: dict | None = None, *, headers: dict | None = None, as_text: bool = False):
        """GET with cache, throttling and backoff. Raises SourceRateLimited after retries."""
        params = params or {}
        cpath = self._cache_path(url, params)
        if cpath and cpath.exists():
            data = json.loads(cpath.read_text())
            return data["text"] if as_text else data["json"]
        if time.monotonic() < self._exhausted_until:
            raise SourceRateLimited(f"skipped: {self.exhausted_reason}")
        last_err: Exception | None = None
        for attempt in range(self.max_attempts):
            self._throttle()
            try:
                resp = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError as e:
                last_err = e
                time.sleep(min(2 ** attempt, 20))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2.0 * (2 ** attempt), 30)
                if delay > self.max_retry_wait:
                    # e.g. OpenAlex's free daily budget: Retry-After = seconds until midnight UTC. Sleeping for
                    # hours would silently freeze the whole run; skip this source instead and say so.
                    self._exhausted_until = time.monotonic() + delay
                    self.exhausted_reason = (f"{self.name} rate-limited for {delay / 3600:.1f} h "
                                             f"(HTTP {resp.status_code}: {_server_message(resp)})")
                    self._emit(f"{self.name}: server asks us to wait {delay / 3600:.1f} h before retrying — "
                               f"skipping {self.name} for the rest of this run. {_server_message(resp)}")
                    raise SourceRateLimited(self.exhausted_reason)
                if attempt + 1 < self.max_attempts:
                    self._emit(f"{self.name}: HTTP {resp.status_code}, retrying in {delay:.0f}s (attempt {attempt + 1}/{self.max_attempts})")
                else:
                    self._emit(f"{self.name}: HTTP {resp.status_code} after {self.max_attempts} attempts; giving up on this query")
                    delay = 0
                last_err = SourceRateLimited(f"HTTP {resp.status_code}") if resp.status_code == 429 else RuntimeError(
                    f"HTTP {resp.status_code}")
                time.sleep(delay)
                continue
            resp.raise_for_status()
            payload = {"text": resp.text} if as_text else {"json": resp.json()}
            if cpath:
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_text(json.dumps(payload))
            return payload["text"] if as_text else payload["json"]
        assert last_err is not None
        raise last_err

    def _outcome_from_error(self, e: Exception) -> SearchOutcome:
        if isinstance(e, SourceRateLimited):
            return SearchOutcome(status="rate_limited", error=str(e))
        return SearchOutcome(status="error", error=f"{type(e).__name__}: {e}")

    def search(self, query: str, *, from_year: Optional[int] = None, to_year: Optional[int] = None,
               limit: int = 25) -> SearchOutcome:  # pragma: no cover - abstract
        raise NotImplementedError


def _server_message(resp: httpx.Response) -> str:
    """Short human-readable reason from a JSON error body, if any."""
    try:
        body = resp.json()
        msg = body.get("message") or body.get("error") or ""
    except Exception:  # noqa: BLE001
        msg = resp.text or ""
    msg = " ".join(str(msg).split())
    return msg[:200] + ("…" if len(msg) > 200 else "")


def _clean_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    import re

    s = re.sub(r"<[^>]+>", " ", s)  # strip JATS/HTML tags
    s = re.sub(r"\s+", " ", s).strip()
    return s or None
