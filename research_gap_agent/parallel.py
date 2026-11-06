"""Run independent model calls concurrently while preserving result order."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(fn: Callable[[T], R], items: Iterable[T], workers: int) -> list[R]:
    """Apply ``fn`` to every item, up to ``workers`` at a time; results come back in input order.

    With ``workers <= 1`` (or a single item) this is a plain sequential loop, so behaviour and failure modes are
    identical to the old code. The first exception raised by any item propagates after the others finish.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items)), thread_name_prefix="rga-parallel") as ex:
        return list(ex.map(fn, items))
