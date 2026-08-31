"""In-process knowledge stage timings. No document contents."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType


class StageTimer:
    def __init__(self) -> None:
        self._stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self._stages[name] = (time.perf_counter() - started) * 1000.0

    def snapshot(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._stages))


__all__ = ["StageTimer"]
