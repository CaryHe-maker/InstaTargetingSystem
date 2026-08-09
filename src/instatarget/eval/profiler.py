"""Runtime profiler."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter_ns

import numpy as np


@dataclass(slots=True)
class TimingStat:
    count: int = 0
    totalNs: int = 0
    minNs: int | None = None
    maxNs: int | None = None

    @property
    def meanNs(self) -> float:
        return float(self.totalNs / self.count) if self.count else 0.0


@dataclass(slots=True)
class RuntimeProfiler:
    stats: dict[str, TimingStat] = field(default_factory=dict)

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = perf_counter_ns()
        try:
            yield
        finally:
            self.record(name, perf_counter_ns() - start)

    def record(self, name: str, elapsedNs: int) -> None:
        stat = self.stats.setdefault(name, TimingStat())
        stat.count += 1
        stat.totalNs += int(elapsedNs)
        stat.minNs = int(elapsedNs) if stat.minNs is None else min(stat.minNs, int(elapsedNs))
        stat.maxNs = int(elapsedNs) if stat.maxNs is None else max(stat.maxNs, int(elapsedNs))

    def summarize(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "count": float(stat.count),
                "meanNs": stat.meanNs,
                "minNs": float(stat.minNs or 0),
                "maxNs": float(stat.maxNs or 0),
            }
            for name, stat in self.stats.items()
        }


def measure(func, *args, **kwargs):
    start = perf_counter_ns()
    value = func(*args, **kwargs)
    return value, perf_counter_ns() - start


__all__ = ["RuntimeProfiler", "TimingStat", "measure"]
