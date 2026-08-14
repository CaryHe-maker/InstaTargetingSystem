"""Optional whole-project runtime measurement for visualization artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns


@dataclass(slots=True)
class TimeCounter:
    """Measure one project run and persist its duration as a JSON artifact."""

    _startedAtNs: int | None = field(default=None, init=False, repr=False)
    _startedAtUtc: datetime | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Start measuring this run."""
        if self._startedAtNs is not None:
            raise RuntimeError("time counter has already started")
        self._startedAtNs = perf_counter_ns()
        self._startedAtUtc = datetime.now(UTC)

    def stop(self, outputPath: str | Path) -> Path:
        """Stop measuring and atomically write the timing artifact."""
        if self._startedAtNs is None or self._startedAtUtc is None:
            raise RuntimeError("time counter has not started")

        finishedAtNs = perf_counter_ns()
        finishedAtUtc = datetime.now(UTC)
        elapsedNs = max(0, finishedAtNs - self._startedAtNs)
        output = Path(outputPath)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "instatarget.time.v1",
            "elapsedNanoseconds": elapsedNs,
            "elapsedMilliseconds": elapsedNs / 1_000_000,
            "elapsedSeconds": elapsedNs / 1_000_000_000,
            "startedAtUtc": self._startedAtUtc.isoformat(),
            "finishedAtUtc": finishedAtUtc.isoformat(),
        }
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        return output


__all__ = ["TimeCounter"]
