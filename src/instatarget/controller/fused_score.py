"""Monotonic fused-score contrast stretching before state evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from instatarget.core.types import LocalObservation

FUSED_SCORE_REMAP_POINTS = (
    (0.00, 0.00),
    (0.60, 0.10),
    (0.80, 0.40),
    (0.90, 0.70),
    (0.95, 0.90),
    (1.00, 1.00),
)


def remapFusedScore(score: float) -> float:
    """Stretch a probability using a continuous monotonic piecewise-linear curve."""
    value = float(score)
    for (inputLow, outputLow), (inputHigh, outputHigh) in zip(
        FUSED_SCORE_REMAP_POINTS,
        FUSED_SCORE_REMAP_POINTS[1:],
        strict=True,
    ):
        if value <= inputHigh:
            ratio = (value - inputLow) / (inputHigh - inputLow)
            return outputLow + ratio * (outputHigh - outputLow)
    return 1.0


def remapLocalObservationFusedScores(
    observations: Sequence[LocalObservation],
) -> tuple[LocalObservation, ...]:
    """Return immutable observation copies carrying contrast-stretched fused scores."""
    return tuple(
        replace(observation, fusedScore=remapFusedScore(observation.fusedScore))
        for observation in observations
    )


__all__ = [
    "FUSED_SCORE_REMAP_POINTS",
    "remapFusedScore",
    "remapLocalObservationFusedScores",
]
