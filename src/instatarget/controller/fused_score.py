"""Monotonic beta calibration for backend fused scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import exp, isfinite, log, log1p

from instatarget.core.errors import ProtocolError
from instatarget.core.types import LocalObservation

# Parameters solve the beta-calibration model for these anchors:
# 0.80 -> 0.15, 0.90 -> 0.45, 0.95 -> 0.70.
FUSED_SCORE_BETA_PARAMETERS = (7.62702021, 0.91697230, -1.50849067)


def remapFusedScore(score: float) -> float:
    """Calibrate a backend probability with a monotonic beta map."""
    value = float(score)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ProtocolError(f"backend fusedScore must be in [0, 1], actual={score}")
    if value == 0.0 or value == 1.0:
        return value

    alpha, beta, intercept = FUSED_SCORE_BETA_PARAMETERS
    calibratedLogit = intercept + alpha * log(value) - beta * log1p(-value)
    if calibratedLogit >= 0.0:
        return 1.0 / (1.0 + exp(-calibratedLogit))
    exponential = exp(calibratedLogit)
    return exponential / (1.0 + exponential)


def remapLocalObservationFusedScores(
    observations: Sequence[LocalObservation],
) -> tuple[LocalObservation, ...]:
    """Return immutable observations carrying beta-calibrated fused scores."""
    return tuple(
        replace(observation, fusedScore=remapFusedScore(observation.fusedScore))
        for observation in observations
    )


__all__ = [
    "FUSED_SCORE_BETA_PARAMETERS",
    "remapFusedScore",
    "remapLocalObservationFusedScores",
]
