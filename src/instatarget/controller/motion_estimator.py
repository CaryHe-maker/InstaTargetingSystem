"""Lightweight spherical motion estimation for the DTC control thread."""

from __future__ import annotations

from math import isfinite

import numpy as np

from instatarget.core.errors import ProtocolError
from instatarget.core.protocols import MotionEstimator as MotionEstimatorProtocol
from instatarget.core.types import DepthSummary, MotionState3D, SphericalPoint


class SphericalMotionEstimator(MotionEstimatorProtocol):
    """Alpha-Beta estimator over unit-vector direction and optional range."""

    def __init__(self, alpha: float = 0.70, beta: float = 0.20) -> None:
        if not 0.0 < alpha <= 1.0 or not 0.0 < beta <= 1.0:
            raise ValueError("alpha and beta must be in (0, 1]")
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._initialized = False
        self._position = np.zeros(3, dtype=np.float64)
        self._velocity = np.zeros(3, dtype=np.float64)
        self._rangeDepth = 0.0
        self._rangeVelocity = 0.0
        self._confidence = 0.0
        self._timestampNs = 0

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(
        self,
        point: SphericalPoint,
        depth: DepthSummary | None,
        timestampNs: int,
    ) -> MotionState3D:
        _requireTimestamp(timestampNs)
        self._position = _unitVector(point)
        self._velocity = np.zeros(3, dtype=np.float64)
        self._rangeDepth = _depthMeasurement(depth) or 0.0
        self._rangeVelocity = 0.0
        self._confidence = _clamp01(depth.confidence if depth is not None else 1.0)
        self._timestampNs = int(timestampNs)
        self._initialized = True
        return self._state()

    def predict(self, timestampNs: int) -> MotionState3D:
        _requireTimestamp(timestampNs)
        self._requireInitialized()
        if timestampNs < self._timestampNs:
            raise ProtocolError(
                f"motion timestamp must be monotonic: {timestampNs} < {self._timestampNs}"
            )
        return self._predictedState(int(timestampNs))

    def update(
        self,
        point: SphericalPoint,
        depth: DepthSummary | None,
        timestampNs: int,
        observationConfidence: float,
    ) -> MotionState3D:
        _requireTimestamp(timestampNs)
        self._requireInitialized()
        if timestampNs < self._timestampNs:
            raise ProtocolError(
                f"motion timestamp must be monotonic: {timestampNs} < {self._timestampNs}"
            )
        if not 0.0 <= observationConfidence <= 1.0 or not isfinite(observationConfidence):
            raise ProtocolError("observationConfidence must be in [0, 1]")

        predicted = self._predictedState(int(timestampNs))
        dt = max((int(timestampNs) - self._timestampNs) / 1e9, 1e-3)
        measuredPosition = _unitVector(point)
        residual = measuredPosition - np.asarray(predicted.position, dtype=np.float64)
        correctedPosition = _normalize(
            np.asarray(predicted.position, dtype=np.float64) + self._alpha * residual
        )
        correctedVelocity = (
            np.asarray(predicted.velocity, dtype=np.float64) + self._beta * residual / dt
        )

        measuredRange = _depthMeasurement(depth)
        if measuredRange is None:
            correctedRange = predicted.rangeDepth
            correctedRangeVelocity = predicted.rangeVelocity
        else:
            rangeResidual = measuredRange - predicted.rangeDepth
            correctedRange = max(0.0, predicted.rangeDepth + self._alpha * rangeResidual)
            correctedRangeVelocity = predicted.rangeVelocity + self._beta * rangeResidual / dt

        self._position = correctedPosition
        self._velocity = correctedVelocity
        self._rangeDepth = correctedRange
        self._rangeVelocity = correctedRangeVelocity
        self._confidence = _clamp01(0.5 * predicted.confidence + 0.5 * observationConfidence)
        self._timestampNs = int(timestampNs)
        return self._state()

    def _predictedState(self, timestampNs: int) -> MotionState3D:
        dt = max((timestampNs - self._timestampNs) / 1e9, 0.0)
        position = _normalize(self._position + self._velocity * dt)
        return MotionState3D(
            position=tuple(float(value) for value in position),
            velocity=tuple(float(value) for value in self._velocity),
            rangeDepth=max(0.0, self._rangeDepth + self._rangeVelocity * dt),
            rangeVelocity=float(self._rangeVelocity),
            confidence=_clamp01(self._confidence),
        )

    def _state(self) -> MotionState3D:
        return MotionState3D(
            position=tuple(float(value) for value in self._position),
            velocity=tuple(float(value) for value in self._velocity),
            rangeDepth=max(0.0, float(self._rangeDepth)),
            rangeVelocity=float(self._rangeVelocity),
            confidence=_clamp01(self._confidence),
        )

    def _requireInitialized(self) -> None:
        if not self._initialized:
            raise ProtocolError("motion estimator has not been initialized")


def _unitVector(point: SphericalPoint) -> np.ndarray:
    return _normalize(np.asarray((point.x, point.y, point.z), dtype=np.float64))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ProtocolError("motion vector must be finite and non-zero")
    return vector / norm


def _depthMeasurement(depth: DepthSummary | None) -> float | None:
    if depth is None or depth.validRatio <= 0.0 or depth.confidence <= 0.0:
        return None
    return float(depth.medianDepth)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _requireTimestamp(timestampNs: int) -> None:
    if isinstance(timestampNs, bool) or not isinstance(timestampNs, int) or timestampNs < 0:
        raise ProtocolError("motion timestampNs must be a non-negative integer")


MotionEstimatorImpl = SphericalMotionEstimator

__all__ = ["MotionEstimatorImpl", "SphericalMotionEstimator"]
