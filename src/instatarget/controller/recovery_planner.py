"""Deterministic multi-view planning for normal tracking and recovery."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, pi, sin

from instatarget.core.config import GeometryConfig, RecoveryConfig, TrackingConfig
from instatarget.core.errors import ProtocolError
from instatarget.core.types import BBoxXYWH, BFoV, MotionState3D, TrackStatus, ViewSpec
from instatarget.geometry.projection_math import makeSphericalPoint


@dataclass(frozen=True, slots=True)
class PlannedView:
    spec: ViewSpec
    role: str


class RecoveryPlanner:
    """Build guard, adaptive, ring and global views under a hard per-frame budget."""

    def __init__(
        self,
        geometryConfig: GeometryConfig,
        trackingConfig: TrackingConfig,
        recoveryConfig: RecoveryConfig,
    ) -> None:
        self._geometry = geometryConfig
        self._tracking = trackingConfig
        self._recovery = recoveryConfig

    def buildViews(
        self,
        frameIndex: int,
        frameWidthPx: int,
        frameHeightPx: int,
        anchorBox: BBoxXYWH,
        currentBox: BBoxXYWH,
        fallbackBfov: BFoV,
        predictedMotion: MotionState3D | None,
        status: TrackStatus,
    ) -> tuple[PlannedView, ...]:
        if frameWidthPx <= 0 or frameHeightPx <= 0:
            raise ProtocolError("frame dimensions must be positive")
        center = (
            _motionCenter(predictedMotion)
            if predictedMotion is not None
            else fallbackBfov.center
        )
        horizontalFov, verticalFov = self._contextFov(
            frameWidthPx, frameHeightPx, anchorBox, currentBox
        )
        planned: list[PlannedView] = []
        keys: set[tuple[int, int, int, int]] = set()

        def add(yawRad: float, pitchRad: float, hFov: float, vFov: float, role: str) -> None:
            if len(planned) >= self._recovery.maxViewsPerFrame:
                return
            spec = ViewSpec(
                viewId=len(planned),
                bfov=BFoV(
                    center=makeSphericalPoint(yawRad, pitchRad),
                    horizontalFovRad=hFov,
                    verticalFovRad=vFov,
                ),
                outputWidthPx=self._geometry.viewWidthPx,
                outputHeightPx=self._geometry.viewHeightPx,
            )
            key = (
                round(spec.bfov.center.yawRad * 10000),
                round(spec.bfov.center.pitchRad * 10000),
                round(hFov * 10000),
                round(vFov * 10000),
            )
            if key in keys:
                return
            keys.add(key)
            planned.append(PlannedView(spec=spec, role=role))

        # The guard triplet is mandatory in every state.
        step = self._tracking.guardYawStepRad
        for offset in (-step, 0.0, step):
            add(center.yawRad + offset, center.pitchRad, horizontalFov, verticalFov, "guard")

        add(center.yawRad, center.pitchRad, horizontalFov, verticalFov, "main")

        if status is TrackStatus.TRACKING:
            add(
                center.yawRad,
                center.pitchRad,
                _clampFov(horizontalFov * 0.75, self._geometry),
                _clampFov(verticalFov * 0.75, self._geometry),
                "scale_narrow",
            )
            add(
                center.yawRad,
                center.pitchRad,
                _clampFov(horizontalFov * 1.25, self._geometry),
                _clampFov(verticalFov * 1.25, self._geometry),
                "scale_wide",
            )
        elif status is TrackStatus.UNCERTAIN:
            for offset in (-0.5 * step, 0.5 * step):
                add(
                    center.yawRad + offset,
                    center.pitchRad,
                    _clampFov(horizontalFov * 1.25, self._geometry),
                    _clampFov(verticalFov * 1.25, self._geometry),
                    "hypothesis",
                )
        elif status is TrackStatus.RECOVERING:
            self._addRings(add, center.yawRad, center.pitchRad, horizontalFov, verticalFov)
        elif status is TrackStatus.LOST and frameIndex % self._recovery.globalSearchInterval == 0:
            self._addGlobal(add)

        if len(planned) < 3:
            raise ProtocolError("recovery planner failed to produce the mandatory guard triplet")
        return tuple(planned)

    def contextBfov(
        self,
        center,
        frameWidthPx: int,
        frameHeightPx: int,
        anchorBox: BBoxXYWH,
        currentBox: BBoxXYWH,
    ) -> BFoV:
        horizontalFov, verticalFov = self._contextFov(
            frameWidthPx, frameHeightPx, anchorBox, currentBox
        )
        return BFoV(
            center=center,
            horizontalFovRad=horizontalFov,
            verticalFovRad=verticalFov,
        )

    def _contextFov(
        self,
        frameWidthPx: int,
        frameHeightPx: int,
        anchorBox: BBoxXYWH,
        currentBox: BBoxXYWH,
    ) -> tuple[float, float]:
        widthPx = self._tracking.contextScale * max(anchorBox.widthPx, currentBox.widthPx)
        heightPx = self._tracking.contextScale * max(anchorBox.heightPx, currentBox.heightPx)
        widthPx *= 1.0 + self._tracking.contextMarginRatio
        heightPx *= 1.0 + self._tracking.contextMarginRatio
        horizontal = 2.0 * pi * widthPx / frameWidthPx
        vertical = pi * heightPx / frameHeightPx
        return (
            _clampFov(horizontal, self._geometry),
            _clampFov(vertical, self._geometry),
        )

    def _addRings(
        self,
        add,
        yawRad: float,
        pitchRad: float,
        horizontalFov: float,
        verticalFov: float,
    ) -> None:
        baseRadius = max(horizontalFov, verticalFov)
        for radius, count in zip(
            self._recovery.ringRadii,
            self._recovery.viewsPerRing,
            strict=True,
        ):
            for index in range(count):
                angle = 2.0 * pi * index / count
                offset = radius * baseRadius
                add(
                    yawRad + cos(angle) * offset,
                    pitchRad + sin(angle) * offset,
                    _clampFov(horizontalFov * 1.25, self._geometry),
                    _clampFov(verticalFov * 1.25, self._geometry),
                    "recovery_ring",
                )

    def _addGlobal(self, add) -> None:
        count = max(4, self._recovery.viewsPerRing[-1])
        globalHorizontal = self._geometry.maxFovRad
        globalVertical = self._geometry.maxFovRad
        for index in range(count):
            add(
                -pi + 2.0 * pi * index / count,
                0.0,
                globalHorizontal,
                globalVertical,
                "global",
            )


def _motionCenter(motion: MotionState3D):
    x, y, z = motion.position
    return makeSphericalPoint(atan2(x, z), asin(max(-1.0, min(1.0, y))))


def _clampFov(value: float, geometry: GeometryConfig) -> float:
    return min(geometry.maxFovRad, max(geometry.minFovRad, value))


__all__ = ["PlannedView", "RecoveryPlanner"]
