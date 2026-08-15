"""Fixed 120-degree four-corner and cubemap search planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, pi, sin, sqrt, tan

from instatarget.controller.state_model import RecoveryMemory
from instatarget.core.config import GeometryConfig, RecoveryConfig, TrackingConfig
from instatarget.core.errors import ProtocolError
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    MotionState3D,
    SphericalPoint,
    TrackStatus,
    ViewSpec,
)
from instatarget.geometry.projection_math import makeSphericalPoint, unitVectorToYawPitch


@dataclass(frozen=True, slots=True)
class PlannedView:
    spec: ViewSpec
    role: str


class RecoveryPlanner:
    """Build the state-specific bounded view sequence from the design specification."""

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
        *,
        searchSeedCenter: SphericalPoint | None = None,
        attemptIndex: int = 0,
        viewIdStart: int = 0,
        viewBudget: int | None = None,
        recoveryMemory: RecoveryMemory | None = None,
    ) -> tuple[PlannedView, ...]:
        del frameIndex, anchorBox, currentBox, recoveryMemory
        if frameWidthPx <= 0 or frameHeightPx <= 0:
            raise ProtocolError("frame dimensions must be positive")
        if attemptIndex < 0 or attemptIndex >= self._tracking.maxAttemptsPerFrame:
            raise ProtocolError(f"unsupported attemptIndex: {attemptIndex}")

        center = searchSeedCenter or (
            _motionCenter(predictedMotion) if predictedMotion is not None else fallbackBfov.center
        )
        useCubeMap = (
            status is TrackStatus.LOST and attemptIndex == 0
        ) or (
            status in {TrackStatus.UNCERTAIN, TrackStatus.RECOVERING} and attemptIndex == 2
        )
        if status in {TrackStatus.TRACKING, TrackStatus.LOST} and attemptIndex >= 2:
            raise ProtocolError(f"{status.name} does not support a third search round")

        requiredViews = 6 if useCubeMap else 4
        budget = viewBudget if viewBudget is not None else self._tracking.maxViewsPerFrameTotal
        if budget < requiredViews:
            raise ProtocolError(
                f"view budget cannot fit attempt: required={requiredViews}, available={budget}"
            )
        if useCubeMap:
            return self._cubeMap(viewIdStart, attemptIndex)
        return self._fourCorners(center, viewIdStart, attemptIndex)

    def contextBfov(
        self,
        center: SphericalPoint,
        frameWidthPx: int,
        frameHeightPx: int,
        anchorBox: BBoxXYWH,
        currentBox: BBoxXYWH,
        uncertaintyRad: float = 0.0,
    ) -> BFoV:
        """Build the motion fallback envelope; search ViewSpecs do not use this size."""
        widthPx = self._tracking.contextScale * max(anchorBox.widthPx, currentBox.widthPx)
        heightPx = self._tracking.contextScale * max(anchorBox.heightPx, currentBox.heightPx)
        widthPx *= 1.0 + self._tracking.contextMarginRatio
        heightPx *= 1.0 + self._tracking.contextMarginRatio
        horizontalFov = 2.0 * pi * widthPx / frameWidthPx + 2.0 * uncertaintyRad
        verticalFov = pi * heightPx / frameHeightPx + 2.0 * uncertaintyRad
        return BFoV(
            center=center,
            horizontalFovRad=_clampFov(horizontalFov, self._geometry),
            verticalFovRad=_clampFov(verticalFov, self._geometry),
        )

    def _fourCorners(
        self,
        center: SphericalPoint,
        viewIdStart: int,
        attemptIndex: int,
    ) -> tuple[PlannedView, ...]:
        offset = 40.0 * pi / 180.0
        roles = (
            ("left_top", -offset, offset),
            ("right_top", offset, offset),
            ("left_bottom", -offset, -offset),
            ("right_bottom", offset, -offset),
        )
        return tuple(
            PlannedView(
                spec=self._viewSpec(
                    viewIdStart + index,
                    _offsetDirection(center, yawOffset, pitchOffset),
                ),
                role=f"round{attemptIndex + 1}_{role}",
            )
            for index, (role, yawOffset, pitchOffset) in enumerate(roles)
        )

    def _cubeMap(self, viewIdStart: int, attemptIndex: int) -> tuple[PlannedView, ...]:
        directions = (
            ("front", 0.0, 0.0),
            ("right", pi / 2.0, 0.0),
            ("back", -pi, 0.0),
            ("left", -pi / 2.0, 0.0),
            ("up", 0.0, pi / 2.0),
            ("down", 0.0, -pi / 2.0),
        )
        return tuple(
            PlannedView(
                spec=self._viewSpec(viewIdStart + index, makeSphericalPoint(yaw, pitch)),
                role=f"round{attemptIndex + 1}_cubemap_{role}",
            )
            for index, (role, yaw, pitch) in enumerate(directions)
        )

    def _viewSpec(self, viewId: int, center: SphericalPoint) -> ViewSpec:
        return ViewSpec(
            viewId=viewId,
            bfov=BFoV(
                center=center,
                horizontalFovRad=self._geometry.maxFovRad,
                verticalFovRad=self._geometry.maxFovRad,
            ),
            outputWidthPx=self._geometry.viewWidthPx,
            outputHeightPx=self._geometry.viewHeightPx,
        )


def _offsetDirection(
    center: SphericalPoint,
    localYawOffsetRad: float,
    localPitchOffsetRad: float,
) -> SphericalPoint:
    yaw = center.yawRad
    pitch = center.pitchRad
    forward = (center.x, center.y, center.z)
    right = (cos(yaw), 0.0, -sin(yaw))
    up = (-sin(pitch) * sin(yaw), cos(pitch), -sin(pitch) * cos(yaw))
    yawScale = tan(localYawOffsetRad)
    pitchScale = tan(localPitchOffsetRad)
    vector = tuple(
        forward[index] + yawScale * right[index] + pitchScale * up[index]
        for index in range(3)
    )
    norm = sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise ProtocolError("local view offset produced a degenerate direction")
    normalized = tuple(value / norm for value in vector)
    yawRad, pitchRad = unitVectorToYawPitch(normalized)
    return makeSphericalPoint(yawRad, pitchRad)


def _motionCenter(motion: MotionState3D) -> SphericalPoint:
    x, y, z = motion.position
    return makeSphericalPoint(atan2(x, z), asin(max(-1.0, min(1.0, y))))


def _clampFov(value: float, geometry: GeometryConfig) -> float:
    return min(geometry.maxFovRad, max(geometry.minFovRad, value))


__all__ = ["PlannedView", "RecoveryPlanner"]
