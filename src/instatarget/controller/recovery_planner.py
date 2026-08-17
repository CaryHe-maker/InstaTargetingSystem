"""Fixed 120-degree four-corner and cubemap search planning."""

from __future__ import annotations

from collections.abc import Sequence
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
from instatarget.geometry.projection_math import (
    cameraBasis,
    makeSphericalPoint,
    unitVectorToYawPitch,
)


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
        targetCenters: Sequence[SphericalPoint] | None = None,
    ) -> tuple[PlannedView, ...]:
        del frameIndex, anchorBox, currentBox, recoveryMemory, targetCenters
        if frameWidthPx <= 0 or frameHeightPx <= 0:
            raise ProtocolError("frame dimensions must be positive")
        if attemptIndex < 0 or attemptIndex >= self._tracking.maxAttemptsPerFrame:
            raise ProtocolError(f"unsupported attemptIndex: {attemptIndex}")

        center = searchSeedCenter or (
            _motionCenter(predictedMotion) if predictedMotion is not None else fallbackBfov.center
        )
        if status is TrackStatus.LOST:
            if attemptIndex != 0:
                raise ProtocolError("LOST uses one combined 12-view recovery attempt")
            firstDirections = _cubeDirections(center)
            expansionCenter = firstDirections[1][1]
            requiredViews = 12
            budget = viewBudget if viewBudget is not None else self._tracking.maxViewsPerFrameTotal
            if budget < requiredViews:
                raise ProtocolError(
                    "view budget cannot fit LOST recovery: "
                    f"required={requiredViews}, available={budget}"
                )
            return (
                self._cubeMap(center, viewIdStart, attemptIndex, rolePrefix="cubemap")
                + self._cubeMap(
                    expansionCenter,
                    viewIdStart + 6,
                    attemptIndex,
                    rolePrefix="recovery_cubemap",
                )
            )

        if attemptIndex == 1:
            requiredViews = 4
        elif attemptIndex == 0:
            requiredViews = 6 if status is TrackStatus.UNCERTAIN else 4
        else:
            raise ProtocolError(f"unsupported {status.name} attemptIndex: {attemptIndex}")
        budget = viewBudget if viewBudget is not None else self._tracking.maxViewsPerFrameTotal
        if budget < requiredViews:
            raise ProtocolError(
                f"view budget cannot fit attempt: required={requiredViews}, available={budget}"
            )
        if attemptIndex == 1:
            return self._fourCorners(center, viewIdStart, attemptIndex)
        if status is TrackStatus.UNCERTAIN:
            return self._cubeMap(center, viewIdStart, attemptIndex, rolePrefix="cubemap")
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

    def _cubeMap(
        self,
        center: SphericalPoint,
        viewIdStart: int,
        attemptIndex: int,
        *,
        rolePrefix: str,
    ) -> tuple[PlannedView, ...]:
        return tuple(
            PlannedView(
                spec=self._viewSpec(viewIdStart + index, target),
                role=f"round{attemptIndex + 1}_{rolePrefix}_{role}",
            )
            for index, (role, target) in enumerate(_cubeDirections(center),)
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


def _cubeDirections(center: SphericalPoint) -> tuple[tuple[str, SphericalPoint], ...]:
    """Build six 120-degree faces in a frame whose front points at ``center``."""
    forward, right, up = cameraBasis(
        BFoV(center=center, horizontalFovRad=pi * 2.0 / 3.0, verticalFovRad=pi * 2.0 / 3.0)
    )
    vectors = (
        ("front", forward),
        ("right", right),
        ("back", -forward),
        ("left", -right),
        ("up", up),
        ("down", -up),
    )
    return tuple(
        (
            role,
            makeSphericalPoint(
                float(atan2(vector[0], vector[2])),
                float(asin(max(-1.0, min(1.0, float(vector[1]))))),
            ),
        )
        for role, vector in vectors
    )


__all__ = ["PlannedView", "RecoveryPlanner"]
