"""State-specific four-corner and cubemap search planning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import asin, atan2, cos, isfinite, pi, sin, sqrt, tan

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


def ViewSpecType1(
    center: SphericalPoint,
    horizontalSizeRad: float | tuple[float, float],
    verticalSizeRad: float | None = None,
    *,
    viewIdStart: int = 0,
    outputWidthPx: int = 256,
    outputHeightPx: int = 256,
    minFovRad: float = pi / 6.0,
    maxFovRad: float = 2.0 * pi / 3.0,
) -> tuple[ViewSpec, ...]:
    """Build the four dynamic VStype1 views around ``center``.

    ``horizontalSizeRad`` and ``verticalSizeRad`` are the predicted target angular
    width and height from the previous frame.  Each view is three times as wide and
    three times as high as that target (nine times its area), clamped to
    ``[minFovRad, maxFovRad]``.  The view centers are offset by a third of their
    corresponding FOV, preserving the overlap ratio of the former
    120-degree/40-degree layout.

    A two-element ``(width, height)`` tuple is accepted as a convenience for callers
    that already keep the predicted size together.
    """
    if verticalSizeRad is None:
        if not isinstance(horizontalSizeRad, tuple) or len(horizontalSizeRad) != 2:
            raise ProtocolError("ViewSpecType1 requires horizontal and vertical sizes")
        horizontalSizeRad, verticalSizeRad = horizontalSizeRad
    if (
        not isfinite(float(horizontalSizeRad))
        or not isfinite(float(verticalSizeRad))
        or float(horizontalSizeRad) <= 0.0
        or float(verticalSizeRad) <= 0.0
    ):
        raise ProtocolError("ViewSpecType1 sizes must be finite and positive")
    if viewIdStart < 0:
        raise ProtocolError("viewIdStart must be non-negative")
    if outputWidthPx <= 0 or outputHeightPx <= 0:
        raise ProtocolError("view output dimensions must be positive")
    if (
        not isfinite(minFovRad)
        or not isfinite(maxFovRad)
        or not 0.0 < minFovRad <= maxFovRad < pi
    ):
        raise ProtocolError("FOV limits must satisfy 0 < minFovRad <= maxFovRad < pi")

    # The target angular extent is tripled independently on each axis.  Keep the
    # resulting BFoV in the perspective-camera domain for unusually large targets;
    # ordinary tracking boxes are unaffected by this safety bound.
    horizontalFov = _dynamicFov(
        3.0 * float(horizontalSizeRad),
        minFovRad,
        maxFovRad,
    )
    verticalFov = _dynamicFov(
        3.0 * float(verticalSizeRad),
        minFovRad,
        maxFovRad,
    )
    yawOffset = horizontalFov / 3.0
    pitchOffset = verticalFov / 3.0
    roles = (
        (-yawOffset, pitchOffset),
        (yawOffset, pitchOffset),
        (-yawOffset, -pitchOffset),
        (yawOffset, -pitchOffset),
    )
    return tuple(
        ViewSpec(
            viewId=viewIdStart + index,
            bfov=BFoV(
                center=_offsetDirection(center, localYawOffset, localPitchOffset),
                horizontalFovRad=horizontalFov,
                verticalFovRad=verticalFov,
            ),
            outputWidthPx=outputWidthPx,
            outputHeightPx=outputHeightPx,
        )
        for index, (localYawOffset, localPitchOffset) in enumerate(roles)
    )


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
        trackingSize = (
            _trackingSize(predictedMotion, fallbackBfov)
            if status is TrackStatus.TRACKING
            else None
        )
        if attemptIndex == 1:
            return self._fourCorners(
                center,
                viewIdStart,
                attemptIndex,
                dynamicSize=trackingSize,
            )
        if status is TrackStatus.UNCERTAIN:
            return self._cubeMap(center, viewIdStart, attemptIndex, rolePrefix="cubemap")
        return self._fourCorners(
            center,
            viewIdStart,
            attemptIndex,
            dynamicSize=trackingSize,
        )

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
        *,
        dynamicSize: tuple[float, float] | None = None,
    ) -> tuple[PlannedView, ...]:
        roles = (
            "left_top",
            "right_top",
            "left_bottom",
            "right_bottom",
        )
        if dynamicSize is not None:
            specs = ViewSpecType1(
                center,
                dynamicSize,
                viewIdStart=viewIdStart,
                outputWidthPx=self._geometry.viewWidthPx,
                outputHeightPx=self._geometry.viewHeightPx,
                maxFovRad=self._geometry.maxFovRad,
            )
        else:
            offset = 40.0 * pi / 180.0
            specs = tuple(
                self._viewSpec(
                    viewIdStart + index,
                    _offsetDirection(center, yawOffset, pitchOffset),
                )
                for index, (yawOffset, pitchOffset) in enumerate(
                    (
                        (-offset, offset),
                        (offset, offset),
                        (-offset, -offset),
                        (offset, -offset),
                    )
                )
            )
        return tuple(
            PlannedView(spec=spec, role=f"round{attemptIndex + 1}_{role}")
            for spec, role in zip(specs, roles, strict=True)
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


def _trackingSize(
    motion: MotionState3D | None,
    fallbackBfov: BFoV,
) -> tuple[float, float]:
    """Return the previous-frame target size used by every TRACKING attempt."""
    if motion is not None and motion.horizontalSizeRad > 0.0 and motion.verticalSizeRad > 0.0:
        return motion.horizontalSizeRad, motion.verticalSizeRad
    # The initialized/current BFoV is the frame-0 (or last committed) standard
    # tracking box, so it is the correct first-frame basis when an estimator has
    # not exposed angular scale yet.  TRACKING never falls back to fixed corners.
    return fallbackBfov.horizontalFovRad, fallbackBfov.verticalFovRad


def _dynamicFov(value: float, minFovRad: float, maxFovRad: float) -> float:
    """Keep a dynamic view valid for perspective projection."""
    return min(maxFovRad, max(minFovRad, value))


__all__ = ["PlannedView", "RecoveryPlanner", "ViewSpecType1"]
