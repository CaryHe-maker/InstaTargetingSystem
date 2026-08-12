"""State-aware five-view, recovery-ring, and true cubemap planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, pi, sin

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
from instatarget.geometry.projection_math import makeSphericalPoint


@dataclass(frozen=True, slots=True)
class PlannedView:
    spec: ViewSpec
    role: str


class RecoveryPlanner:
    """Build deterministic views while retaining cross-frame recovery coverage."""

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
        if frameWidthPx <= 0 or frameHeightPx <= 0:
            raise ProtocolError("frame dimensions must be positive")
        center = searchSeedCenter or (
            _motionCenter(predictedMotion) if predictedMotion is not None else fallbackBfov.center
        )
        horizontalFov, verticalFov = self._contextFov(
            frameWidthPx, frameHeightPx, anchorBox, currentBox
        )
        budget = min(
            viewBudget or self._recovery.maxViewsPerFrame,
            self._tracking.maxViewsPerFrameTotal,
        )
        planned: list[PlannedView] = []
        keys: set[tuple[int, int, int, int, int, int]] = set()

        def add(
            yawRad: float,
            pitchRad: float,
            hFov: float,
            vFov: float,
            role: str,
            *,
            allowPreviouslyAttempted: bool = False,
        ) -> None:
            if len(planned) >= budget:
                return
            point = makeSphericalPoint(yawRad, max(-pi / 2.0, min(pi / 2.0, pitchRad)))
            spec = ViewSpec(
                viewId=viewIdStart + len(planned),
                bfov=BFoV(
                    center=point,
                    horizontalFovRad=_clampFov(hFov, self._geometry),
                    verticalFovRad=_clampFov(vFov, self._geometry),
                ),
                outputWidthPx=self._geometry.viewWidthPx,
                outputHeightPx=self._geometry.viewHeightPx,
            )
            epochId = recoveryMemory.epochId if recoveryMemory is not None else 0
            key = _planKey(spec, role, epochId)
            if key in keys or (
                not allowPreviouslyAttempted
                and recoveryMemory is not None
                and key in recoveryMemory.attemptedPlanKeys
            ):
                return
            keys.add(key)
            planned.append(PlannedView(spec=spec, role=role))
            if recoveryMemory is not None and status in {TrackStatus.RECOVERING, TrackStatus.LOST}:
                if len(recoveryMemory.attemptedPlanKeys) >= self._recovery.maxCoveredCells:
                    recoveryMemory.attemptedPlanKeys.clear()
                recoveryMemory.attemptedPlanKeys.add(key)
                recoveryMemory.coveredCells.add(_sphereCell(point))

        if status is TrackStatus.TRACKING:
            self._addLocalFive(add, center, horizontalFov, verticalFov, "local_corner_guard")
        elif status is TrackStatus.UNCERTAIN:
            scale = self._tracking.uncertainFovScale
            self._addLocalFive(
                add,
                center,
                horizontalFov * scale,
                verticalFov * scale,
                "uncertain_corner_guard",
            )
        elif status is TrackStatus.RECOVERING:
            add(
                center.yawRad,
                center.pitchRad,
                horizontalFov * 1.25,
                verticalFov * 1.25,
                "recovery_seed",
            )
            self._addRings(add, center, horizontalFov, verticalFov)
        elif frameIndex % self._recovery.globalSearchInterval == 0 or attemptIndex > 0:
            phase = recoveryMemory.globalScanPhase if recoveryMemory is not None else 0
            self._addCubeMap(add, center.yawRad + phase * pi / 8.0)
            if recoveryMemory is not None:
                recoveryMemory.globalScanPhase = (phase + 1) % 8
                recoveryMemory.lastGlobalScanFrameIndex = frameIndex
        else:
            add(
                center.yawRad, center.pitchRad, horizontalFov * 1.5, verticalFov * 1.5, "lost_probe"
            )

        if not planned:
            # Deduplication against previous attempts may consume the entire preferred plan.  A
            # phase-shifted seed remains bounded and guarantees forward progress.
            add(
                center.yawRad + 0.125,
                center.pitchRad,
                horizontalFov * 1.25,
                verticalFov * 1.25,
                "fallback_probe",
                allowPreviouslyAttempted=True,
            )
        return tuple(planned)

    def contextBfov(
        self,
        center: SphericalPoint,
        frameWidthPx: int,
        frameHeightPx: int,
        anchorBox: BBoxXYWH,
        currentBox: BBoxXYWH,
        uncertaintyRad: float = 0.0,
    ) -> BFoV:
        horizontalFov, verticalFov = self._contextFov(
            frameWidthPx, frameHeightPx, anchorBox, currentBox
        )
        return BFoV(
            center=center,
            horizontalFovRad=_clampFov(horizontalFov + 2.0 * uncertaintyRad, self._geometry),
            verticalFovRad=_clampFov(verticalFov + 2.0 * uncertaintyRad, self._geometry),
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
        return (
            _clampFov(2.0 * pi * widthPx / frameWidthPx, self._geometry),
            _clampFov(pi * heightPx / frameHeightPx, self._geometry),
        )

    def _addLocalFive(
        self, add, center, horizontalFov: float, verticalFov: float, role: str
    ) -> None:
        add(center.yawRad, center.pitchRad, horizontalFov, verticalFov, "primary")
        yawOffset = 0.45 * horizontalFov
        pitchOffset = 0.45 * verticalFov
        for yawSign, pitchSign in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            add(
                center.yawRad + yawSign * yawOffset,
                center.pitchRad + pitchSign * pitchOffset,
                horizontalFov,
                verticalFov,
                role,
            )

    def _addRings(self, add, center, horizontalFov: float, verticalFov: float) -> None:
        baseRadius = max(horizontalFov, verticalFov)
        for radius, count in zip(
            self._recovery.ringRadii, self._recovery.viewsPerRing, strict=True
        ):
            for index in range(count):
                angle = 2.0 * pi * index / count
                offset = radius * baseRadius
                add(
                    center.yawRad + cos(angle) * offset,
                    center.pitchRad + sin(angle) * offset,
                    horizontalFov * 1.25,
                    verticalFov * 1.25,
                    "recovery_ring",
                )

    def _addCubeMap(self, add, phaseYawRad: float) -> None:
        fov = min(self._geometry.maxFovRad, pi / 2.0 * (1.0 + self._recovery.cubeMapOverlapRatio))
        for index in range(4):
            add(phaseYawRad + index * pi / 2.0, 0.0, fov, fov, "cubemap_equator")
        add(phaseYawRad, pi / 2.0 - 1e-5, fov, fov, "cubemap_pole")
        add(phaseYawRad, -pi / 2.0 + 1e-5, fov, fov, "cubemap_pole")


def _motionCenter(motion: MotionState3D) -> SphericalPoint:
    x, y, z = motion.position
    return makeSphericalPoint(atan2(x, z), asin(max(-1.0, min(1.0, y))))


def _clampFov(value: float, geometry: GeometryConfig) -> float:
    return min(geometry.maxFovRad, max(geometry.minFovRad, value))


def _planKey(spec: ViewSpec, role: str, epochId: int) -> tuple[int, int, int, int, int, int]:
    return (
        round(spec.bfov.center.yawRad * 1000),
        round(spec.bfov.center.pitchRad * 1000),
        round(spec.bfov.horizontalFovRad * 1000),
        round(spec.bfov.verticalFovRad * 1000),
        sum((index + 1) * ord(char) for index, char in enumerate(role)),
        epochId,
    )


def _sphereCell(point: SphericalPoint) -> tuple[int, int]:
    return round(point.yawRad * 8.0 / pi), round(point.pitchRad * 8.0 / pi)


__all__ = ["PlannedView", "RecoveryPlanner"]
