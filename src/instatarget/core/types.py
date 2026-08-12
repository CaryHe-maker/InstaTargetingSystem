"""Immutable data contracts shared by all project modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from math import isfinite, sqrt
from typing import NewType

import numpy as np
from numpy.typing import NDArray

from instatarget.core.errors import DepthError, ProtocolError

FrameIndex = NewType("FrameIndex", int)
SequenceId = NewType("SequenceId", str)

UNIT_VECTOR_TOLERANCE = 1e-6


def _requireFinite(name: str, *values: float) -> None:
    if not all(isfinite(value) for value in values):
        raise ProtocolError(f"{name} must contain only finite values")


def _requireProbability(name: str, value: float) -> None:
    _requireFinite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ProtocolError(f"{name} must be in [0, 1], actual={value}")


@dataclass(frozen=True, slots=True)
class BBoxXYWH:
    """An ERP or local-view bounding box in continuous pixel coordinates."""

    xPx: float
    yPx: float
    widthPx: float
    heightPx: float

    def __post_init__(self) -> None:
        _requireFinite("bbox", self.xPx, self.yPx, self.widthPx, self.heightPx)
        if self.widthPx <= 0.0 or self.heightPx <= 0.0:
            raise ProtocolError(
                f"bbox dimensions must be positive, actual=({self.widthPx}, {self.heightPx})"
            )


@dataclass(frozen=True, slots=True)
class SphericalPoint:
    """A unit direction and its yaw/pitch representation in radians."""

    x: float
    y: float
    z: float
    yawRad: float
    pitchRad: float

    def __post_init__(self) -> None:
        _requireFinite("spherical point", self.x, self.y, self.z, self.yawRad, self.pitchRad)
        norm = sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        if abs(norm - 1.0) > UNIT_VECTOR_TOLERANCE:
            raise ProtocolError(f"spherical point must be a unit vector, norm={norm}")
        if not -np.pi <= self.yawRad < np.pi:
            raise ProtocolError(f"yawRad must be in [-pi, pi), actual={self.yawRad}")
        if not -np.pi / 2.0 <= self.pitchRad <= np.pi / 2.0:
            raise ProtocolError(f"pitchRad must be in [-pi/2, pi/2], actual={self.pitchRad}")


@dataclass(frozen=True, slots=True)
class BFoV:
    """A bounded field of view on the unit sphere."""

    center: SphericalPoint
    horizontalFovRad: float
    verticalFovRad: float
    rollRad: float = 0.0

    def __post_init__(self) -> None:
        _requireFinite("BFoV", self.horizontalFovRad, self.verticalFovRad, self.rollRad)
        if not 0.0 < self.horizontalFovRad < 2.0 * np.pi:
            raise ProtocolError("horizontalFovRad must be in (0, 2*pi)")
        if not 0.0 < self.verticalFovRad < np.pi:
            raise ProtocolError("verticalFovRad must be in (0, pi)")


@dataclass(frozen=True, slots=True)
class DepthPlane:
    """A depth array and its explicit validity mask."""

    values: NDArray[np.float32]
    validMask: NDArray[np.bool_]
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray) or self.values.dtype != np.float32:
            raise DepthError("depth values must be a float32 NumPy array")
        if self.values.ndim != 2:
            raise DepthError(f"depth values must have shape [H, W], actual={self.values.shape}")
        if not isinstance(self.validMask, np.ndarray) or self.validMask.dtype != np.bool_:
            raise DepthError("depth validMask must be a bool NumPy array")
        if self.validMask.shape != self.values.shape:
            raise DepthError(
                "depth values and validMask must have identical shapes, "
                f"actual={self.values.shape} and {self.validMask.shape}"
            )
        if not self.unit.strip():
            raise DepthError("depth unit must be a non-empty string")
        if not np.isfinite(self.values[self.validMask]).all():
            raise DepthError("valid depth values must be finite")
        if (self.values[self.validMask] < 0.0).any():
            raise DepthError("valid depth values must be non-negative")


@dataclass(frozen=True, slots=True)
class SegmentationPlane:
    """Optional semantic and instance maps aligned with an ERP frame."""

    semantic: NDArray[np.int32] | None
    instance: NDArray[np.int32] | None
    classNames: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shapes: set[tuple[int, ...]] = set()
        for name, plane in (("semantic", self.semantic), ("instance", self.instance)):
            if plane is None:
                continue
            if not isinstance(plane, np.ndarray) or plane.dtype != np.int32:
                raise ProtocolError(f"{name} segmentation must be an int32 NumPy array")
            if plane.ndim != 2:
                raise ProtocolError(f"{name} segmentation must have shape [H, W]")
            shapes.add(plane.shape)
        if len(shapes) > 1:
            raise ProtocolError("semantic and instance segmentation shapes must match")
        if any(
            not isinstance(key, int) or not isinstance(value, str)
            for key, value in self.classNames.items()
        ):
            raise ProtocolError("classNames must map integer IDs to strings")


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One aligned ERP frame and all modalities available for it."""

    sequenceId: SequenceId
    frameIndex: FrameIndex
    timestampNs: int
    rgb: NDArray[np.uint8]
    depth: DepthPlane | None = None
    segmentation: SegmentationPlane | None = None

    def __post_init__(self) -> None:
        if not str(self.sequenceId):
            raise ProtocolError("sequenceId must be non-empty")
        if int(self.frameIndex) < 0:
            raise ProtocolError(f"frameIndex must be non-negative, actual={self.frameIndex}")
        if self.timestampNs < 0:
            raise ProtocolError(f"timestampNs must be non-negative, actual={self.timestampNs}")
        if not isinstance(self.rgb, np.ndarray) or self.rgb.dtype != np.uint8:
            raise ProtocolError("rgb must be a uint8 NumPy array")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ProtocolError(f"rgb must have shape [H, W, 3], actual={self.rgb.shape}")
        if self.rgb.shape[0] == 0 or self.rgb.shape[1] == 0:
            raise ProtocolError("rgb frame dimensions must be positive")
        frameShape = self.rgb.shape[:2]
        if self.depth is not None and self.depth.values.shape != frameShape:
            raise ProtocolError(
                f"depth must align with rgb, actual={self.depth.values.shape} and {frameShape}"
            )
        if self.segmentation is not None:
            for plane in (self.segmentation.semantic, self.segmentation.instance):
                if plane is not None and plane.shape != frameShape:
                    raise ProtocolError(
                        f"segmentation must align with rgb, actual={plane.shape} and {frameShape}"
                    )


@dataclass(frozen=True, slots=True)
class DepthSummary:
    """Robust statistics for the depth values inside a target region."""

    medianDepth: float
    meanDepth: float
    validRatio: float
    minDepth: float
    maxDepth: float
    confidence: float

    def __post_init__(self) -> None:
        _requireFinite(
            "depth summary",
            self.medianDepth,
            self.meanDepth,
            self.validRatio,
            self.minDepth,
            self.maxDepth,
            self.confidence,
        )
        _requireProbability("validRatio", self.validRatio)
        _requireProbability("depth confidence", self.confidence)
        if min(self.medianDepth, self.meanDepth, self.minDepth, self.maxDepth) < 0.0:
            raise DepthError("depth summary values must be non-negative")
        if self.minDepth > self.maxDepth:
            raise DepthError("minDepth must not exceed maxDepth")
        if not self.minDepth <= self.medianDepth <= self.maxDepth:
            raise DepthError("medianDepth must be within [minDepth, maxDepth]")
        if not self.minDepth <= self.meanDepth <= self.maxDepth:
            raise DepthError("meanDepth must be within [minDepth, maxDepth]")


@dataclass(frozen=True, slots=True)
class MotionState3D:
    """Predicted spherical direction, velocity, range, and confidence."""

    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    rangeDepth: float
    rangeVelocity: float
    confidence: float

    def __post_init__(self) -> None:
        if len(self.position) != 3 or len(self.velocity) != 3:
            raise ProtocolError("motion position and velocity must contain three components")
        _requireFinite(
            "motion state",
            *self.position,
            *self.velocity,
            self.rangeDepth,
            self.rangeVelocity,
            self.confidence,
        )
        if self.rangeDepth < 0.0:
            raise ProtocolError("rangeDepth must be non-negative")
        _requireProbability("motion confidence", self.confidence)


class TrackStatus(Enum):
    TRACKING = auto()
    UNCERTAIN = auto()
    RECOVERING = auto()
    LOST = auto()


class ResultSource(Enum):
    """How the controller produced the committed frame result."""

    INITIAL = auto()
    OBSERVED_CONFIRMED = auto()
    OBSERVED_REACQUIRED = auto()
    OBSERVED_WEAK_BLEND = auto()
    MOTION_PREDICTED = auto()


@dataclass(frozen=True, slots=True)
class TrackResult:
    """The single committed output for one input frame."""

    sequenceId: SequenceId
    frameIndex: FrameIndex
    bbox: BBoxXYWH
    bfov: BFoV
    confidence: float
    status: TrackStatus
    valid: bool
    depthSummary: DepthSummary | None = None
    resultSource: ResultSource = ResultSource.OBSERVED_CONFIRMED

    def __post_init__(self) -> None:
        if not str(self.sequenceId):
            raise ProtocolError("sequenceId must be non-empty")
        if int(self.frameIndex) < 0:
            raise ProtocolError("frameIndex must be non-negative")
        _requireProbability("track confidence", self.confidence)


@dataclass(frozen=True, slots=True)
class ViewSpec:
    """The perspective view requested from an ERP frame."""

    viewId: int
    bfov: BFoV
    outputWidthPx: int
    outputHeightPx: int

    def __post_init__(self) -> None:
        if self.viewId < 0:
            raise ProtocolError("viewId must be non-negative")
        if self.outputWidthPx <= 0 or self.outputHeightPx <= 0:
            raise ProtocolError("view output dimensions must be positive")


@dataclass(frozen=True, slots=True)
class LocalView:
    """A perspective RGB crop and an optional synchronized depth crop."""

    spec: ViewSpec
    rgb: NDArray[np.uint8]
    depth: DepthPlane | None = None

    def __post_init__(self) -> None:
        expectedShape = (self.spec.outputHeightPx, self.spec.outputWidthPx)
        if not isinstance(self.rgb, np.ndarray) or self.rgb.dtype != np.uint8:
            raise ProtocolError("local rgb must be a uint8 NumPy array")
        if self.rgb.shape != (*expectedShape, 3):
            raise ProtocolError(
                f"local rgb shape must be {(*expectedShape, 3)}, actual={self.rgb.shape}"
            )
        if self.depth is not None and self.depth.values.shape != expectedShape:
            raise ProtocolError("local depth must align with the local rgb view")


@dataclass(frozen=True, slots=True)
class LocalObservation:
    viewId: int
    bbox: BBoxXYWH
    modelScore: float
    appearanceScore: float
    depthScore: float
    fusedScore: float
    depthSummary: DepthSummary | None
    latencyNs: int

    def __post_init__(self) -> None:
        if self.viewId < 0 or self.latencyNs < 0:
            raise ProtocolError("viewId and latencyNs must be non-negative")
        for name, value in (
            ("modelScore", self.modelScore),
            ("appearanceScore", self.appearanceScore),
            ("depthScore", self.depthScore),
            ("fusedScore", self.fusedScore),
        ):
            _requireProbability(name, value)


class TemplateCommandKind(Enum):
    KEEP = auto()
    UPDATE_RECENT = auto()
    UPDATE_STABLE = auto()
    RESET_TO_ANCHOR = auto()


@dataclass(frozen=True, slots=True)
class TemplateCommand:
    kind: TemplateCommandKind
    frameIndex: FrameIndex
    viewId: int | None
    localBox: BBoxXYWH | None
    expectedRevision: int

    def __post_init__(self) -> None:
        if int(self.frameIndex) < 0 or self.expectedRevision < 0:
            raise ProtocolError("template frameIndex and revision must be non-negative")
        if self.viewId is not None and self.viewId < 0:
            raise ProtocolError("template viewId must be non-negative")
        hasSelection = self.viewId is not None and self.localBox is not None
        if self.kind in {TemplateCommandKind.UPDATE_RECENT, TemplateCommandKind.UPDATE_STABLE}:
            if not hasSelection:
                raise ProtocolError("template update commands require viewId and localBox")
        elif self.viewId is not None or self.localBox is not None:
            raise ProtocolError("KEEP and RESET_TO_ANCHOR must not select a local box")


@dataclass(frozen=True, slots=True)
class SearchPlan:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    views: tuple[ViewSpec, ...]
    templateCommand: TemplateCommand
    predictedMotion: MotionState3D | None
    transactionId: int = 0
    attemptIndex: int = 0
    recoveryEpochId: int = 0
    viewRoles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.sequenceId) or int(self.frameIndex) < 0 or self.stateRevision < 0:
            raise ProtocolError("search plan identity and revision must be valid")
        viewIds = tuple(view.viewId for view in self.views)
        if len(viewIds) != len(set(viewIds)):
            raise ProtocolError("search plan viewIds must be unique")
        if self.templateCommand.frameIndex != self.frameIndex:
            raise ProtocolError("template command and search plan frameIndex must match")
        # Backend template commands advance once per inference attempt, while controller state
        # revisions advance once per committed frame.  They are equal on the first attempt of a
        # simple frame but intentionally diverge after same-frame escalation.
        if self.transactionId < 0 or self.attemptIndex < 0 or self.recoveryEpochId < 0:
            raise ProtocolError("search plan transaction identity must be non-negative")
        if self.viewRoles and len(self.viewRoles) != len(self.views):
            raise ProtocolError("search plan viewRoles must align with views")


@dataclass(frozen=True, slots=True)
class InitializationPlan:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    templateView: ViewSpec
    templateBox: BBoxXYWH

    def __post_init__(self) -> None:
        if int(self.frameIndex) != 0:
            raise ProtocolError("initialization plan must target frame 0")
        if self.stateRevision < 0 or not str(self.sequenceId):
            raise ProtocolError("initialization identity and revision must be valid")


@dataclass(frozen=True, slots=True)
class ProjectedObservation:
    viewId: int
    bfov: BFoV
    bbox: BBoxXYWH
    modelScore: float
    appearanceScore: float
    motionScore: float
    scaleScore: float
    depthScore: float
    fusedScore: float
    depthSummary: DepthSummary | None
    localBox: BBoxXYWH | None = None

    def __post_init__(self) -> None:
        if self.viewId < 0:
            raise ProtocolError("viewId must be non-negative")
        for name, value in (
            ("modelScore", self.modelScore),
            ("appearanceScore", self.appearanceScore),
            ("motionScore", self.motionScore),
            ("scaleScore", self.scaleScore),
            ("depthScore", self.depthScore),
            ("fusedScore", self.fusedScore),
        ):
            _requireProbability(name, value)


@dataclass(frozen=True, slots=True)
class AirSim360Record:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    rgbPath: str
    depthPath: str | None
    semanticPath: str | None
    instancePath: str | None

    def __post_init__(self) -> None:
        if not str(self.sequenceId) or int(self.frameIndex) < 0:
            raise ProtocolError("AirSim360 record identity must be valid")
        if not self.rgbPath.strip():
            raise ProtocolError("AirSim360 rgbPath must be non-empty")


@dataclass(frozen=True, slots=True)
class InitRequest:
    plan: InitializationPlan
    frame: FramePacket

    def __post_init__(self) -> None:
        if (
            self.plan.sequenceId != self.frame.sequenceId
            or self.plan.frameIndex != self.frame.frameIndex
        ):
            raise ProtocolError("initialization plan and frame identity must match")


@dataclass(frozen=True, slots=True)
class InitResponse:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    depthSummary: DepthSummary | None

    def __post_init__(self) -> None:
        if not str(self.sequenceId) or int(self.frameIndex) < 0 or self.stateRevision < 0:
            raise ProtocolError("initialization response identity and revision must be valid")


@dataclass(frozen=True, slots=True)
class SearchRequest:
    plan: SearchPlan
    frame: FramePacket

    def __post_init__(self) -> None:
        if (
            self.plan.sequenceId != self.frame.sequenceId
            or self.plan.frameIndex != self.frame.frameIndex
        ):
            raise ProtocolError("search plan and frame identity must match")


@dataclass(frozen=True, slots=True)
class InferResponse:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    observations: tuple[LocalObservation, ...]
    depthSummaries: dict[int, DepthSummary]
    transactionId: int = 0
    attemptIndex: int = 0
    recoveryEpochId: int = 0

    def __post_init__(self) -> None:
        if not str(self.sequenceId) or int(self.frameIndex) < 0 or self.stateRevision < 0:
            raise ProtocolError("inference response identity and revision must be valid")
        observationIds = tuple(observation.viewId for observation in self.observations)
        if len(observationIds) != len(set(observationIds)):
            raise ProtocolError("inference response observation viewIds must be unique")
        if any(viewId < 0 for viewId in self.depthSummaries):
            raise ProtocolError("inference response depth summary viewIds must be non-negative")
        if self.transactionId < 0 or self.attemptIndex < 0 or self.recoveryEpochId < 0:
            raise ProtocolError("inference response transaction identity must be non-negative")


@dataclass(frozen=True, slots=True)
class ResultPacket:
    result: TrackResult
    totalLatencyNs: int

    def __post_init__(self) -> None:
        if self.totalLatencyNs < 0:
            raise ProtocolError("totalLatencyNs must be non-negative")


@dataclass(frozen=True, slots=True)
class FatalError:
    stage: str
    message: str
    frameIndex: FrameIndex | None

    def __post_init__(self) -> None:
        if not self.stage.strip() or not self.message.strip():
            raise ProtocolError("fatal error stage and message must be non-empty")
        if self.frameIndex is not None and int(self.frameIndex) < 0:
            raise ProtocolError("fatal error frameIndex must be non-negative")
