"""Immutable state records and bounded controller transaction data for V2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto

from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    DepthSummary,
    FrameIndex,
    MotionState3D,
    ProjectedObservation,
    ResultSource,
    SequenceId,
    SphericalPoint,
    TrackStatus,
)


class TrackMode(Enum):
    INIT = auto()
    TRACKING = auto()
    UNCERTAIN = auto()
    RECOVERING = auto()
    LOST = auto()
    TERMINATED = auto()


class AttemptKind(Enum):
    PRIMARY = auto()
    ESCALATION = auto()


class EvidenceLevel(Enum):
    CONFIRMED = auto()
    WEAK = auto()
    REJECTED = auto()
    REACQUIRED = auto()


class EvaluationReason(StrEnum):
    NO_ELIGIBLE_CLUSTER = "no_eligible_cluster"
    INSUFFICIENT_VIEW_SUPPORT = "insufficient_view_support"
    BELOW_UNCERTAIN_THRESHOLD = "below_uncertain_threshold"


class TransitionReason(Enum):
    INITIALIZED = auto()
    RELIABLE_MEASUREMENT = auto()
    WEAK_MEASUREMENT = auto()
    HARD_MISS = auto()
    PATIENCE_EXHAUSTED = auto()
    RECOVERY_PROGRESS = auto()
    REACQUIRED = auto()
    RECOVERY_EXHAUSTED = auto()
    END_OF_STREAM = auto()
    EXTERNAL_RESET = auto()


@dataclass(frozen=True, slots=True)
class MotionSample:
    frameIndex: FrameIndex
    timestampNs: int
    center: SphericalPoint
    horizontalSizeRad: float
    verticalSizeRad: float
    confidence: float
    rangeDepth: float | None
    rangeConfidence: float


@dataclass(frozen=True, slots=True)
class MotionPrediction:
    """Prediction plus uncertainty used by planning and candidate scoring."""

    sourceRevision: int
    targetFrameIndex: FrameIndex
    horizonFrames: int
    center: SphericalPoint
    horizontalSizeRad: float
    verticalSizeRad: float
    tangentVelocityRadPerSec: tuple[float, float]
    rangeDepth: float | None
    rangeVelocityPerSec: float | None
    angularUncertaintyRad: float
    scaleUncertainty: float
    rangeUncertainty: float | None
    confidence: float
    degradedReasons: tuple[str, ...] = ()

    @property
    def motionState(self) -> MotionState3D:
        """Expose the legacy public motion type without losing V2 uncertainty."""
        return MotionState3D(
            position=(self.center.x, self.center.y, self.center.z),
            velocity=(
                self.tangentVelocityRadPerSec[0],
                self.tangentVelocityRadPerSec[1],
                0.0,
            ),
            rangeDepth=self.rangeDepth or 0.0,
            rangeVelocity=self.rangeVelocityPerSec or 0.0,
            confidence=self.confidence,
        )


@dataclass(frozen=True, slots=True)
class ConfirmedTargetState:
    frameIndex: FrameIndex
    bbox: BBoxXYWH
    bfov: BFoV
    confidence: float
    depthSummary: DepthSummary | None


@dataclass(frozen=True, slots=True)
class StateInstance:
    stateId: int
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    mode: TrackMode
    enteredFrom: TrackMode | None
    entryReason: TransitionReason
    prediction: MotionPrediction | None
    searchSeedCenter: SphericalPoint | None
    recoveryEpochId: int
    modeAgeFrames: int
    stableStreak: int
    weakStreak: int
    missStreak: int

    @property
    def publicStatus(self) -> TrackStatus:
        if self.mode is TrackMode.RECOVERING:
            return TrackStatus.RECOVERING
        if self.mode is TrackMode.LOST:
            return TrackStatus.LOST
        if self.mode is TrackMode.UNCERTAIN:
            return TrackStatus.UNCERTAIN
        return TrackStatus.TRACKING


@dataclass(frozen=True, slots=True)
class StateObservation:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    transactionId: int
    stateId: int
    attemptIndex: int
    evaluatedMode: TrackMode
    predictedCenter: SphericalPoint
    searchSeedCenter: SphericalPoint
    measuredBfov: BFoV | None
    measuredBbox: BBoxXYWH | None
    measuredCenter: SphericalPoint | None
    proposedOutputBfov: BFoV
    proposedOutputBbox: BBoxXYWH
    proposedResultSource: ResultSource
    candidateCount: int
    eligibleCandidateCount: int
    clusterCount: int
    sourceViewIds: tuple[int, ...]
    representativeViewId: int | None
    representativeLocalBox: BBoxXYWH | None
    supportViewCount: int
    backendScore: float
    motionScore: float
    scaleScore: float
    depthConsistencyScore: float | None
    supportScore: float
    agreementScore: float
    stateScore: float
    evidence: EvidenceLevel
    hardGatePassed: bool
    supported: bool
    escalationRecommended: bool
    reacquired: bool
    depthSummary: DepthSummary | None
    rejectionReasons: tuple[EvaluationReason, ...] = ()

    @property
    def resultSource(self) -> ResultSource:
        """Compatibility alias for diagnostics written before the V2 field was explicit."""
        return self.proposedResultSource


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    kind: AttemptKind
    attemptIndex: int
    plan: object
    observations: tuple[ProjectedObservation, ...]
    evaluation: StateObservation


@dataclass(slots=True)
class RecoveryMemory:
    epochId: int = 0
    startedFrameIndex: FrameIndex | None = None
    framesSpent: int = 0
    globalScanPhase: int = 0
    coveredCells: set[tuple[int, int]] = field(default_factory=set)
    attemptedPlanKeys: set[tuple[int, int, int, int, int, int]] = field(default_factory=set)
    bestSeedCenter: SphericalPoint | None = None
    bestSeedScore: float = 0.0
    bestSeedFrameIndex: FrameIndex | None = None
    lastGlobalScanFrameIndex: FrameIndex | None = None

    def reset(self, frameIndex: FrameIndex) -> None:
        self.epochId += 1
        self.startedFrameIndex = frameIndex
        self.framesSpent = 0
        self.globalScanPhase = 0
        self.coveredCells.clear()
        self.attemptedPlanKeys.clear()
        self.bestSeedCenter = None
        self.bestSeedScore = 0.0
        self.bestSeedFrameIndex = None
        self.lastGlobalScanFrameIndex = None


@dataclass(slots=True)
class FrameTransaction:
    transactionId: int
    frame: object
    state: StateInstance
    attemptIndex: int = 0
    escalationUsed: bool = False
    remainingViews: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)
    recoveryMemory: RecoveryMemory | None = None


@dataclass(frozen=True, slots=True)
class TransitionDecision:
    action: str
    nextMode: TrackMode
    reason: TransitionReason
    acceptMeasurement: bool
    resetMotionHistory: bool = False
    resetRecoveryEpoch: bool = False


def newMotionHistory(maxlen: int) -> deque[MotionSample]:
    return deque(maxlen=maxlen)


__all__ = [
    "AttemptKind",
    "AttemptRecord",
    "ConfirmedTargetState",
    "EvidenceLevel",
    "EvaluationReason",
    "FrameTransaction",
    "MotionPrediction",
    "MotionSample",
    "RecoveryMemory",
    "StateInstance",
    "StateObservation",
    "TrackMode",
    "TransitionDecision",
    "TransitionReason",
    "newMotionHistory",
]
