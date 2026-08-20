"""State and validation rules for the disabled-by-default speculative pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from math import acos, atan2, isfinite, log

import numpy as np

from instatarget.core.config import SpeculativePipelineConfig
from instatarget.core.errors import ProtocolError
from instatarget.core.types import (
    FrameIndex,
    InferenceRole,
    RoutedLocalObservation,
    SequenceId,
    SphericalPoint,
    TaskKey,
    TrackResult,
    TrackStatus,
    ViewSpec,
)
from instatarget.geometry.projection_math import cameraBasis


class RollbackReason(StrEnum):
    NONE = "none"
    PIPELINE_DISABLED = "pipeline_disabled"
    SEQUENCE_MISMATCH = "sequence_mismatch"
    FRAME_AGE_MISMATCH = "frame_age_mismatch"
    DIRECTION_CONFIDENCE = "direction_confidence"
    GENERATION_MISMATCH = "generation_mismatch"
    REVISION_MISMATCH = "revision_mismatch"
    STALE = "stale"
    EMPTY_OUTPUT = "empty_output"
    ROUTING_MISMATCH = "routing_mismatch"
    NONFINITE_OUTPUT = "nonfinite_output"
    EXPLICIT_LOST = "explicit_lost"
    CENTER_GAP = "center_gap"
    SCALE_GAP = "scale_gap"
    COVERAGE = "coverage"
    SEQUENCE_CLOSED = "sequence_closed"


@dataclass(frozen=True, slots=True)
class SpeculativeState:
    """An immutable next-frame plan derived without mutating committed state."""

    sequenceId: SequenceId
    frameIndex: FrameIndex
    directionCenter: SphericalPoint
    horizontalSizeRad: float
    verticalSizeRad: float
    motionUncertaintyRad: float
    directionConfidence: float
    sourceStateRevision: int
    generation: int
    views: tuple[ViewSpec, ...]
    taskKeys: tuple[TaskKey, ...]
    stale: bool = False

    def __post_init__(self) -> None:
        if not str(self.sequenceId) or int(self.frameIndex) <= 0:
            raise ProtocolError("speculative state identity must target a non-initial frame")
        values = (
            self.horizontalSizeRad,
            self.verticalSizeRad,
            self.motionUncertaintyRad,
            self.directionConfidence,
        )
        if not all(isfinite(value) for value in values):
            raise ProtocolError("speculative state values must be finite")
        if self.horizontalSizeRad <= 0.0 or self.verticalSizeRad <= 0.0:
            raise ProtocolError("speculative angular sizes must be positive")
        if self.motionUncertaintyRad < 0.0:
            raise ProtocolError("speculative motion uncertainty must be non-negative")
        if not 0.0 <= self.directionConfidence <= 1.0:
            raise ProtocolError("speculative direction confidence must be in [0, 1]")
        if self.sourceStateRevision < 0 or self.generation < 0:
            raise ProtocolError("speculative revision and generation must be non-negative")
        viewIds = tuple(view.viewId for view in self.views)
        if not viewIds or len(viewIds) != len(set(viewIds)):
            raise ProtocolError("speculative views must be non-empty with unique viewIds")
        if len(self.taskKeys) != len(self.views):
            raise ProtocolError("speculative task keys must align with views")
        for key, view in zip(self.taskKeys, self.views, strict=True):
            if (
                key.sequenceId != self.sequenceId
                or key.frameIndex != self.frameIndex
                or key.viewId != view.viewId
                or key.attemptIndex != 0
                or key.generation != self.generation
                or key.role is not InferenceRole.SPECULATIVE_ROUND1_DIRECTION
            ):
                raise ProtocolError("speculative TaskKey does not match its state and view")


@dataclass(frozen=True, slots=True)
class SpeculativeDecision:
    accepted: bool
    invalidated: bool
    rollbackReason: RollbackReason
    centerGapRad: float | None
    centerGapRatio: float | None
    logScaleGap: float | None
    coverageAfterCorrection: bool
    speculativeGeneration: int
    formalStateRevision: int


@dataclass(frozen=True, slots=True)
class SpeculativeSummary:
    evaluatedCount: int
    acceptedCount: int
    rollbackCount: int
    acceptanceRate: float
    rollbackRate: float
    rollbackReasons: dict[str, int]
    rollbackTargetMet: bool


class SpeculativePipeline:
    """Own generations and consume a speculative result at most once."""

    def __init__(self, config: SpeculativePipelineConfig) -> None:
        self._config = config
        self._generation = 0
        self._pending: SpeculativeState | None = None
        self._closedSequence: str | None = None
        self._decisions: list[SpeculativeDecision] = []

    @property
    def pending(self) -> SpeculativeState | None:
        return self._pending

    @property
    def generation(self) -> int:
        return self._generation

    def create(
        self,
        *,
        sequenceId: SequenceId,
        frameIndex: FrameIndex,
        directionCenter: SphericalPoint,
        horizontalSizeRad: float,
        verticalSizeRad: float,
        motionUncertaintyRad: float,
        directionConfidence: float,
        sourceStateRevision: int,
        views: Sequence[ViewSpec],
    ) -> SpeculativeState:
        if not self._config.enabled:
            raise ProtocolError("speculative pipeline is disabled")
        if self._closedSequence == str(sequenceId):
            raise ProtocolError("cannot create speculation after sequence close")
        self._generation += 1
        viewTuple = tuple(views)
        taskKeys = tuple(
            TaskKey(
                sequenceId=sequenceId,
                frameIndex=frameIndex,
                attemptIndex=0,
                viewId=view.viewId,
                generation=self._generation,
                role=InferenceRole.SPECULATIVE_ROUND1_DIRECTION,
            )
            for view in viewTuple
        )
        self._pending = SpeculativeState(
            sequenceId=sequenceId,
            frameIndex=frameIndex,
            directionCenter=directionCenter,
            horizontalSizeRad=horizontalSizeRad,
            verticalSizeRad=verticalSizeRad,
            motionUncertaintyRad=motionUncertaintyRad,
            directionConfidence=directionConfidence,
            sourceStateRevision=sourceStateRevision,
            generation=self._generation,
            views=viewTuple,
            taskKeys=taskKeys,
        )
        return self._pending

    def invalidatePending(self) -> None:
        if self._pending is not None:
            self._pending = replace(self._pending, stale=True)

    def evaluate(
        self,
        *,
        committedResult: TrackResult,
        formalStateRevision: int,
        routedObservations: Sequence[RoutedLocalObservation],
    ) -> SpeculativeDecision:
        state = self._pending
        if state is None:
            raise ProtocolError("no speculative state is pending")
        decision = evaluateSpeculation(
            config=self._config,
            state=state,
            committedResult=committedResult,
            formalStateRevision=formalStateRevision,
            currentGeneration=self._generation,
            routedObservations=routedObservations,
        )
        self._decisions.append(decision)
        self._pending = None
        return decision

    def closeSequence(self, sequenceId: SequenceId) -> None:
        if self._pending is not None and self._pending.sequenceId == sequenceId:
            self._pending = replace(self._pending, stale=True)
        self._generation += 1
        self._pending = None
        self._closedSequence = str(sequenceId)

    def summary(self) -> SpeculativeSummary:
        evaluated = len(self._decisions)
        accepted = sum(decision.accepted for decision in self._decisions)
        rollbackReasons = Counter(
            decision.rollbackReason.value
            for decision in self._decisions
            if decision.invalidated
        )
        rollback = evaluated - accepted
        denominator = max(evaluated, 1)
        return SpeculativeSummary(
            evaluatedCount=evaluated,
            acceptedCount=accepted,
            rollbackCount=rollback,
            acceptanceRate=accepted / denominator,
            rollbackRate=rollback / denominator,
            rollbackReasons=dict(sorted(rollbackReasons.items())),
            rollbackTargetMet=(rollback / denominator) <= self._config.maxRollbackRate,
        )


def evaluateSpeculation(
    *,
    config: SpeculativePipelineConfig,
    state: SpeculativeState,
    committedResult: TrackResult,
    formalStateRevision: int,
    currentGeneration: int,
    routedObservations: Sequence[RoutedLocalObservation],
) -> SpeculativeDecision:
    """Validate cached R1(t+1) against the committed result for frame t."""

    def reject(
        reason: RollbackReason,
        *,
        centerGapRad: float | None = None,
        centerGapRatio: float | None = None,
        scaleGap: float | None = None,
        coverage: bool = False,
    ) -> SpeculativeDecision:
        return SpeculativeDecision(
            accepted=False,
            invalidated=True,
            rollbackReason=reason,
            centerGapRad=centerGapRad,
            centerGapRatio=centerGapRatio,
            logScaleGap=scaleGap,
            coverageAfterCorrection=coverage,
            speculativeGeneration=state.generation,
            formalStateRevision=formalStateRevision,
        )

    if not config.enabled:
        return reject(RollbackReason.PIPELINE_DISABLED)
    if state.sequenceId != committedResult.sequenceId:
        return reject(RollbackReason.SEQUENCE_MISMATCH)
    frameAge = int(state.frameIndex) - int(committedResult.frameIndex)
    if frameAge <= 0 or frameAge > config.maxSpeculativeAgeFrames:
        return reject(RollbackReason.FRAME_AGE_MISMATCH)
    if state.stale:
        return reject(RollbackReason.STALE)
    if state.generation != currentGeneration:
        return reject(RollbackReason.GENERATION_MISMATCH)
    if state.sourceStateRevision != formalStateRevision:
        return reject(RollbackReason.REVISION_MISMATCH)
    if committedResult.status is TrackStatus.LOST:
        return reject(RollbackReason.EXPLICIT_LOST)
    if state.directionConfidence < config.minimumDirectionConfidence:
        return reject(RollbackReason.DIRECTION_CONFIDENCE)

    if not routedObservations:
        return reject(RollbackReason.EMPTY_OUTPUT)
    actualKeys = tuple(item.key for item in routedObservations)
    if actualKeys != state.taskKeys:
        return reject(RollbackReason.ROUTING_MISMATCH)
    if not _outputsFinite(routedObservations):
        return reject(RollbackReason.NONFINITE_OUTPUT)

    centerGapRad = _greatCircleDistance(state.directionCenter, committedResult.bfov.center)
    normalizer = max(
        state.horizontalSizeRad,
        state.verticalSizeRad,
        state.motionUncertaintyRad,
        float(np.finfo(np.float64).eps),
    )
    centerGapRatio = centerGapRad / normalizer
    scaleGap = _logScaleGap(
        state.horizontalSizeRad,
        state.verticalSizeRad,
        committedResult.bfov.horizontalFovRad,
        committedResult.bfov.verticalFovRad,
    )
    coverage = any(_viewCovers(view, committedResult.bfov.center) for view in state.views)
    metrics = {
        "centerGapRad": centerGapRad,
        "centerGapRatio": centerGapRatio,
        "scaleGap": scaleGap,
        "coverage": coverage,
    }
    if centerGapRatio > config.centerGapRatio:
        return reject(RollbackReason.CENTER_GAP, **metrics)
    if scaleGap > config.logScaleGap:
        return reject(RollbackReason.SCALE_GAP, **metrics)
    if not coverage:
        return reject(RollbackReason.COVERAGE, **metrics)
    return SpeculativeDecision(
        accepted=True,
        invalidated=False,
        rollbackReason=RollbackReason.NONE,
        centerGapRad=centerGapRad,
        centerGapRatio=centerGapRatio,
        logScaleGap=scaleGap,
        coverageAfterCorrection=True,
        speculativeGeneration=state.generation,
        formalStateRevision=formalStateRevision,
    )


def _greatCircleDistance(first: SphericalPoint, second: SphericalPoint) -> float:
    dot = first.x * second.x + first.y * second.y + first.z * second.z
    return acos(max(-1.0, min(1.0, dot)))


def _logScaleGap(
    speculativeWidth: float,
    speculativeHeight: float,
    committedWidth: float,
    committedHeight: float,
) -> float:
    return max(
        abs(log(speculativeWidth / committedWidth)),
        abs(log(speculativeHeight / committedHeight)),
    )


def _viewCovers(view: ViewSpec, point: SphericalPoint) -> bool:
    forward, right, up = cameraBasis(view.bfov)
    vector = np.asarray((point.x, point.y, point.z), dtype=np.float64)
    depth = float(vector @ forward)
    if depth <= 0.0:
        return False
    horizontal = atan2(float(vector @ right), depth)
    vertical = atan2(float(vector @ up), depth)
    epsilon = 1e-12
    return (
        abs(horizontal) <= view.bfov.horizontalFovRad / 2.0 + epsilon
        and abs(vertical) <= view.bfov.verticalFovRad / 2.0 + epsilon
    )


def _outputsFinite(observations: Sequence[RoutedLocalObservation]) -> bool:
    for item in observations:
        observation = item.observation
        required = (
            observation.bbox.xPx,
            observation.bbox.yPx,
            observation.bbox.widthPx,
            observation.bbox.heightPx,
            observation.modelScore,
            observation.appearanceScore,
            observation.fusedScore,
        )
        optional = (
            observation.presenceProbability,
            observation.qualityProbability,
            observation.predictedIoU,
            observation.appearanceProbability,
        )
        if not all(isfinite(value) for value in required):
            return False
        if any(value is not None and not isfinite(value) for value in optional):
            return False
    return True


__all__ = [
    "RollbackReason",
    "SpeculativeDecision",
    "SpeculativePipeline",
    "SpeculativeState",
    "SpeculativeSummary",
    "evaluateSpeculation",
]
