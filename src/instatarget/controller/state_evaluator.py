"""Evaluate one bounded search round with deterministic two-box fusion."""

from __future__ import annotations

from collections.abc import Sequence
from math import pi

import numpy as np

from instatarget.controller.state_model import (
    EvaluatedCandidate,
    EvaluationReason,
    MeasurementEvidence,
    MotionPrediction,
    StateInstance,
    StateObservation,
    TrackMode,
)
from instatarget.core.config import DecisionGateConfig, EvaluatorConfig, TrackingConfig
from instatarget.core.errors import ProtocolError
from instatarget.core.protocols import SphericalGeometry
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    DepthSummary,
    ProjectedObservation,
    ResultSource,
    SearchPlan,
)
from instatarget.geometry.projection_math import erpPixelToSphericalPoint


class StateEvaluator:
    """Pure cumulative same-frame fusion, ranking, escalation and evidence evaluation."""

    def __init__(
        self,
        gateConfig: DecisionGateConfig,
        trackingConfig: TrackingConfig,
        evaluatorConfig: EvaluatorConfig | None = None,
    ) -> None:
        self._gateConfig = gateConfig
        self._tracking = trackingConfig
        self._config = evaluatorConfig or EvaluatorConfig()

    def evaluate(
        self,
        *,
        state: StateInstance,
        plan: SearchPlan,
        observations: Sequence[ProjectedObservation],
        priorObservations: Sequence[ProjectedObservation] = (),
        prediction: MotionPrediction,
        predictedBfov: BFoV,
        geometry: SphericalGeometry,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> StateObservation:
        self._validate(plan, observations)
        cumulativeObservations = (*priorObservations, *observations)
        viewIds = tuple(item.viewId for item in cumulativeObservations)
        if len(viewIds) != len(set(viewIds)):
            raise ProtocolError("cumulative projected observations must have unique viewIds")
        fusionThreshold = self._fusionThreshold(state.mode, plan.attemptIndex)
        localCandidates = tuple(_localCandidate(item) for item in cumulativeObservations)
        fusedCandidates = _fuseCandidates(
            cumulativeObservations,
            fusionThreshold,
            self._config.fusionSourceMinConfidence,
            geometry,
            frameWidthPx,
            frameHeightPx,
        )
        candidates = (*localCandidates, *fusedCandidates)
        best = max(candidates, key=_candidateRank) if candidates else None
        isFinalAttempt = _isFinalAttempt(state.mode, plan.attemptIndex)
        evidence = self.classifyFinal(best)
        outputEligible = self._outputEligible(
            state.mode,
            plan.attemptIndex,
            isFinalAttempt,
            best,
        )
        escalation = not outputEligible and not isFinalAttempt

        if best is None:
            outputBfov = predictedBfov
            outputBbox = geometry.bfovToBbox(predictedBfov, frameWidthPx, frameHeightPx)
            proposedSource = ResultSource.MOTION_PREDICTED
            searchSeed = prediction.center
            representative = None
        else:
            outputBfov = best.bfov
            outputBbox = best.bbox
            proposedSource = (
                ResultSource.OBSERVED_CONFIRMED
                if evidence in {
                    MeasurementEvidence.RELIABLE_FUSED,
                    MeasurementEvidence.RELIABLE_SINGLE,
                }
                else ResultSource.OBSERVED_WEAK_BLEND
            )
            searchSeed = best.bfov.center
            representative = next(
                (
                    item
                    for item in cumulativeObservations
                    if item.viewId == best.representativeViewId
                ),
                None,
            )

        reasons: list[EvaluationReason] = []
        if best is None:
            reasons.append(EvaluationReason.NO_ELIGIBLE_CLUSTER)
        elif (
            best.fused
            and best.confidence > self._config.successRate
            and not best.sourceConfidencePassed
        ):
            reasons.append(EvaluationReason.SOURCE_CONFIDENCE_BELOW_THRESHOLD)
        elif evidence is MeasurementEvidence.WEAK:
            reasons.append(EvaluationReason.BELOW_UNCERTAIN_THRESHOLD)

        supportCount = len(best.sourceViewIds) if best is not None else 0
        supportScore = min(1.0, supportCount / 2.0)
        agreementScore = (
            best.overlapRate
            if best is not None and best.overlapRate is not None
            else (1.0 if best is not None else 0.0)
        )
        return StateObservation(
            sequenceId=plan.sequenceId,
            frameIndex=plan.frameIndex,
            stateRevision=plan.stateRevision,
            transactionId=plan.transactionId,
            stateId=state.stateId,
            attemptIndex=plan.attemptIndex,
            evaluatedMode=state.mode,
            isFinalAttempt=isFinalAttempt,
            successRate=self._config.successRate,
            fusionThreshold=fusionThreshold,
            overlapThreshold=self._config.overlapThreshold,
            fusionSourceMinConfidence=self._config.fusionSourceMinConfidence,
            bestCandidate=best,
            predictedCenter=prediction.center,
            searchSeedCenter=searchSeed,
            measuredBfov=best.bfov if best is not None else None,
            measuredBbox=best.bbox if best is not None else None,
            measuredCenter=best.bfov.center if best is not None else None,
            proposedOutputBfov=outputBfov,
            proposedOutputBbox=outputBbox,
            proposedResultSource=proposedSource,
            candidateCount=len(cumulativeObservations),
            eligibleCandidateCount=len(candidates),
            clusterCount=len(fusedCandidates),
            sourceViewIds=best.sourceViewIds if best is not None else (),
            representativeViewId=(best.representativeViewId if best is not None else None),
            representativeLocalBox=(
                best.representativeLocalBox if best is not None else None
            ),
            selectedIsFused=best.fused if best is not None else False,
            selectedOverlapRate=best.overlapRate if best is not None else None,
            selectedMinSourceConfidence=(
                best.minSourceConfidence if best is not None else None
            ),
            selectedSourceConfidencePassed=(
                best.sourceConfidencePassed if best is not None else False
            ),
            fusedCandidateCount=len(fusedCandidates),
            outputEligible=outputEligible,
            supportViewCount=supportCount,
            backendScore=representative.fusedScore if representative is not None else 0.0,
            motionScore=representative.motionScore if representative is not None else 0.0,
            scaleScore=representative.scaleScore if representative is not None else 0.0,
            depthConsistencyScore=(
                representative.depthScore
                if representative is not None and representative.depthSummary is not None
                else None
            ),
            supportScore=supportScore,
            agreementScore=float(agreementScore),
            stateScore=best.confidence if best is not None else 0.0,
            evidence=evidence,
            hardGatePassed=best is not None,
            supported=best.fused if best is not None else False,
            escalationRecommended=escalation,
            reacquired=(
                state.mode in {TrackMode.LOST, TrackMode.RECOVERING}
                and evidence is MeasurementEvidence.RELIABLE_FUSED
            ),
            depthSummary=best.depthSummary if best is not None else None,
            rejectionReasons=tuple(reasons),
        )

    def classifyFinal(
        self,
        candidate: EvaluatedCandidate | None,
    ) -> MeasurementEvidence:
        if candidate is None:
            return MeasurementEvidence.MISSING
        if (
            candidate.fused
            and candidate.overlapRate is not None
            and candidate.overlapRate > self._config.overlapThreshold
            and candidate.confidence > self._config.successRate
            and candidate.sourceConfidencePassed
        ):
            return MeasurementEvidence.RELIABLE_FUSED
        if not candidate.fused and candidate.confidence > self._config.successRate:
            return MeasurementEvidence.RELIABLE_SINGLE
        return MeasurementEvidence.WEAK

    def _fusionThreshold(self, mode: TrackMode, attemptIndex: int) -> float:
        if attemptIndex == 0 and mode is not TrackMode.LOST:
            return self._config.firstRoundFusionOverlap
        return self._config.overlapThreshold

    def _outputEligible(
        self,
        mode: TrackMode,
        attemptIndex: int,
        isFinalAttempt: bool,
        best: EvaluatedCandidate | None,
    ) -> bool:
        if isFinalAttempt:
            return True
        if attemptIndex == 0:
            return self.classifyFinal(best) is MeasurementEvidence.RELIABLE_FUSED
        if mode in {TrackMode.UNCERTAIN, TrackMode.RECOVERING} and attemptIndex == 1:
            return best is not None and best.confidence > self._config.successRate
        return False

    def _validate(
        self,
        plan: SearchPlan,
        observations: Sequence[ProjectedObservation],
    ) -> None:
        expected = tuple(view.viewId for view in plan.views)
        actual = tuple(item.viewId for item in observations)
        if len(actual) != len(set(actual)):
            raise ProtocolError("projected observations must have unique viewIds")
        if any(item not in expected for item in actual):
            raise ProtocolError("projected observation contains an unknown viewId")
        if actual and actual != tuple(item for item in expected if item in set(actual)):
            raise ProtocolError("projected observations must preserve requested view order")


def _isFinalAttempt(mode: TrackMode, attemptIndex: int) -> bool:
    if mode in {TrackMode.TRACKING, TrackMode.LOST}:
        return attemptIndex >= 1
    if mode in {TrackMode.UNCERTAIN, TrackMode.RECOVERING}:
        return attemptIndex >= 2
    return True


def _localCandidate(observation: ProjectedObservation) -> EvaluatedCandidate:
    return EvaluatedCandidate(
        bfov=observation.bfov,
        bbox=observation.bbox,
        confidence=float(np.clip(observation.fusedScore, 0.0, 1.0)),
        sourceViewIds=(observation.viewId,),
        fused=False,
        overlapRate=None,
        minSourceConfidence=None,
        sourceConfidencePassed=True,
        representativeViewId=observation.viewId,
        representativeLocalBox=observation.localBox,
        depthSummary=observation.depthSummary,
    )


def _fuseCandidates(
    observations: Sequence[ProjectedObservation],
    threshold: float,
    sourceMinConfidence: float,
    geometry: SphericalGeometry,
    frameWidthPx: int,
    frameHeightPx: int,
) -> tuple[EvaluatedCandidate, ...]:
    edges: list[tuple[float, float, int, int, ProjectedObservation, ProjectedObservation]] = []
    for firstIndex, first in enumerate(observations):
        for second in observations[firstIndex + 1 :]:
            if first.viewId == second.viewId:
                continue
            overlap = _overlapRate(first.bbox, second.bbox, frameWidthPx)
            if overlap <= threshold:
                continue
            edges.append(
                (
                    overlap,
                    max(first.fusedScore, second.fusedScore),
                    min(first.viewId, second.viewId),
                    max(first.viewId, second.viewId),
                    first,
                    second,
                )
            )
    edges.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

    used: set[int] = set()
    fused: list[EvaluatedCandidate] = []
    for overlap, _, _, _, first, second in edges:
        if first.viewId in used or second.viewId in used:
            continue
        used.update((first.viewId, second.viewId))
        bbox = _intersectionBox(first.bbox, second.bbox, frameWidthPx, frameHeightPx)
        if bbox is None:
            continue
        bfov = _intersectionBfov(bbox, geometry, frameWidthPx, frameHeightPx)
        confidence = 1.0 - (
            (2.0 - second.fusedScore - first.fusedScore) * (1.0 - overlap) / 2.0
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))
        representative = max(
            (first, second),
            key=lambda item: (item.fusedScore, -item.viewId),
        )
        minSource = min(first.fusedScore, second.fusedScore)
        fused.append(
            EvaluatedCandidate(
                bfov=bfov,
                bbox=bbox,
                confidence=confidence,
                sourceViewIds=tuple(sorted((first.viewId, second.viewId))),
                fused=True,
                overlapRate=overlap,
                minSourceConfidence=minSource,
                sourceConfidencePassed=minSource >= sourceMinConfidence,
                representativeViewId=representative.viewId,
                representativeLocalBox=representative.localBox,
                depthSummary=_mergeDepth(first, second),
            )
        )
    return tuple(fused)


def _intersectionBfov(
    bbox: BBoxXYWH,
    geometry: SphericalGeometry,
    frameWidthPx: int,
    frameHeightPx: int,
) -> BFoV:
    horizontalSpan = 2.0 * pi * min(bbox.widthPx, frameWidthPx) / frameWidthPx
    if horizontalSpan < pi:
        return geometry.bboxToBfov(bbox, frameWidthPx, frameHeightPx)

    # A perspective BFoV cannot span 180 degrees. Keep the exact ERP intersection bbox
    # and attach an equirectangular envelope for this unusual wide intersection.
    center = erpPixelToSphericalPoint(
        (bbox.xPx + bbox.widthPx / 2.0) % frameWidthPx,
        min(float(frameHeightPx), max(0.0, bbox.yPx + bbox.heightPx / 2.0)),
        frameWidthPx,
        frameHeightPx,
    )
    return BFoV(
        center=center,
        horizontalFovRad=min(horizontalSpan, float(np.nextafter(2.0 * pi, 0.0))),
        verticalFovRad=min(
            pi * min(bbox.heightPx, frameHeightPx) / frameHeightPx,
            float(np.nextafter(pi, 0.0)),
        ),
    )


def _candidateRank(candidate: EvaluatedCandidate) -> tuple[float, bool, int]:
    return candidate.confidence, candidate.fused, -candidate.representativeViewId


def _overlapRate(first: BBoxXYWH, second: BBoxXYWH, frameWidthPx: int) -> float:
    yStart = max(first.yPx, second.yPx)
    yEnd = min(first.yPx + first.heightPx, second.yPx + second.heightPx)
    if yEnd <= yStart:
        return 0.0
    horizontal = 0.0
    for firstStart, firstEnd in _xSegments(first, frameWidthPx):
        for secondStart, secondEnd in _xSegments(second, frameWidthPx):
            horizontal += max(0.0, min(firstEnd, secondEnd) - max(firstStart, secondStart))
    intersection = horizontal * (yEnd - yStart)
    smallerArea = min(first.widthPx * first.heightPx, second.widthPx * second.heightPx)
    if smallerArea <= 0.0:
        return 0.0
    return float(np.clip(intersection / smallerArea, 0.0, 1.0))


def _xSegments(box: BBoxXYWH, frameWidthPx: int) -> tuple[tuple[float, float], ...]:
    width = min(float(frameWidthPx), box.widthPx)
    start = box.xPx % frameWidthPx
    end = start + width
    if end <= frameWidthPx:
        return ((start, end),)
    return ((start, float(frameWidthPx)), (0.0, end - frameWidthPx))


def _intersectionBox(
    first: BBoxXYWH,
    second: BBoxXYWH,
    frameWidthPx: int,
    frameHeightPx: int,
) -> BBoxXYWH | None:
    intersections = sorted(
        (
            max(firstStart, secondStart),
            min(firstEnd, secondEnd),
        )
        for firstStart, firstEnd in _xSegments(first, frameWidthPx)
        for secondStart, secondEnd in _xSegments(second, frameWidthPx)
        if min(firstEnd, secondEnd) > max(firstStart, secondStart)
    )
    if not intersections:
        return None

    # A circular intersection can be split at the ERP seam. Merge only seam-adjacent
    # pieces; if two large arcs intersect in disconnected pieces, retain the largest
    # connected component because BBoxXYWH can represent one region only.
    merged: list[list[float]] = []
    for start, end in intersections:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if len(merged) > 1 and merged[0][0] == 0.0 and merged[-1][1] == frameWidthPx:
        seamStart = merged[-1][0]
        seamWidth = merged[0][1] + frameWidthPx - seamStart
        horizontalStart, horizontalWidth = seamStart, seamWidth
    else:
        component = max(merged, key=lambda item: item[1] - item[0])
        horizontalStart = component[0]
        horizontalWidth = component[1] - component[0]
    yStart = max(0.0, first.yPx, second.yPx)
    yEnd = min(float(frameHeightPx), first.yPx + first.heightPx, second.yPx + second.heightPx)
    if yEnd <= yStart or horizontalWidth <= 0.0:
        return None
    return BBoxXYWH(
        xPx=horizontalStart,
        yPx=yStart,
        widthPx=horizontalWidth,
        heightPx=yEnd - yStart,
    )


def _mergeDepth(
    first: ProjectedObservation,
    second: ProjectedObservation,
) -> DepthSummary | None:
    if first.depthSummary is None:
        return second.depthSummary
    if second.depthSummary is None:
        return first.depthSummary
    firstWeight = max(first.fusedScore, 1e-6)
    secondWeight = max(second.fusedScore, 1e-6)
    total = firstWeight + secondWeight
    return DepthSummary(
        medianDepth=(
            first.depthSummary.medianDepth * firstWeight
            + second.depthSummary.medianDepth * secondWeight
        )
        / total,
        meanDepth=(
            first.depthSummary.meanDepth * firstWeight
            + second.depthSummary.meanDepth * secondWeight
        )
        / total,
        validRatio=(
            first.depthSummary.validRatio * firstWeight
            + second.depthSummary.validRatio * secondWeight
        )
        / total,
        minDepth=min(first.depthSummary.minDepth, second.depthSummary.minDepth),
        maxDepth=max(first.depthSummary.maxDepth, second.depthSummary.maxDepth),
        confidence=(
            first.depthSummary.confidence * firstWeight
            + second.depthSummary.confidence * secondWeight
        )
        / total,
    )


__all__ = ["StateEvaluator"]
