"""Evaluate a search attempt and its transaction-scoped candidate pool."""

from __future__ import annotations

from collections.abc import Sequence

from instatarget.controller.fusor import FUSION_OVERLAP_RATE, FusionBoxMode, Fusor
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
from instatarget.core.types import BFoV, ProjectedObservation, ResultSource, SearchPlan


class StateEvaluator:
    """Select one best candidate from all observations eligible at this attempt.

    The evaluator owns candidate geometry and measurement eligibility.  The state machine
    receives only the resulting StateScore for next-state selection.
    """

    def __init__(
        self,
        gateConfig: DecisionGateConfig,
        trackingConfig: TrackingConfig,
        evaluatorConfig: EvaluatorConfig | None = None,
    ) -> None:
        del gateConfig
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
        if priorObservations and plan.attemptIndex == 0:
            raise ProtocolError("first attempt cannot contain prior observations")
        candidatePool = _combineObservations(priorObservations, observations)
        best = Fusor(
            geometry,
            overlapRate=FUSION_OVERLAP_RATE,
            sourceMinConfidence=self._config.fusionSourceMinConfidence,
            boxMode=(
                FusionBoxMode.MIN_UNION
                if state.mode is TrackMode.TRACKING
                else FusionBoxMode.MAX_INTERSECTION
            ),
        ).fuse(
            candidatePool,
            frameWidthPx=frameWidthPx,
            frameHeightPx=frameHeightPx,
        )
        isFinalAttempt = not (
            state.mode in {TrackMode.TRACKING, TrackMode.UNCERTAIN}
            and plan.attemptIndex == 0
        )
        accepted = _measurementAccepted(best, self._tracking.candidateMinScore)
        evidence = _evidence(best, accepted)
        outputBfov = best.bfov if best is not None else predictedBfov
        outputBbox = (
            best.bbox
            if best is not None
            else geometry.bfovToBbox(predictedBfov, frameWidthPx, frameHeightPx)
        )
        representative = next(
            (
                item
                for item in candidatePool
                if best is not None and item.viewId == best.representativeViewId
            ),
            None,
        )
        stateScore = best.confidence if best is not None else 0.0
        reasons: list[EvaluationReason] = []
        if best is None:
            reasons.append(EvaluationReason.NO_ELIGIBLE_CLUSTER)
        elif not accepted:
            reasons.append(EvaluationReason.BELOW_UNCERTAIN_THRESHOLD)
        return StateObservation(
            sequenceId=plan.sequenceId,
            frameIndex=plan.frameIndex,
            stateRevision=plan.stateRevision,
            transactionId=plan.transactionId,
            stateId=state.stateId,
            attemptIndex=plan.attemptIndex,
            evaluatedMode=state.mode,
            isFinalAttempt=isFinalAttempt,
            appearanceOnlyScoring=False,
            successRate=self._config.successRate,
            fusionThreshold=FUSION_OVERLAP_RATE,
            overlapThreshold=FUSION_OVERLAP_RATE,
            fusionSourceMinConfidence=self._config.fusionSourceMinConfidence,
            bestCandidate=best,
            predictedCenter=prediction.center,
            searchSeedCenter=best.bfov.center if best is not None else prediction.center,
            measuredBfov=best.bfov if best is not None else None,
            measuredBbox=best.bbox if best is not None else None,
            measuredCenter=best.bfov.center if best is not None else None,
            proposedOutputBfov=outputBfov,
            proposedOutputBbox=outputBbox,
            proposedResultSource=(
                ResultSource.OBSERVED_CONFIRMED
                if accepted
                else ResultSource.OBSERVED_WEAK_BLEND
                if best is not None
                else ResultSource.MOTION_PREDICTED
            ),
            candidateCount=len(candidatePool),
            eligibleCandidateCount=1 if best is not None else 0,
            clusterCount=1 if best is not None and best.fused else 0,
            sourceViewIds=best.sourceViewIds if best is not None else (),
            representativeViewId=best.representativeViewId if best is not None else None,
            representativeLocalBox=best.representativeLocalBox if best is not None else None,
            selectedIsFused=best.fused if best is not None else False,
            selectedOverlapRate=best.overlapRate if best is not None else None,
            selectedMinSourceConfidence=best.minSourceConfidence if best is not None else None,
            selectedSourceConfidencePassed=(
                best.sourceConfidencePassed if best is not None else False
            ),
            fusedCandidateCount=1 if best is not None and best.fused else 0,
            outputEligible=best is not None,
            supportViewCount=len(best.sourceViewIds) if best is not None else 0,
            backendScore=_backendScore(representative),
            motionScore=representative.motionScore if representative is not None else 0.0,
            scaleScore=representative.scaleScore if representative is not None else 0.0,
            depthConsistencyScore=representative.depthScore if representative is not None else None,
            supportScore=min(1.0, len(best.sourceViewIds) / 2.0) if best is not None else 0.0,
            agreementScore=(
                best.overlapRate
                if best is not None and best.overlapRate is not None
                else 0.0
            ),
            stateScore=stateScore,
            evidence=evidence,
            hardGatePassed=accepted,
            supported=best.fused if best is not None else False,
            escalationRecommended=(
                state.mode in {TrackMode.TRACKING, TrackMode.UNCERTAIN}
                and plan.attemptIndex == 0
            ),
            reacquired=state.mode is TrackMode.LOST and accepted,
            depthSummary=best.depthSummary if best is not None else None,
            rejectionReasons=tuple(reasons),
            rawMotionScore=representative.rawMotionScore if representative is not None else None,
            motionProbability=(
                representative.motionProbability if representative is not None else None
            ),
            motionReliability=(
                representative.motionReliability
                if representative is not None
                else prediction.reliability
            ),
            motionSampleCount=prediction.sampleCount,
            motionDegradedReasons=prediction.degradedReasons,
            measurementAccepted=accepted,
        )

    def classifyFinal(
        self,
        candidate: EvaluatedCandidate | None,
        successRate: float | None = None,
        overlapThreshold: float | None = None,
    ) -> MeasurementEvidence:
        del successRate, overlapThreshold
        if candidate is None:
            return MeasurementEvidence.MISSING
        if candidate.fused:
            return (
                MeasurementEvidence.RELIABLE_FUSED
                if candidate.sourceConfidencePassed
                else MeasurementEvidence.WEAK
            )
        return MeasurementEvidence.RELIABLE_SINGLE

    def _validate(self, plan: SearchPlan, observations: Sequence[ProjectedObservation]) -> None:
        expected = tuple(view.viewId for view in plan.views)
        actual = tuple(item.viewId for item in observations)
        if len(actual) != len(set(actual)):
            raise ProtocolError("projected observations must have unique viewIds")
        if any(item not in expected for item in actual):
            raise ProtocolError("projected observation contains an unknown viewId")
        if actual and actual != tuple(item for item in expected if item in set(actual)):
            raise ProtocolError("projected observations must preserve requested view order")


def _measurementAccepted(candidate: EvaluatedCandidate | None, minimumScore: float) -> bool:
    if candidate is None or candidate.confidence < minimumScore:
        return False
    return not candidate.fused or candidate.sourceConfidencePassed


def _combineObservations(
    priorObservations: Sequence[ProjectedObservation],
    observations: Sequence[ProjectedObservation],
) -> tuple[ProjectedObservation, ...]:
    combined = tuple(priorObservations) + tuple(observations)
    viewIds = tuple(item.viewId for item in combined)
    if len(viewIds) != len(set(viewIds)):
        raise ProtocolError("cross-round candidate pool must have unique viewIds")
    return combined


def _evidence(candidate: EvaluatedCandidate | None, accepted: bool) -> MeasurementEvidence:
    if candidate is None:
        return MeasurementEvidence.MISSING
    if not accepted:
        return MeasurementEvidence.WEAK
    return (
        MeasurementEvidence.RELIABLE_FUSED
        if candidate.fused
        else MeasurementEvidence.RELIABLE_SINGLE
    )


def _backendScore(observation: ProjectedObservation | None) -> float:
    if observation is None:
        return 0.0
    return float(
        observation.backendFusedScore
        if observation.backendFusedScore is not None
        else observation.fusedScore
    )


__all__ = ["StateEvaluator"]
