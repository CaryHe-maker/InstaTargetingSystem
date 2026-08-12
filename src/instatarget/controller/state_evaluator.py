"""Turn all projected observations from one attempt into one V2 state observation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from instatarget.controller.decision_gate import DecisionGate, FrameAggregate
from instatarget.controller.state_model import (
    EvaluationReason,
    EvidenceLevel,
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
    """Pure per-attempt candidate evaluation.

    Disjoint observations remain separate hypotheses.  Only the selected compatible cluster is
    fused, and its dimensions use robust medians rather than a union envelope.
    """

    def __init__(
        self,
        gateConfig: DecisionGateConfig,
        trackingConfig: TrackingConfig,
        evaluatorConfig: EvaluatorConfig | None = None,
    ) -> None:
        self._tracking = trackingConfig
        self._gate = DecisionGate(gateConfig, trackingConfig)
        self._config = evaluatorConfig or EvaluatorConfig()

    def evaluate(
        self,
        *,
        state: StateInstance,
        plan: SearchPlan,
        observations: Sequence[ProjectedObservation],
        prediction: MotionPrediction,
        predictedBfov: BFoV,
        geometry: SphericalGeometry,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> StateObservation:
        self._validate(plan, observations)
        scored = tuple(self._gate.score(item) for item in observations)
        eligible = tuple(
            item for item in scored if item.decisionScore >= self._tracking.candidateMinScore
        )
        aggregate = self._gate.aggregate(
            observations,
            geometry,
            frameWidthPx,
            frameHeightPx,
        )
        evidence, reacquired = self._classify(state.mode, aggregate)
        representative = None
        if aggregate is not None:
            representative = next(
                (item for item in observations if item.viewId == aggregate.representativeViewId),
                None,
            )
        supported = aggregate.supported if aggregate is not None else False
        if aggregate is not None and evidence in {
            EvidenceLevel.CONFIRMED,
            EvidenceLevel.REACQUIRED,
        }:
            outputBfov = aggregate.bfov
            outputBbox = aggregate.bbox
            proposedSource = (
                ResultSource.OBSERVED_REACQUIRED if reacquired else ResultSource.OBSERVED_CONFIRMED
            )
        else:
            # Weak and disjoint candidates guide the next search only.  They do not enlarge the
            # emitted box or form an ERP union that would lower IoU.
            outputBfov = predictedBfov
            outputBbox = geometry.bfovToBbox(predictedBfov, frameWidthPx, frameHeightPx)
            proposedSource = ResultSource.MOTION_PREDICTED
        searchSeed = aggregate.bfov.center if aggregate is not None else prediction.center
        stateScore = aggregate.decisionScore if aggregate is not None else 0.0
        supportCount = len(aggregate.sourceViewIds) if aggregate is not None else 0
        requiredViews = (
            self._config.minReacquireViews
            if state.mode in {TrackMode.RECOVERING, TrackMode.LOST}
            else self._tracking.minViewsForCommit
        )
        supportScore = min(1.0, supportCount / max(1, requiredViews))
        agreementScore = aggregate.agreementScore if aggregate is not None else 0.0
        baseWeight = 1.0 - self._config.supportWeight - self._config.agreementWeight
        stateScore = stateScore * (
            baseWeight
            + self._config.supportWeight * supportScore
            + self._config.agreementWeight * agreementScore
        )
        escalation = evidence in {EvidenceLevel.WEAK, EvidenceLevel.REJECTED}
        reasons: list[EvaluationReason] = []
        if aggregate is None:
            reasons.append(EvaluationReason.NO_ELIGIBLE_CLUSTER)
        elif not supported:
            reasons.append(EvaluationReason.INSUFFICIENT_VIEW_SUPPORT)
        if evidence is EvidenceLevel.REJECTED:
            reasons.append(EvaluationReason.BELOW_UNCERTAIN_THRESHOLD)
        return StateObservation(
            sequenceId=plan.sequenceId,
            frameIndex=plan.frameIndex,
            stateRevision=plan.stateRevision,
            transactionId=plan.transactionId,
            stateId=state.stateId,
            attemptIndex=plan.attemptIndex,
            evaluatedMode=state.mode,
            predictedCenter=prediction.center,
            searchSeedCenter=searchSeed,
            measuredBfov=aggregate.bfov if aggregate is not None else None,
            measuredBbox=aggregate.bbox if aggregate is not None else None,
            measuredCenter=aggregate.bfov.center if aggregate is not None else None,
            proposedOutputBfov=outputBfov,
            proposedOutputBbox=outputBbox,
            proposedResultSource=proposedSource,
            candidateCount=len(observations),
            eligibleCandidateCount=len(eligible),
            clusterCount=aggregate.clusterCount if aggregate is not None else 0,
            sourceViewIds=aggregate.sourceViewIds if aggregate is not None else (),
            representativeViewId=aggregate.representativeViewId if aggregate is not None else None,
            representativeLocalBox=aggregate.localBox if aggregate is not None else None,
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
            agreementScore=agreementScore,
            stateScore=float(np.clip(stateScore, 0.0, 1.0)),
            evidence=evidence,
            hardGatePassed=aggregate is not None,
            supported=supported,
            escalationRecommended=escalation,
            reacquired=reacquired,
            depthSummary=aggregate.depthSummary if aggregate is not None else None,
            rejectionReasons=tuple(reasons),
        )

    def _classify(
        self,
        mode: TrackMode,
        aggregate: FrameAggregate | None,
    ) -> tuple[EvidenceLevel, bool]:
        if aggregate is None:
            return EvidenceLevel.REJECTED, False
        score = aggregate.decisionScore
        recoveryMode = mode in {TrackMode.RECOVERING, TrackMode.LOST}
        enoughViews = len(aggregate.sourceViewIds) >= self._config.minReacquireViews
        if (
            recoveryMode
            and aggregate.supported
            and enoughViews
            and score >= self._tracking.recoverAcceptThreshold
        ):
            return EvidenceLevel.REACQUIRED, True
        if aggregate.supported and score >= self._tracking.acceptThreshold:
            return EvidenceLevel.CONFIRMED, False
        if score >= self._tracking.uncertainThreshold:
            return EvidenceLevel.WEAK, False
        return EvidenceLevel.REJECTED, False

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
        # Empty observations are a valid model miss.  Non-empty normal responses must retain the
        # request order; explicit partial-inference messages should be introduced separately.
        if actual and actual != tuple(item for item in expected if item in set(actual)):
            raise ProtocolError("projected observations must preserve requested view order")


__all__ = ["StateEvaluator"]
