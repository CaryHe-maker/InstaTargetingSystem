"""V2 transactional DTC facade for multi-view spherical tracking."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from math import asin, atan2, tan
from typing import TYPE_CHECKING

from instatarget.controller.motion_estimator import SphericalMotionEstimator
from instatarget.controller.recovery_planner import PlannedView, RecoveryPlanner
from instatarget.controller.state_evaluator import StateEvaluator
from instatarget.controller.state_machine import TrackStateMachine
from instatarget.controller.state_model import (
    AttemptKind,
    AttemptRecord,
    FrameTransaction,
    MotionPrediction,
    RecoveryMemory,
    StateInstance,
    StateObservation,
    TrackMode,
    TransitionDecision,
    TransitionReason,
)
from instatarget.controller.template_policy import TemplateDecision, TemplatePolicy
from instatarget.core.config import (
    AppConfig,
    DecisionGateConfig,
    EvaluatorConfig,
    GeometryConfig,
    MotionConfig,
    RecoveryConfig,
    TrackingConfig,
)
from instatarget.core.errors import ProtocolError
from instatarget.core.protocols import (
    FrameCommitted,
    MoreViewsRequired,
    SphericalGeometry,
    TrackController,
)
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    DepthSummary,
    FrameIndex,
    FramePacket,
    InitializationPlan,
    MotionState3D,
    ProjectedObservation,
    ResultSource,
    SearchPlan,
    TemplateCommand,
    TemplateCommandKind,
    TrackResult,
    TrackStatus,
    ViewSpec,
)
from instatarget.geometry.projection_math import fovToFocalLengthPx, makeSphericalPoint

if TYPE_CHECKING:
    from instatarget.core.protocols import MotionEstimator


@dataclass(frozen=True, slots=True)
class _PlannedAttempt:
    frame: FramePacket
    plan: SearchPlan
    prediction: MotionPrediction
    predictedBfov: BFoV
    state: StateInstance
    viewsById: dict[int, PlannedView]


class DepthAwareTrackController(TrackController):
    """Single-writer controller with bounded same-frame escalation and atomic commit."""

    def __init__(
        self,
        geometry: SphericalGeometry,
        config: AppConfig | None = None,
        *,
        geometryConfig: GeometryConfig | None = None,
        trackingConfig: TrackingConfig | None = None,
        recoveryConfig: RecoveryConfig | None = None,
        decisionGateConfig: DecisionGateConfig | None = None,
        evaluatorConfig: EvaluatorConfig | None = None,
        motionConfig: MotionConfig | None = None,
        motionEstimator: MotionEstimator | None = None,
        enableLostState: bool = False,
    ) -> None:
        if config is not None:
            geometryConfig = config.geometry
            trackingConfig = config.tracking
            recoveryConfig = config.recovery
            decisionGateConfig = config.decisionGate
            evaluatorConfig = config.evaluator
            motionConfig = config.motion
        if geometryConfig is None or trackingConfig is None or recoveryConfig is None:
            raise ValueError("geometryConfig, trackingConfig and recoveryConfig are required")
        decisionGateConfig = decisionGateConfig or DecisionGateConfig(0.25, 0.15)
        evaluatorConfig = evaluatorConfig or EvaluatorConfig()
        motionConfig = motionConfig or MotionConfig()
        self._geometry = geometry
        self._geometryConfig = geometryConfig
        self._trackingConfig = trackingConfig
        self._recoveryConfig = recoveryConfig
        self._motionMinSamples = motionConfig.minSamplesForVelocity
        self._motion: MotionEstimator = motionEstimator or SphericalMotionEstimator(
            windowLength=trackingConfig.windowLength,
            maxPredictionHorizon=trackingConfig.maxPredictionHorizon,
            minSamplesForVelocity=motionConfig.minSamplesForVelocity,
            maxTangentSpanRad=motionConfig.maxTangentSpanRad,
            huberDeltaRad=motionConfig.huberDeltaRad,
            processNoiseRadPerSec=motionConfig.processNoiseRadPerSec,
            maxAngularSpeedRadPerSec=motionConfig.maxAngularSpeedRadPerSec,
            maxLogScaleRatePerSec=motionConfig.maxLogScaleRatePerSec,
        )
        self._evaluator = StateEvaluator(decisionGateConfig, trackingConfig, evaluatorConfig)
        self._planner = RecoveryPlanner(geometryConfig, trackingConfig, recoveryConfig)
        self._stateMachine = TrackStateMachine(
            trackingConfig,
            enableLostState=enableLostState,
        )
        self._templatePolicy = TemplatePolicy(trackingConfig)
        self._recovery = RecoveryMemory()

        self._initialized = False
        self._sequenceId: str | None = None
        self._lastFrameIndex = -1
        self._stateRevision = -1
        self._backendRevision = 0
        self._stateId = 0
        self._transactionId = 0
        self._mode = TrackMode.INIT
        self._entryReason = TransitionReason.INITIALIZED
        self._modeAgeFrames = 0
        self._weakFrames = 0
        self._recoveryFrames = 0
        self._stableFrames = 0
        self._reacquireCooldown = 0
        self._lastFrame: FramePacket | None = None
        self._initialBox: BBoxXYWH | None = None
        self._currentBox: BBoxXYWH | None = None
        self._currentBfov: BFoV | None = None
        self._currentDepth: DepthSummary | None = None
        self._pendingTemplate = TemplateDecision(TemplateCommandKind.KEEP)
        self._planned: _PlannedAttempt | None = None
        self._transaction: FrameTransaction | None = None
        self._initialPlan: InitializationPlan | None = None
        self._lastStateObservation: StateObservation | None = None
        self._lastTransition: TransitionDecision | None = None

    @property
    def status(self) -> TrackStatus | None:
        if not self._initialized:
            return None
        return _publicStatus(self._mode)

    @property
    def stateRevision(self) -> int:
        return max(0, self._stateRevision)

    @property
    def lastStateObservation(self) -> StateObservation | None:
        return self._lastStateObservation

    @property
    def lastTransition(self) -> TransitionDecision | None:
        return self._lastTransition

    def buildInitialization(self, frame: FramePacket, initialBox: BBoxXYWH) -> InitializationPlan:
        if self._initialized or self._initialPlan is not None:
            raise ProtocolError("controller is already initialized or has a pending initialization")
        if int(frame.frameIndex) != 0:
            raise ProtocolError("initialization must use frameIndex 0")
        objectBfov = self._geometry.bboxToBfov(initialBox, frame.rgb.shape[1], frame.rgb.shape[0])
        templateBfov = BFoV(
            center=objectBfov.center,
            horizontalFovRad=self._geometryConfig.maxFovRad,
            verticalFovRad=self._geometryConfig.maxFovRad,
        )
        plan = InitializationPlan(
            sequenceId=frame.sequenceId,
            frameIndex=FrameIndex(0),
            stateRevision=0,
            templateView=self._makeViewSpec(0, templateBfov),
            templateBox=_boxForBfov(
                self._geometryConfig.viewWidthPx,
                self._geometryConfig.viewHeightPx,
                templateBfov,
                objectBfov,
            ),
        )
        self._initialPlan = plan
        self._lastFrame = frame
        self._sequenceId = str(frame.sequenceId)
        self._initialBox = initialBox
        self._currentBox = initialBox
        self._currentBfov = objectBfov
        return plan

    def commitInitialization(
        self,
        plan: InitializationPlan,
        depthSummary: DepthSummary | None,
    ) -> TrackResult:
        if self._initialPlan != plan:
            raise ProtocolError("initialization response does not match the pending plan")
        if self._lastFrame is None or self._initialBox is None or self._currentBfov is None:
            raise ProtocolError("initialization frame state is incomplete")
        if hasattr(self._motion, "resetFromMeasurement"):
            self._motion.resetFromMeasurement(  # type: ignore[attr-defined]
                self._currentBfov.center,
                depthSummary,
                self._lastFrame.timestampNs,
                0,
                1.0,
                self._currentBfov.horizontalFovRad,
                self._currentBfov.verticalFovRad,
            )
        else:
            self._motion.initialize(
                self._currentBfov.center,
                depthSummary,
                self._lastFrame.timestampNs,
            )
        self._stateMachine.initialize()
        self._initialized = True
        self._initialPlan = None
        self._stateRevision = 0
        self._lastFrameIndex = 0
        self._mode = TrackMode.TRACKING
        self._currentDepth = depthSummary
        return TrackResult(
            sequenceId=self._lastFrame.sequenceId,
            frameIndex=FrameIndex(0),
            bbox=self._initialBox,
            bfov=self._currentBfov,
            confidence=1.0,
            status=TrackStatus.TRACKING,
            valid=True,
            depthSummary=depthSummary,
            resultSource=ResultSource.INITIAL,
        )

    def beginFrame(self, frame: FramePacket) -> SearchPlan:
        return self.plan(frame)

    def plan(self, frame: FramePacket) -> SearchPlan:
        self._requireInitialized()
        if self._planned is not None or self._transaction is not None:
            raise ProtocolError("a frame transaction is already awaiting update")
        self._requireFrameOrder(frame)
        if self._currentBox is None or self._currentBfov is None:
            raise ProtocolError("controller target state is incomplete")
        prediction = self._predictDetailed(frame)
        predictedBfov = self._planner.contextBfov(
            prediction.center,
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            self._initialBox or self._currentBox,
            self._currentBox,
            prediction.angularUncertaintyRad,
        )
        self._stateId += 1
        state = StateInstance(
            stateId=self._stateId,
            sequenceId=frame.sequenceId,
            frameIndex=frame.frameIndex,
            stateRevision=self._stateRevision + 1,
            mode=self._mode,
            enteredFrom=self._mode,
            entryReason=self._entryReason,
            prediction=prediction,
            searchSeedCenter=prediction.center,
            recoveryEpochId=self._recovery.epochId,
            modeAgeFrames=self._modeAgeFrames,
            stableStreak=self._stableFrames,
            weakStreak=self._weakFrames,
            missStreak=self._recoveryFrames,
        )
        self._transactionId += 1
        self._transaction = FrameTransaction(
            transactionId=self._transactionId,
            frame=frame,
            state=state,
            startingMode=self._mode,
            remainingViews=self._trackingConfig.maxViewsPerFrameTotal,
            recoveryMemory=deepcopy(self._recovery),
        )
        return self._buildAttempt(
            frame=frame,
            state=state,
            prediction=prediction,
            predictedBfov=predictedBfov,
            attemptIndex=0,
            searchSeed=prediction.center,
            viewIdStart=0,
        )

    def consume(
        self,
        plan: SearchPlan,
        observations: Sequence[ProjectedObservation],
    ) -> MoreViewsRequired | FrameCommitted:
        return self._consume(plan, observations, allowEscalation=True)

    def update(
        self,
        plan: SearchPlan,
        observations: Sequence[ProjectedObservation],
    ) -> TrackResult:
        """Compatibility path: commit one attempt and never request more backend work."""
        step = self._consume(
            plan,
            observations,
            allowEscalation=False,
        )
        if isinstance(step, MoreViewsRequired):
            raise ProtocolError("legacy update path cannot return a second search plan")
        return step.result

    def _consume(
        self,
        plan: SearchPlan,
        observations: Sequence[ProjectedObservation],
        *,
        allowEscalation: bool,
    ) -> MoreViewsRequired | FrameCommitted:
        self._requireInitialized()
        planned = self._planned
        transaction = self._transaction
        if planned is None or transaction is None or planned.plan != plan:
            raise ProtocolError("search response does not match the pending attempt")
        if (
            plan.transactionId != transaction.transactionId
            or plan.attemptIndex != transaction.attemptIndex
        ):
            raise ProtocolError("search response transaction identity mismatch")
        expectedBackendRevision = self._backendRevision + 1
        if plan.templateCommand.expectedRevision != expectedBackendRevision:
            raise ProtocolError(
                "backend template revision mismatch: "
                f"expected={expectedBackendRevision}, "
                f"actual={plan.templateCommand.expectedRevision}"
            )
        self._backendRevision = plan.templateCommand.expectedRevision
        priorObservations = (
            tuple(
                observation
                for attempt in transaction.attempts
                for observation in attempt.observations
            )
            if plan.attemptIndex > 0
            and planned.state.mode in {TrackMode.TRACKING, TrackMode.UNCERTAIN}
            else ()
        )
        evaluation = self._evaluator.evaluate(
            state=planned.state,
            plan=plan,
            observations=observations,
            priorObservations=priorObservations,
            prediction=planned.prediction,
            predictedBfov=planned.predictedBfov,
            geometry=self._geometry,
            frameWidthPx=planned.frame.rgb.shape[1],
            frameHeightPx=planned.frame.rgb.shape[0],
        )
        thresholds = self._stateMachine.scoreGroup.thresholds()
        if thresholds is not None:
            evaluation = replace(
                evaluation,
                uncertainThreshold=thresholds[0],
                lostThreshold=thresholds[1],
            )
        transaction.attempts.append(
            AttemptRecord(
                kind=AttemptKind.PRIMARY if plan.attemptIndex == 0 else AttemptKind.ESCALATION,
                attemptIndex=plan.attemptIndex,
                plan=plan,
                observations=tuple(observations),
                evaluation=evaluation,
            )
        )
        transaction.completedAttempts += 1
        if self._shouldEscalate(evaluation, transaction, allowEscalation):
            transaction.attemptIndex += 1
            self._planned = None
            nextPlan = self._buildAttempt(
                frame=planned.frame,
                state=planned.state,
                prediction=planned.prediction,
                predictedBfov=planned.predictedBfov,
                attemptIndex=transaction.attemptIndex,
                searchSeed=evaluation.searchSeedCenter,
                viewIdStart=max((view.viewId for view in plan.views), default=-1) + 1,
            )
            return MoreViewsRequired(nextPlan)
        decision = self._stateMachine.transition(
            planned.state.mode,
            evaluation.stateScore,
            measurementAccepted=evaluation.measurementAccepted,
        )
        result = self._commit(planned, evaluation, decision)
        self._stateMachine.recordScore(evaluation.stateScore)
        self._lastStateObservation = evaluation
        self._lastTransition = decision
        return FrameCommitted(result)

    def _buildAttempt(
        self,
        *,
        frame: FramePacket,
        state: StateInstance,
        prediction: MotionPrediction,
        predictedBfov: BFoV,
        attemptIndex: int,
        searchSeed,
        viewIdStart: int,
    ) -> SearchPlan:
        transaction = self._transaction
        if transaction is None:
            raise ProtocolError("attempt requires an active frame transaction")
        status = _publicStatus(transaction.startingMode)
        views = self._planner.buildViews(
            int(frame.frameIndex),
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            self._initialBox or self._currentBox,
            self._currentBox,
            self._currentBfov,
            prediction.motionState,
            status,
            searchSeedCenter=searchSeed,
            attemptIndex=attemptIndex,
            viewIdStart=viewIdStart,
            viewBudget=transaction.remainingViews,
            recoveryMemory=transaction.recoveryMemory,
        )
        if not views:
            raise ProtocolError("view planner returned an empty attempt")
        transaction.remainingViews -= len(views)
        nextRevision = self._stateRevision + 1
        command = TemplateCommand(
            kind=self._pendingTemplate.kind if attemptIndex == 0 else TemplateCommandKind.KEEP,
            frameIndex=frame.frameIndex,
            viewId=self._pendingTemplate.viewId if attemptIndex == 0 else None,
            localBox=self._pendingTemplate.localBox if attemptIndex == 0 else None,
            expectedRevision=self._backendRevision + 1,
        )
        searchPlan = SearchPlan(
            sequenceId=frame.sequenceId,
            frameIndex=frame.frameIndex,
            stateRevision=nextRevision,
            views=tuple(item.spec for item in views),
            templateCommand=command,
            predictedMotion=prediction.motionState,
            transactionId=transaction.transactionId,
            attemptIndex=attemptIndex,
            recoveryEpochId=(
                transaction.recoveryMemory.epochId
                if transaction.recoveryMemory is not None
                else self._recovery.epochId
            ),
            viewRoles=tuple(item.role for item in views),
            appearanceOnlyScoring=False,
        )
        self._planned = _PlannedAttempt(
            frame=frame,
            plan=searchPlan,
            prediction=prediction,
            predictedBfov=predictedBfov,
            state=state,
            viewsById={item.spec.viewId: item for item in views},
        )
        self._pendingTemplate = TemplateDecision(TemplateCommandKind.KEEP)
        return searchPlan

    def _shouldEscalate(
        self,
        evaluation: StateObservation,
        transaction: FrameTransaction,
        allowEscalation: bool,
    ) -> bool:
        return (
            evaluation.escalationRecommended
            and allowEscalation
            and self._trackingConfig.sameFrameEscalationEnabled
            and transaction.attemptIndex + 1 < self._trackingConfig.maxAttemptsPerFrame
            and transaction.remainingViews > 0
        )

    def _commit(self, planned, evaluation, decision) -> TrackResult:
        if self._transaction is not None and self._transaction.recoveryMemory is not None:
            self._recovery = self._transaction.recoveryMemory
        hasCandidate = evaluation.bestCandidate is not None
        accepted = decision.acceptMeasurement and hasCandidate
        outputBfov = evaluation.proposedOutputBfov
        outputBox = evaluation.proposedOutputBbox
        outputDepth = evaluation.depthSummary if hasCandidate else self._currentDepth
        outputConfidence = (
            evaluation.stateScore
            if hasCandidate
            else max(0.0, planned.prediction.confidence * 0.5)
        )
        if accepted:
            assert evaluation.measuredBfov is not None
            assert evaluation.measuredBbox is not None
            outputBfov = evaluation.measuredBfov
            outputBox = evaluation.measuredBbox
            assert outputBox is not None
            if decision.resetMotionHistory and hasattr(self._motion, "resetFromMeasurement"):
                self._motion.resetFromMeasurement(  # type: ignore[attr-defined]
                    outputBfov.center,
                    outputDepth,
                    planned.frame.timestampNs,
                    int(planned.frame.frameIndex),
                    outputConfidence,
                    outputBfov.horizontalFovRad,
                    outputBfov.verticalFovRad,
                )
                self._reacquireCooldown = self._trackingConfig.reacquireCooldownFrames
                source = ResultSource.OBSERVED_REACQUIRED
            else:
                self._recordMeasurement(planned, outputBfov, outputDepth, outputConfidence)
                source = ResultSource.OBSERVED_CONFIRMED
            self._currentBox = outputBox
            self._currentBfov = outputBfov
            self._currentDepth = outputDepth
        else:
            source = (
                ResultSource.OBSERVED_WEAK_BLEND if hasCandidate else ResultSource.MOTION_PREDICTED
            )
            # Break the motion/acceptance bootstrap cycle without committing a weak box as the
            # public target state.  A single bounded provisional observation supplies the second
            # timestamped point required for velocity fitting; subsequent weak frames do not
            # continue polluting the history.
            if (
                hasCandidate
                and planned.prediction.sampleCount < self._motionMinSamples
                and planned.state.mode in {TrackMode.TRACKING, TrackMode.UNCERTAIN}
                and evaluation.measuredBfov is not None
            ):
                self._recordMeasurement(
                    planned,
                    evaluation.measuredBfov,
                    evaluation.depthSummary,
                    max(self._trackingConfig.candidateMinScore, evaluation.stateScore),
                )
        oldMode = self._mode
        self._mode = decision.nextMode
        self._entryReason = decision.reason
        self._modeAgeFrames = self._modeAgeFrames + 1 if self._mode is oldMode else 0
        if accepted and self._mode is TrackMode.TRACKING:
            self._stableFrames += 1
            self._weakFrames = 0
            self._recoveryFrames = 0
        else:
            self._stableFrames = 0
            if self._mode is TrackMode.UNCERTAIN:
                self._weakFrames = 1 if oldMode is TrackMode.TRACKING else self._weakFrames + 1
            elif self._mode is TrackMode.LOST:
                self._recoveryFrames += 1
                self._recovery.framesSpent += 1
        if oldMode is not TrackMode.LOST and self._mode is TrackMode.LOST:
            self._recovery.reset(planned.frame.frameIndex)
        if decision.resetRecoveryEpoch:
            self._recovery = RecoveryMemory(epochId=self._recovery.epochId + 1)
        if (
            evaluation.measuredCenter is not None
            and evaluation.stateScore > self._recovery.bestSeedScore
        ):
            self._recovery.bestSeedCenter = evaluation.measuredCenter
            self._recovery.bestSeedScore = evaluation.stateScore
            self._recovery.bestSeedFrameIndex = planned.frame.frameIndex
        aggregate = _aggregateAdapter(evaluation)
        self._pendingTemplate = self._templatePolicy.decide(
            _publicStatus(self._mode),
            self._stableFrames if self._reacquireCooldown == 0 else 0,
            aggregate,
        )
        if self._reacquireCooldown > 0:
            self._reacquireCooldown -= 1
        self._stateRevision = planned.plan.stateRevision
        self._lastFrameIndex = int(planned.frame.frameIndex)
        self._lastFrame = planned.frame
        self._planned = None
        self._transaction = None
        return TrackResult(
            sequenceId=planned.frame.sequenceId,
            frameIndex=planned.frame.frameIndex,
            bbox=outputBox,
            bfov=outputBfov,
            confidence=outputConfidence,
            status=_publicStatus(self._mode),
            valid=accepted,
            depthSummary=outputDepth,
            resultSource=source,
        )

    def _recordMeasurement(self, planned, bfov, depth, confidence) -> None:
        if hasattr(self._motion, "recordMeasurement"):
            self._motion.recordMeasurement(  # type: ignore[attr-defined]
                frameIndex=int(planned.frame.frameIndex),
                timestampNs=planned.frame.timestampNs,
                point=bfov.center,
                depth=depth,
                confidence=confidence,
                horizontalSizeRad=bfov.horizontalFovRad,
                verticalSizeRad=bfov.verticalFovRad,
            )
        else:
            self._motion.update(bfov.center, depth, planned.frame.timestampNs, confidence)

    def _predictDetailed(self, frame: FramePacket) -> MotionPrediction:
        if hasattr(self._motion, "predictDetailed"):
            prediction = self._motion.predictDetailed(  # type: ignore[attr-defined]
                frame.timestampNs,
                min(self._trackingConfig.maxPredictionHorizon, max(1, self._recoveryFrames + 1)),
            )
            return replace(
                prediction,
                sourceRevision=max(0, self._stateRevision),
                targetFrameIndex=frame.frameIndex,
            )
        motion = self._motion.predict(frame.timestampNs)
        center = _motionCenter(motion)
        return MotionPrediction(
            sourceRevision=max(0, self._stateRevision),
            targetFrameIndex=frame.frameIndex,
            horizonFrames=1,
            center=center,
            horizontalSizeRad=self._currentBfov.horizontalFovRad,
            verticalSizeRad=self._currentBfov.verticalFovRad,
            tangentVelocityRadPerSec=(0.0, 0.0),
            rangeDepth=motion.rangeDepth or None,
            rangeVelocityPerSec=motion.rangeVelocity,
            angularUncertaintyRad=0.05,
            scaleUncertainty=0.10,
            rangeUncertainty=None,
            confidence=motion.confidence,
            centerCovarianceRad2=motion.centerCovarianceRad2,
            scaleCovarianceLog2=motion.scaleCovarianceLog2,
            rangeVariance=motion.rangeVariance,
            reliability=motion.reliability,
        )

    def _makeViewSpec(self, viewId: int, bfov: BFoV) -> ViewSpec:
        return ViewSpec(
            viewId=viewId,
            bfov=bfov,
            outputWidthPx=self._geometryConfig.viewWidthPx,
            outputHeightPx=self._geometryConfig.viewHeightPx,
        )

    def _requireInitialized(self) -> None:
        if not self._initialized:
            raise ProtocolError("controller has not been initialized")

    def _requireFrameOrder(self, frame: FramePacket) -> None:
        if self._sequenceId != str(frame.sequenceId):
            raise ProtocolError("frame sequence does not match controller sequence")
        if int(frame.frameIndex) != self._lastFrameIndex + 1:
            raise ProtocolError(
                f"frame index must be {self._lastFrameIndex + 1}, actual={frame.frameIndex}"
            )


def _motionCenter(motion: MotionState3D):
    x, y, z = motion.position
    return makeSphericalPoint(atan2(x, z), asin(max(-1.0, min(1.0, y))))


def _publicStatus(mode: TrackMode) -> TrackStatus:
    if mode is TrackMode.UNCERTAIN:
        return TrackStatus.UNCERTAIN
    if mode is TrackMode.LOST:
        return TrackStatus.LOST
    return TrackStatus.TRACKING


def _boxForBfov(
    viewWidthPx: int,
    viewHeightPx: int,
    viewBfov: BFoV,
    objectBfov: BFoV,
) -> BBoxXYWH:
    focalX = fovToFocalLengthPx(viewBfov.horizontalFovRad, viewWidthPx)
    focalY = fovToFocalLengthPx(viewBfov.verticalFovRad, viewHeightPx)
    width = 2.0 * focalX * tan(objectBfov.horizontalFovRad / 2.0)
    height = 2.0 * focalY * tan(objectBfov.verticalFovRad / 2.0)
    width = max(2.0, min(float(viewWidthPx), width))
    height = max(2.0, min(float(viewHeightPx), height))
    return BBoxXYWH(
        xPx=(float(viewWidthPx) - width) / 2.0,
        yPx=(float(viewHeightPx) - height) / 2.0,
        widthPx=width,
        heightPx=height,
    )


def _aggregateAdapter(observation):
    """Provide the existing TemplatePolicy with only the selected-cluster fields it consumes."""
    if observation.representativeViewId is None or observation.measuredBfov is None:
        return None
    from instatarget.controller.decision_gate import FrameAggregate

    return FrameAggregate(
        bfov=observation.measuredBfov,
        bbox=observation.measuredBbox,
        confidence=observation.stateScore,
        decisionScore=observation.stateScore,
        sourceViewIds=observation.sourceViewIds,
        representativeViewId=observation.representativeViewId,
        localBox=observation.representativeLocalBox,
        depthSummary=observation.depthSummary,
        supported=observation.supported,
        clusterCount=observation.clusterCount,
        agreementScore=observation.agreementScore,
    )


TrackerControllerImpl = DepthAwareTrackController

__all__ = ["DepthAwareTrackController", "TrackerControllerImpl"]
