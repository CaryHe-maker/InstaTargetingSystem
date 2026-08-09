"""Stateful DTC facade that closes the geometry -> tracker -> control loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from instatarget.controller.decision_gate import DecisionGate
from instatarget.controller.motion_estimator import SphericalMotionEstimator
from instatarget.controller.recovery_planner import PlannedView, RecoveryPlanner
from instatarget.controller.state_machine import TrackStateMachine
from instatarget.controller.template_policy import TemplateDecision, TemplatePolicy
from instatarget.core.config import (
    AppConfig,
    DecisionGateConfig,
    GeometryConfig,
    RecoveryConfig,
    TrackingConfig,
)
from instatarget.core.errors import ProtocolError
from instatarget.core.protocols import SphericalGeometry, TrackController
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    DepthSummary,
    FrameIndex,
    FramePacket,
    InitializationPlan,
    MotionState3D,
    ProjectedObservation,
    SearchPlan,
    TemplateCommand,
    TemplateCommandKind,
    TrackResult,
    TrackStatus,
)
from instatarget.geometry.projection_math import makeSphericalPoint

if TYPE_CHECKING:
    from instatarget.core.protocols import MotionEstimator


@dataclass(frozen=True, slots=True)
class _PlannedFrame:
    frame: FramePacket
    plan: SearchPlan
    predictedMotion: MotionState3D
    predictedBfov: BFoV
    viewsById: dict[int, PlannedView]


class DepthAwareTrackController(TrackController):
    """Single-writer controller for multi-view spherical tracking."""

    def __init__(
        self,
        geometry: SphericalGeometry,
        config: AppConfig | None = None,
        *,
        geometryConfig: GeometryConfig | None = None,
        trackingConfig: TrackingConfig | None = None,
        recoveryConfig: RecoveryConfig | None = None,
        decisionGateConfig: DecisionGateConfig | None = None,
        motionEstimator: MotionEstimator | None = None,
    ) -> None:
        if config is not None:
            geometryConfig = config.geometry
            trackingConfig = config.tracking
            recoveryConfig = config.recovery
            decisionGateConfig = config.decisionGate
        if geometryConfig is None or trackingConfig is None or recoveryConfig is None:
            raise ValueError("geometryConfig, trackingConfig and recoveryConfig are required")
        if decisionGateConfig is None:
            decisionGateConfig = DecisionGateConfig(0.25, 0.15)

        self._geometry = geometry
        self._geometryConfig = geometryConfig
        self._trackingConfig = trackingConfig
        self._recoveryConfig = recoveryConfig
        self._motion: MotionEstimator = motionEstimator or SphericalMotionEstimator()
        self._gate = DecisionGate(decisionGateConfig, trackingConfig)
        self._planner = RecoveryPlanner(geometryConfig, trackingConfig, recoveryConfig)
        self._stateMachine = TrackStateMachine(trackingConfig)
        self._templatePolicy = TemplatePolicy(trackingConfig)

        self._initialized = False
        self._sequenceId: str | None = None
        self._lastFrameIndex = -1
        self._stateRevision = -1
        self._lastFrame: FramePacket | None = None
        self._initialBox: BBoxXYWH | None = None
        self._currentBox: BBoxXYWH | None = None
        self._currentBfov: BFoV | None = None
        self._currentDepth: DepthSummary | None = None
        self._stableFrames = 0
        self._pendingTemplate = TemplateDecision(TemplateCommandKind.KEEP)
        self._planned: _PlannedFrame | None = None
        self._initialPlan: InitializationPlan | None = None

    @property
    def status(self) -> TrackStatus | None:
        return self._stateMachine.status

    @property
    def stateRevision(self) -> int:
        return max(0, self._stateRevision)

    def buildInitialization(
        self,
        frame: FramePacket,
        initialBox: BBoxXYWH,
    ) -> InitializationPlan:
        if self._initialized or self._initialPlan is not None:
            raise ProtocolError("controller is already initialized or has a pending initialization")
        if int(frame.frameIndex) != 0:
            raise ProtocolError("initialization must use frameIndex 0")
        objectBfov = self._geometry.bboxToBfov(initialBox, frame.rgb.shape[1], frame.rgb.shape[0])
        templateBfov = self._planner.contextBfov(
            objectBfov.center,
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            initialBox,
            initialBox,
        )
        plan = InitializationPlan(
            sequenceId=frame.sequenceId,
            frameIndex=FrameIndex(0),
            stateRevision=0,
            templateView=self._makeViewSpec(0, templateBfov),
            templateBox=_centerBoxInView(
                initialBox,
                self._geometryConfig.viewWidthPx,
                self._geometryConfig.viewHeightPx,
                self._trackingConfig.contextScale,
                self._trackingConfig.contextMarginRatio,
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
        )

    def plan(self, frame: FramePacket) -> SearchPlan:
        self._requireInitialized()
        if self._planned is not None:
            raise ProtocolError("a search plan is already awaiting update")
        self._requireFrameOrder(frame)
        if self._currentBox is None or self._currentBfov is None:
            raise ProtocolError("controller target state is incomplete")
        predictedMotion = self._motion.predict(frame.timestampNs)
        predictedBfov = self._planner.contextBfov(
            _motionCenter(predictedMotion),
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            self._initialBox or self._currentBox,
            self._currentBox,
        )
        status = self._stateMachine.status or TrackStatus.TRACKING
        plannedViews = self._planner.buildViews(
            int(frame.frameIndex),
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            self._initialBox or self._currentBox,
            self._currentBox,
            self._currentBfov,
            predictedMotion,
            status,
        )
        nextRevision = self._stateRevision + 1
        command = TemplateCommand(
            kind=self._pendingTemplate.kind,
            frameIndex=frame.frameIndex,
            viewId=self._pendingTemplate.viewId,
            localBox=self._pendingTemplate.localBox,
            expectedRevision=nextRevision,
        )
        searchPlan = SearchPlan(
            sequenceId=frame.sequenceId,
            frameIndex=frame.frameIndex,
            stateRevision=nextRevision,
            views=tuple(item.spec for item in plannedViews),
            templateCommand=command,
            predictedMotion=predictedMotion,
        )
        self._planned = _PlannedFrame(
            frame=frame,
            plan=searchPlan,
            predictedMotion=predictedMotion,
            predictedBfov=predictedBfov,
            viewsById={item.spec.viewId: item for item in plannedViews},
        )
        self._pendingTemplate = TemplateDecision(TemplateCommandKind.KEEP)
        return searchPlan

    def update(
        self,
        plan: SearchPlan,
        observations: Sequence[ProjectedObservation],
    ) -> TrackResult:
        self._requireInitialized()
        planned = self._planned
        if planned is None or planned.plan != plan:
            raise ProtocolError("search response does not match the pending plan")
        self._validateObservations(plan, observations)
        aggregate = self._gate.aggregate(
            observations,
            self._geometry,
            planned.frame.rgb.shape[1],
            planned.frame.rgb.shape[0],
        )
        stateUpdate = self._stateMachine.update(
            aggregate.confidence if aggregate is not None else None,
            aggregate.supported if aggregate is not None else False,
            aggregate is not None,
        )
        accepted = stateUpdate.accepted and aggregate is not None
        if accepted and aggregate is not None:
            motionState = self._motion.update(
                aggregate.bfov.center,
                aggregate.depthSummary,
                planned.frame.timestampNs,
                aggregate.confidence,
            )
            outputBfov = aggregate.bfov
            outputBox = aggregate.bbox
            outputDepth = aggregate.depthSummary
            outputConfidence = aggregate.confidence
        else:
            motionState = planned.predictedMotion
            # An uncommitted candidate must not overwrite the last accepted
            # state.  Report only the bounded motion prediction when the gate
            # rejects or lacks multi-view support.
            outputBfov = planned.predictedBfov
            outputBox = self._geometry.bfovToBbox(
                outputBfov,
                planned.frame.rgb.shape[1],
                planned.frame.rgb.shape[0],
            )
            outputDepth = self._currentDepth
            outputConfidence = max(0.0, motionState.confidence * 0.5)

        if accepted and stateUpdate.status is TrackStatus.TRACKING:
            self._stableFrames += 1
            self._currentBox = outputBox
            self._currentBfov = outputBfov
            self._currentDepth = outputDepth
        else:
            self._stableFrames = 0
        self._pendingTemplate = self._templatePolicy.decide(
            stateUpdate.status,
            self._stableFrames,
            aggregate if accepted else None,
        )
        self._stateRevision = plan.stateRevision
        self._lastFrameIndex = int(planned.frame.frameIndex)
        self._lastFrame = planned.frame
        self._planned = None
        return TrackResult(
            sequenceId=planned.frame.sequenceId,
            frameIndex=planned.frame.frameIndex,
            bbox=outputBox,
            bfov=outputBfov,
            confidence=outputConfidence,
            status=stateUpdate.status,
            valid=accepted,
            depthSummary=outputDepth,
        )

    def _makeViewSpec(self, viewId: int, bfov: BFoV):
        from instatarget.core.types import ViewSpec

        return ViewSpec(
            viewId=viewId,
            bfov=bfov,
            outputWidthPx=self._geometryConfig.viewWidthPx,
            outputHeightPx=self._geometryConfig.viewHeightPx,
        )

    def _validateObservations(
        self,
        plan: SearchPlan,
        observations: Sequence[ProjectedObservation],
    ) -> None:
        expected = {view.viewId for view in plan.views}
        actual = [observation.viewId for observation in observations]
        if len(actual) != len(set(actual)):
            raise ProtocolError("projected observations must have unique viewIds")
        if not set(actual).issubset(expected):
            raise ProtocolError("projected observation contains an unknown viewId")

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
    from math import asin, atan2

    x, y, z = motion.position
    return makeSphericalPoint(atan2(x, z), asin(max(-1.0, min(1.0, y))))


def _centerBoxInView(
    box: BBoxXYWH,
    viewWidthPx: int,
    viewHeightPx: int,
    contextScale: float,
    contextMarginRatio: float,
) -> BBoxXYWH:
    if viewWidthPx <= 0 or viewHeightPx <= 0:
        raise ProtocolError("template view dimensions must be positive")
    if contextScale < 1.0 or contextMarginRatio < 0.0:
        raise ProtocolError("template context parameters are invalid")
    # The template view is a context crop.  Preserve the target's normalized
    # fraction within that crop instead of copying ERP pixel dimensions into
    # local coordinates.
    width = float(viewWidthPx) / (contextScale * (1.0 + contextMarginRatio))
    height = float(viewHeightPx) / (contextScale * (1.0 + contextMarginRatio))
    width = max(2.0, min(float(viewWidthPx), width))
    height = max(2.0, min(float(viewHeightPx), height))
    return BBoxXYWH(
        xPx=(float(viewWidthPx) - width) / 2.0,
        yPx=(float(viewHeightPx) - height) / 2.0,
        widthPx=width,
        heightPx=height,
    )


TrackerControllerImpl = DepthAwareTrackController

__all__ = ["DepthAwareTrackController", "TrackerControllerImpl"]
