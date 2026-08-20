"""Controller components for spherical RGB tracking."""

from instatarget.controller.classifier import (
    CLASSIFY_RADIUS_RAD,
    Classifier,
    ClusterCenter,
    classify,
)
from instatarget.controller.decision_gate import DecisionGate, FrameAggregate, ScoredObservation
from instatarget.controller.fused_score import (
    MotionScore,
    calibrateBackendFusedScore,
    calibrateLocalAppearanceProbabilities,
    calibrateMotionScore,
    composeSingleScore,
    scoreMotionConsistency,
    scoreViewCenterMotion,
)
from instatarget.controller.fusor import (
    FUSION_AGREEMENT_BONUS_WEIGHT,
    FUSION_MAX_SCORE_GAIN,
    FUSION_OVERLAP_RATE,
    FUSION_SCORE_CAP,
    FusionBoxMode,
    Fusor,
    fuse,
)
from instatarget.controller.motion_estimator import MotionEstimatorImpl, SphericalMotionEstimator
from instatarget.controller.recovery_planner import PlannedView, RecoveryPlanner, ViewSpecType1
from instatarget.controller.score_calibration import (
    UNCALIBRATED_STAGE3_SCORE_CALIBRATION,
    BetaCalibration,
    ScoreCalibration,
    loadScoreCalibration,
)
from instatarget.controller.speculative_pipeline import (
    RollbackReason,
    SpeculativeDecision,
    SpeculativePipeline,
    SpeculativeState,
    SpeculativeSummary,
    evaluateSpeculation,
)
from instatarget.controller.state_evaluator import StateEvaluator
from instatarget.controller.state_machine import StateUpdate, TrackStateMachine
from instatarget.controller.state_model import (
    EvaluatedCandidate,
    EvidenceLevel,
    MeasurementEvidence,
    MotionPrediction,
    RecoveryMemory,
    StateInstance,
    StateObservation,
    TrackMode,
)
from instatarget.controller.template_policy import TemplateDecision, TemplatePolicy
from instatarget.controller.track_controller import TrackControllerImpl

__all__ = [
    "DecisionGate",
    "CLASSIFY_RADIUS_RAD",
    "Classifier",
    "ClusterCenter",
    "FUSION_AGREEMENT_BONUS_WEIGHT",
    "FUSION_MAX_SCORE_GAIN",
    "FUSION_OVERLAP_RATE",
    "FUSION_SCORE_CAP",
    "FusionBoxMode",
    "Fusor",
    "MotionScore",
    "BetaCalibration",
    "FrameAggregate",
    "fuse",
    "classify",
    "MotionEstimatorImpl",
    "PlannedView",
    "RecoveryPlanner",
    "ScoreCalibration",
    "ViewSpecType1",
    "calibrateBackendFusedScore",
    "calibrateLocalAppearanceProbabilities",
    "calibrateMotionScore",
    "composeSingleScore",
    "UNCALIBRATED_STAGE3_SCORE_CALIBRATION",
    "ScoredObservation",
    "StateEvaluator",
    "StateInstance",
    "StateObservation",
    "EvidenceLevel",
    "EvaluatedCandidate",
    "MeasurementEvidence",
    "MotionPrediction",
    "RecoveryMemory",
    "RollbackReason",
    "SphericalMotionEstimator",
    "SpeculativeDecision",
    "SpeculativePipeline",
    "SpeculativeState",
    "SpeculativeSummary",
    "scoreMotionConsistency",
    "scoreViewCenterMotion",
    "loadScoreCalibration",
    "evaluateSpeculation",
    "StateUpdate",
    "TemplateDecision",
    "TemplatePolicy",
    "TrackStateMachine",
    "TrackMode",
    "TrackControllerImpl",
]
