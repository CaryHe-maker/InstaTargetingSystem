import math
import unittest
from pathlib import Path

import numpy as np

from instatarget.controller import (
    DecisionGate,
    DepthAwareTrackController,
    RecoveryMemory,
    RecoveryPlanner,
    SphericalMotionEstimator,
)
from instatarget.core.config import DecisionGateConfig, loadConfig
from instatarget.core.protocols import FrameCommitted, MoreViewsRequired
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    ProjectedObservation,
    SequenceId,
    TrackStatus,
)
from instatarget.geometry import SphericalGeometryImpl, makeSphericalPoint

ROOT = Path(__file__).resolve().parents[2]


def _candidate(viewId: int, yawRad: float, score: float = 0.9) -> ProjectedObservation:
    bfov = BFoV(makeSphericalPoint(yawRad, 0.0), 0.35, 0.25)
    return ProjectedObservation(
        viewId=viewId,
        bfov=bfov,
        bbox=BBoxXYWH(20.0 + viewId * 120.0, 60.0, 25.0, 20.0),
        modelScore=score,
        appearanceScore=score,
        motionScore=score,
        scaleScore=score,
        depthScore=0.0,
        fusedScore=score,
        depthSummary=None,
        localBox=BBoxXYWH(80.0, 80.0, 30.0, 30.0),
    )


class ControllerV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "RGBonly.yaml")
        self.geometry = SphericalGeometryImpl(
            boundarySamplesPerEdge=self.config.geometry.boundarySamplesPerEdge
        )

    def testDisjointCandidatesAreNotMergedIntoOneUnionBox(self) -> None:
        gate = DecisionGate(DecisionGateConfig(0.25, 0.15, 0.10), self.config.tracking)
        aggregate = gate.aggregate(
            (_candidate(0, 0.0), _candidate(1, 2.2)),
            self.geometry,
            360,
            180,
        )
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(len(aggregate.sourceViewIds), 1)
        self.assertEqual(aggregate.clusterCount, 2)
        self.assertLess(aggregate.bbox.widthPx, 100.0)

    def testControllerEscalatesAtMostOnceAndCommitsOneResult(self) -> None:
        frame0 = FramePacket(
            SequenceId("v2"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = DepthAwareTrackController(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init, None)
        frame1 = FramePacket(
            SequenceId("v2"),
            FrameIndex(1),
            1_000_000_000,
            frame0.rgb,
        )
        firstPlan = controller.beginFrame(frame1)
        firstStep = controller.consume(firstPlan, ())
        self.assertIsInstance(firstStep, MoreViewsRequired)
        assert isinstance(firstStep, MoreViewsRequired)
        self.assertEqual(firstStep.plan.transactionId, firstPlan.transactionId)
        self.assertEqual(firstStep.plan.attemptIndex, 1)
        self.assertEqual(firstPlan.templateCommand.expectedRevision, 1)
        self.assertEqual(firstStep.plan.templateCommand.expectedRevision, 2)
        secondStep = controller.consume(firstStep.plan, ())
        self.assertIsInstance(secondStep, FrameCommitted)
        assert isinstance(secondStep, FrameCommitted)
        self.assertEqual(secondStep.result.frameIndex, FrameIndex(1))
        self.assertFalse(secondStep.result.valid)
        self.assertEqual(secondStep.result.status, TrackStatus.RECOVERING)

    def testMotionHistoryContainsMeasurementsNotPredictions(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=3)
        estimator.initialize(makeSphericalPoint(math.pi - 0.05, 0.0), None, 0)
        estimator.predict(500_000_000)
        self.assertEqual(len(estimator.samples), 1)
        estimator.update(
            makeSphericalPoint(-math.pi + 0.05, 0.0),
            None,
            1_000_000_000,
            0.9,
        )
        self.assertEqual(len(estimator.samples), 2)
        prediction = estimator.predictDetailed(2_000_000_000)
        self.assertGreater(prediction.angularUncertaintyRad, 0.0)
        self.assertIn("missing_depth", prediction.degradedReasons)

    def testMotionPredictionHasStableTangentBasisAtPole(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=3)
        estimator.initialize(makeSphericalPoint(0.0, math.pi / 2.0 - 0.10), None, 0)
        estimator.update(
            makeSphericalPoint(0.0, math.pi / 2.0),
            None,
            1_000_000_000,
            0.9,
        )
        prediction = estimator.predictDetailed(2_000_000_000)
        self.assertTrue(math.isfinite(prediction.center.yawRad))
        self.assertTrue(math.isfinite(prediction.center.pitchRad))
        self.assertLess(prediction.center.pitchRad, math.pi / 2.0)

    def testLostGlobalPlanContainsEquatorAndPolarCubeFaces(self) -> None:
        planner = RecoveryPlanner(
            self.config.geometry,
            self.config.tracking,
            self.config.recovery,
        )
        box = BBoxXYWH(150.0, 70.0, 40.0, 50.0)
        fallback = self.geometry.bboxToBfov(box, 360, 180)
        views = planner.buildViews(
            self.config.recovery.globalSearchInterval,
            360,
            180,
            box,
            box,
            fallback,
            None,
            TrackStatus.LOST,
            recoveryMemory=RecoveryMemory(),
        )
        self.assertEqual(len(views), 6)
        self.assertEqual(sum(item.role == "cubemap_equator" for item in views), 4)
        self.assertEqual(sum(item.role == "cubemap_pole" for item in views), 2)

    def testRecoveryPlanRemainsNonEmptyAfterHistoryDeduplicatesFallback(self) -> None:
        planner = RecoveryPlanner(
            self.config.geometry,
            self.config.tracking,
            self.config.recovery,
        )
        box = BBoxXYWH(150.0, 70.0, 40.0, 50.0)
        fallback = self.geometry.bboxToBfov(box, 360, 180)
        memory = RecoveryMemory()

        plans = [
            planner.buildViews(
                1,
                360,
                180,
                box,
                box,
                fallback,
                None,
                TrackStatus.RECOVERING,
                recoveryMemory=memory,
            )
            for _ in range(5)
        ]

        self.assertTrue(all(plans))
        self.assertEqual(plans[3][0].role, "fallback_probe")
        self.assertEqual(plans[4][0].role, "fallback_probe")


if __name__ == "__main__":
    unittest.main()
