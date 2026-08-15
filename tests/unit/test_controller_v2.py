import math
import unittest
from pathlib import Path

import numpy as np

from instatarget.controller import (
    DecisionGate,
    DepthAwareTrackController,
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


def _boxedCandidate(
    viewId: int,
    xPx: float,
    score: float,
    *,
    widthPx: float = 100.0,
) -> ProjectedObservation:
    bfov = BFoV(makeSphericalPoint(0.0, 0.0), 0.35, 0.25)
    return ProjectedObservation(
        viewId=viewId,
        bfov=bfov,
        bbox=BBoxXYWH(xPx, 40.0, widthPx, 80.0),
        modelScore=score,
        appearanceScore=score,
        motionScore=score,
        scaleScore=score,
        depthScore=0.0,
        fusedScore=score,
        depthSummary=None,
        localBox=BBoxXYWH(50.0, 50.0, 80.0, 80.0),
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

    def testTrackingUsesTwoRoundsAndCommitsOneResult(self) -> None:
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
        self.assertEqual(secondStep.result.status, TrackStatus.UNCERTAIN)

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
        )
        self.assertEqual(len(views), 6)
        self.assertEqual(sum("cubemap" in item.role for item in views), 6)
        self.assertEqual(
            {item.role for item in views},
            {
                "round1_cubemap_front",
                "round1_cubemap_right",
                "round1_cubemap_back",
                "round1_cubemap_left",
                "round1_cubemap_up",
                "round1_cubemap_down",
            },
        )
        self.assertTrue(
            all(
                math.isclose(item.spec.bfov.horizontalFovRad, math.radians(120.0))
                and math.isclose(item.spec.bfov.verticalFovRad, math.radians(120.0))
                for item in views
            )
        )

    def testRecoveringUsesFourCornersThenCubeMap(self) -> None:
        planner = RecoveryPlanner(
            self.config.geometry,
            self.config.tracking,
            self.config.recovery,
        )
        box = BBoxXYWH(150.0, 70.0, 40.0, 50.0)
        fallback = self.geometry.bboxToBfov(box, 360, 180)
        first = planner.buildViews(
            1,
            360,
            180,
            box,
            box,
            fallback,
            None,
            TrackStatus.RECOVERING,
            attemptIndex=0,
        )
        second = planner.buildViews(
            1,
            360,
            180,
            box,
            box,
            fallback,
            None,
            TrackStatus.RECOVERING,
            attemptIndex=1,
            viewIdStart=4,
        )
        third = planner.buildViews(
            1,
            360,
            180,
            box,
            box,
            fallback,
            None,
            TrackStatus.RECOVERING,
            attemptIndex=2,
            viewIdStart=8,
        )

        self.assertEqual((len(first), len(second), len(third)), (4, 4, 6))
        self.assertEqual(
            tuple(item.spec.viewId for item in (*first, *second, *third)),
            tuple(range(14)),
        )
        self.assertTrue(all(item.role.startswith("round3_cubemap_") for item in third))

    def testSourceConfidenceFloorBlocksFirstRoundReliableFusion(self) -> None:
        frame0 = FramePacket(
            SequenceId("floor"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = DepthAwareTrackController(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init, None)
        frame1 = FramePacket(SequenceId("floor"), FrameIndex(1), 1_000_000_000, frame0.rgb)
        first = controller.beginFrame(frame1)
        lowPair = (_boxedCandidate(0, 0.0, 0.79), _boxedCandidate(1, 0.0, 0.79))
        step = controller.consume(first, lowPair)

        self.assertIsInstance(step, MoreViewsRequired)
        assert isinstance(step, MoreViewsRequired)
        self.assertEqual(step.plan.attemptIndex, 1)
        final = controller.consume(
            step.plan,
            (_boxedCandidate(4, 0.0, 0.79), _boxedCandidate(5, 0.0, 0.79)),
        )
        self.assertIsInstance(final, FrameCommitted)
        assert isinstance(final, FrameCommitted)
        self.assertFalse(final.result.valid)
        self.assertEqual(final.result.status, TrackStatus.UNCERTAIN)
        assert controller.lastStateObservation is not None
        self.assertTrue(controller.lastStateObservation.selectedIsFused)
        self.assertFalse(controller.lastStateObservation.selectedSourceConfidencePassed)

    def testOverlapThresholdSevenTenthsAllowsFirstRoundFusion(self) -> None:
        frame0 = FramePacket(
            SequenceId("overlap"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = DepthAwareTrackController(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init, None)
        frame1 = FramePacket(SequenceId("overlap"), FrameIndex(1), 1_000_000_000, frame0.rgb)
        first = controller.beginFrame(frame1)
        pair = (_boxedCandidate(0, 0.0, 0.85), _boxedCandidate(1, 29.0, 0.85))
        result = controller.consume(first, pair)

        self.assertIsInstance(result, FrameCommitted)
        assert isinstance(result, FrameCommitted)
        self.assertTrue(result.result.valid)
        assert controller.lastStateObservation is not None
        self.assertGreater(controller.lastStateObservation.selectedOverlapRate or 0.0, 0.70)

    def testFirstRoundFusionUsesErpIntersectionBox(self) -> None:
        frame0 = FramePacket(
            SequenceId("wide-fusion"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = DepthAwareTrackController(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init, None)
        frame1 = FramePacket(
            SequenceId("wide-fusion"), FrameIndex(1), 1_000_000_000, frame0.rgb
        )
        plan = controller.beginFrame(frame1)

        result = controller.update(
            plan,
            (
                _boxedCandidate(0, 0.0, 0.99, widthPx=120.0),
                _boxedCandidate(1, 80.0, 0.99, widthPx=120.0),
            ),
        )

        self.assertAlmostEqual(result.bbox.xPx, 80.0)
        self.assertAlmostEqual(result.bbox.widthPx, 40.0)
        self.assertLess(result.bfov.horizontalFovRad, math.pi)
        self.assertFalse(result.valid)

    def testFusedResultUsesIntersectionAcrossErpSeam(self) -> None:
        frame0 = FramePacket(
            SequenceId("seam-intersection"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = DepthAwareTrackController(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init, None)
        frame1 = FramePacket(
            SequenceId("seam-intersection"),
            FrameIndex(1),
            1_000_000_000,
            frame0.rgb,
        )
        plan = controller.beginFrame(frame1)

        result = controller.update(
            plan,
            (
                _boxedCandidate(0, 340.0, 0.95, widthPx=40.0),
                _boxedCandidate(1, 350.0, 0.95, widthPx=30.0),
            ),
        )

        self.assertAlmostEqual(result.bbox.xPx, 350.0)
        self.assertAlmostEqual(result.bbox.widthPx, 30.0)
        self.assertTrue(result.valid)

    def testUncertainUsesThirdRoundAndLostUsesCubemapThenLocalRound(self) -> None:
        frame0 = FramePacket(
            SequenceId("routes"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = DepthAwareTrackController(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init, None)

        frame1 = FramePacket(SequenceId("routes"), FrameIndex(1), 1_000_000_000, frame0.rgb)
        first = controller.beginFrame(frame1)
        second = controller.consume(first, ())
        assert isinstance(second, MoreViewsRequired)
        uncertainResult = controller.consume(second.plan, ())
        assert isinstance(uncertainResult, FrameCommitted)
        self.assertEqual(uncertainResult.result.status, TrackStatus.UNCERTAIN)

        frame2 = FramePacket(SequenceId("routes"), FrameIndex(2), 2_000_000_000, frame0.rgb)
        uncertainFirst = controller.beginFrame(frame2)
        uncertainSecond = controller.consume(uncertainFirst, ())
        assert isinstance(uncertainSecond, MoreViewsRequired)
        uncertainThird = controller.consume(uncertainSecond.plan, ())
        assert isinstance(uncertainThird, MoreViewsRequired)
        self.assertEqual(len(uncertainThird.plan.views), 6)
        lostResult = controller.consume(uncertainThird.plan, ())
        assert isinstance(lostResult, FrameCommitted)
        self.assertEqual(lostResult.result.status, TrackStatus.LOST)

        frame3 = FramePacket(SequenceId("routes"), FrameIndex(3), 3_000_000_000, frame0.rgb)
        lostFirst = controller.beginFrame(frame3)
        self.assertEqual(len(lostFirst.views), 6)
        lostSecond = controller.consume(lostFirst, ())
        assert isinstance(lostSecond, MoreViewsRequired)
        self.assertEqual(len(lostSecond.plan.views), 4)
        lostFinal = controller.consume(lostSecond.plan, ())
        assert isinstance(lostFinal, FrameCommitted)
        self.assertEqual(lostFinal.result.status, TrackStatus.LOST)

    def testLostSingleCandidateRequiresRecoveringConfirmation(self) -> None:
        frame0 = FramePacket(
            SequenceId("recover"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = DepthAwareTrackController(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init, None)

        for index in (1, 2):
            frame = FramePacket(
                SequenceId("recover"), FrameIndex(index), index * 1_000_000_000, frame0.rgb
            )
            plan = controller.beginFrame(frame)
            while True:
                step = controller.consume(plan, ())
                if isinstance(step, MoreViewsRequired):
                    plan = step.plan
                    continue
                break
        self.assertEqual(step.result.status, TrackStatus.LOST)

        frame3 = FramePacket(SequenceId("recover"), FrameIndex(3), 3_000_000_000, frame0.rgb)
        lostFirst = controller.beginFrame(frame3)
        lostSecond = controller.consume(lostFirst, ())
        assert isinstance(lostSecond, MoreViewsRequired)
        pending = controller.consume(lostSecond.plan, (_boxedCandidate(6, 0.0, 0.95),))
        assert isinstance(pending, FrameCommitted)
        self.assertEqual(pending.result.status, TrackStatus.RECOVERING)
        self.assertFalse(pending.result.valid)

        frame4 = FramePacket(SequenceId("recover"), FrameIndex(4), 4_000_000_000, frame0.rgb)
        recoveringFirst = controller.beginFrame(frame4)
        recoveringSecond = controller.consume(
            recoveringFirst, (_boxedCandidate(0, 0.0, 0.95),)
        )
        assert isinstance(recoveringSecond, MoreViewsRequired)
        recovered = controller.consume(
            recoveringSecond.plan, (_boxedCandidate(4, 0.0, 0.95),)
        )
        assert isinstance(recovered, FrameCommitted)
        self.assertEqual(recovered.result.status, TrackStatus.TRACKING)
        self.assertTrue(recovered.result.valid)


if __name__ == "__main__":
    unittest.main()
