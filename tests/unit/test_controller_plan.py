import math
import unittest
from pathlib import Path

from instatarget.controller import Classifier, Fusor, RecoveryPlanner
from instatarget.controller.state_model import ScoreGroup
from instatarget.core.config import loadConfig
from instatarget.core.types import BBoxXYWH, BFoV, ProjectedObservation, TrackStatus
from instatarget.geometry import SphericalGeometryImpl, makeSphericalPoint

ROOT = Path(__file__).resolve().parents[2]


def _observation(viewId: int, yaw: float, score: float, x: float = 30.0) -> ProjectedObservation:
    return ProjectedObservation(
        viewId=viewId,
        bfov=BFoV(makeSphericalPoint(yaw, 0.0), 0.5, 0.4),
        bbox=BBoxXYWH(x, 60.0, 100.0, 60.0),
        modelScore=score,
        appearanceScore=score,
        motionScore=score,
        scaleScore=score,
        depthScore=0.0,
        fusedScore=score,
        depthSummary=None,
        localBox=BBoxXYWH(70.0, 70.0, 40.0, 40.0),
        singleScore=score,
    )


class ControllerPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "RGBonly.yaml")
        self.geometry = SphericalGeometryImpl(
            boundarySamplesPerEdge=self.config.geometry.boundarySamplesPerEdge
        )

    def testFusorReturnsTheBestFusedCandidate(self) -> None:
        result = Fusor(self.geometry).fuse(
            (
                _observation(0, 0.0, 0.90, 30.0),
                _observation(1, 0.01, 0.90, 40.0),
                _observation(2, 2.0, 0.99, 220.0),
            ),
            frameWidthPx=360,
            frameHeightPx=180,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.fused)
        self.assertEqual(result.sourceViewIds, (0, 1))
        self.assertGreater(result.confidence, 0.80)

    def testFusorUsesCircularIntersectionAtTheSeam(self) -> None:
        result = Fusor(self.geometry).fuse(
            (
                _observation(0, 0.0, 0.95, 350.0),
                _observation(1, 0.0, 0.95, 355.0),
            ),
            frameWidthPx=360,
            frameHeightPx=180,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.fused)
        self.assertGreater(result.overlapRate or 0.0, 0.70)
        self.assertGreater(result.bbox.widthPx, 0.0)

    def testClassifierEnforcesSphericalThirtyDegreeRadiusAndRanksTopThree(self) -> None:
        result = Classifier().classify(
            (
                _observation(0, 0.0, 0.9),
                _observation(1, math.radians(10.0), 0.8),
                _observation(2, math.radians(120.0), 0.7),
                _observation(3, math.radians(-120.0), 0.6),
            )
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].memberCount, 2)
        self.assertEqual(result[0].memberViewIds, (0, 1))
        self.assertTrue(all(item.memberCount == 1 for item in result[1:]))

    def testScoreGroupUsesWarmupAndRollingOrderStatistics(self) -> None:
        group = ScoreGroup()
        self.assertIsNone(group.thresholds())
        group.append(0.2)
        self.assertIsNone(group.thresholds())
        group.append(0.8)
        self.assertAlmostEqual(group.thresholds()[0], 0.5)
        self.assertAlmostEqual(group.thresholds()[1], 0.32)
        for value in (0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95):
            group.append(value)
        thresholds = group.thresholds()
        self.assertIsNotNone(thresholds)
        assert thresholds is not None
        self.assertEqual(thresholds, (0.6, 0.3))
        self.assertEqual(len(group.values), 10)

    def testPlannerBuildsSecondRoundFourCornersAndTwelveViewLostRoute(self) -> None:
        planner = RecoveryPlanner(
            self.config.geometry,
            self.config.tracking,
            self.config.recovery,
        )
        box = BBoxXYWH(150.0, 70.0, 40.0, 50.0)
        fallback = self.geometry.bboxToBfov(box, 360, 180)
        refined = planner.buildViews(
            1,
            360,
            180,
            box,
            box,
            fallback,
            None,
            TrackStatus.TRACKING,
            attemptIndex=1,
            searchSeedCenter=makeSphericalPoint(0.0, 0.0),
        )
        self.assertEqual(len(refined), 4)
        self.assertEqual(tuple(item.spec.viewId for item in refined), (0, 1, 2, 3))
        self.assertEqual(
            tuple(item.role for item in refined),
            (
                "round2_left_top",
                "round2_right_top",
                "round2_left_bottom",
                "round2_right_bottom",
            ),
        )
        lost = planner.buildViews(
            1,
            360,
            180,
            box,
            box,
            fallback,
            None,
            TrackStatus.LOST,
        )
        self.assertEqual(len(lost), 12)
        self.assertTrue(
            all(
                item.spec.bfov.horizontalFovRad == self.config.geometry.maxFovRad
                for item in lost
            )
        )


if __name__ == "__main__":
    unittest.main()
