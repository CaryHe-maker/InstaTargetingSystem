import unittest
from math import pi
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from instatarget.app.driver import _projectObservation
from instatarget.controller import (
    MotionScore,
    composeSingleScore,
    remapFusedScore,
    remapLocalObservationFusedScores,
    scoreMotionConsistency,
    scoreViewCenterMotion,
)
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    LocalObservation,
    LocalView,
    MotionState3D,
    SequenceId,
    ViewSpec,
)
from instatarget.geometry import makeSphericalPoint


class FusedScoreRemappingTest(unittest.TestCase):
    def testBetaCalibrationHitsRequestedAnchors(self) -> None:
        expected = {
            0.00: 0.00,
            0.80: 0.05,
            0.90: 0.45,
            0.98: 0.97,
            1.00: 1.00,
        }

        for rawScore, remappedScore in expected.items():
            with self.subTest(rawScore=rawScore):
                self.assertAlmostEqual(remapFusedScore(rawScore), remappedScore, places=7)

    def testStronglyStretchesPointEightToPointNineEight(self) -> None:
        rawScores = (0.60, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
        remapped = tuple(remapFusedScore(score) for score in rawScores)

        self.assertEqual(remapped, tuple(sorted(remapped)))
        self.assertGreater(remapFusedScore(0.98) - remapFusedScore(0.80), 0.90)
        self.assertLess(remapFusedScore(0.85), 0.20)
        self.assertGreater(remapFusedScore(0.95), 0.80)

    def testCreatesNewObservationsWithoutOverwritingBackendScore(self) -> None:
        original = _observation(0.85)

        (remapped,) = remapLocalObservationFusedScores((original,))

        self.assertIsNot(remapped, original)
        self.assertEqual(original.fusedScore, 0.85)
        self.assertEqual(remapped.fusedScore, 0.85)
        self.assertAlmostEqual(remapped.appearanceProbability or 0.0, 0.16277496, places=7)
        self.assertEqual(remapped.bbox, original.bbox)
        self.assertEqual(remapped.appearanceScore, original.appearanceScore)

    def testMotionScoreUsesPredictionCovarianceAndReliability(self) -> None:
        prediction = MotionState3D(
            position=(0.0, 0.0, 1.0),
            velocity=(0.0, 0.0, 0.0),
            rangeDepth=0.0,
            rangeVelocity=0.0,
            confidence=0.9,
            horizontalSizeRad=0.4,
            verticalSizeRad=0.3,
            angularUncertaintyRad=0.03,
            scaleUncertainty=0.1,
            reliability=1.0,
            centerCovarianceRad2=((0.0009, 0.0), (0.0, 0.0009)),
            scaleCovarianceLog2=((0.01, 0.0), (0.0, 0.01)),
        )
        aligned = scoreMotionConsistency(
            BFoV(makeSphericalPoint(0.0, 0.0), 0.4, 0.3), None, prediction
        )
        displaced = scoreMotionConsistency(
            BFoV(makeSphericalPoint(0.35, 0.0), 0.4, 0.3), None, prediction
        )

        self.assertGreater(aligned.rawScore, displaced.rawScore)
        self.assertGreater(aligned.effectiveProbability, displaced.effectiveProbability)

        unreliable = scoreMotionConsistency(
            BFoV(makeSphericalPoint(0.35, 0.0), 0.4, 0.3),
            None,
            MotionState3D(
                position=prediction.position,
                velocity=prediction.velocity,
                rangeDepth=0.0,
                rangeVelocity=0.0,
                confidence=0.9,
            ),
        )
        self.assertAlmostEqual(unreliable.effectiveProbability, 0.5)

    def testSingleScoreUsesSeventyThirtyLinearPool(self) -> None:
        self.assertAlmostEqual(composeSingleScore(0.8, 0.2), 0.62)

    def testViewCenterMotionFallsContinuouslyByPointOnePerThirtyDegrees(self) -> None:
        prediction = MotionState3D(
            position=(0.0, 0.0, 1.0),
            velocity=(0.0, 0.0, 0.0),
            rangeDepth=0.0,
            rangeVelocity=0.0,
            confidence=1.0,
            reliability=0.25,
        )
        expected = {
            0.0: 1.0,
            30.0: 0.9,
            45.0: 0.85,
            60.0: 0.8,
            90.0: 0.7,
            180.0: 0.4,
        }

        for angleDeg, expectedScore in expected.items():
            with self.subTest(angleDeg=angleDeg):
                score = scoreViewCenterMotion(
                    makeSphericalPoint(angleDeg * pi / 180.0, 0.0),
                    prediction,
                )
                self.assertAlmostEqual(score.effectiveProbability, expectedScore)

    def testViewCenterMotionUsesLocalViewCenterNotDetectionCenter(self) -> None:
        prediction = MotionState3D(
            position=(0.0, 0.0, 1.0),
            velocity=(0.0, 0.0, 0.0),
            rangeDepth=0.0,
            rangeVelocity=0.0,
            confidence=1.0,
        )

        centered = scoreViewCenterMotion(makeSphericalPoint(0.0, 0.0), prediction)
        sideView = scoreViewCenterMotion(makeSphericalPoint(pi / 3.0, 0.0), prediction)

        self.assertAlmostEqual(centered.effectiveProbability, 1.0)
        self.assertAlmostEqual(sideView.effectiveProbability, 0.8)

    def testProjectionPathScoresTheLocalViewCenter(self) -> None:
        predicted = MotionState3D(
            position=(0.0, 0.0, 1.0),
            velocity=(0.0, 0.0, 0.0),
            rangeDepth=0.0,
            rangeVelocity=0.0,
            confidence=1.0,
        )
        viewCenter = makeSphericalPoint(pi / 3.0, 0.0)
        view = LocalView(
            ViewSpec(3, BFoV(viewCenter, 1.0, 1.0), 256, 256),
            np.zeros((256, 256, 3), dtype=np.uint8),
        )
        observation = _observation(0.90)
        frame = FramePacket(
            SequenceId("view-center-score"),
            FrameIndex(1),
            1,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        geometry = Mock()
        geometry.projectLocalBoxBoundary.return_value = SimpleNamespace(
            bfov=BFoV(makeSphericalPoint(-1.0, 0.2), 0.2, 0.2),
            bbox=BBoxXYWH(20.0, 20.0, 30.0, 30.0),
            erpBoundary=(),
            envelopeInflation=1.0,
        )
        viewScore = MotionScore(0.8, 0.8, 0.8, 0.5, (pi / 3.0) ** 2)

        with patch("instatarget.app.driver.scoreViewCenterMotion", return_value=viewScore) as score:
            projected = _projectObservation(
                frame=frame,
                view=view,
                observation=observation,
                predictedMotion=predicted,
                geometry=geometry,
            )

        score.assert_called_once_with(view.spec.bfov.center, predicted)
        self.assertAlmostEqual(projected.motionScore, 0.8)


def _observation(score: float) -> LocalObservation:
    return LocalObservation(
        viewId=1,
        bbox=BBoxXYWH(10.0, 20.0, 30.0, 40.0),
        modelScore=0.88,
        appearanceScore=0.87,
        depthScore=0.70,
        fusedScore=score,
        depthSummary=None,
        latencyNs=1,
    )


if __name__ == "__main__":
    unittest.main()
