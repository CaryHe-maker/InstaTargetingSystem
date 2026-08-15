import unittest

from instatarget.controller import remapFusedScore, remapLocalObservationFusedScores
from instatarget.core.types import BBoxXYWH, LocalObservation


class FusedScoreRemappingTest(unittest.TestCase):
    def testBetaCalibrationHitsRequestedAnchors(self) -> None:
        expected = {
            0.00: 0.00,
            0.80: 0.15,
            0.90: 0.45,
            0.95: 0.70,
            1.00: 1.00,
        }

        for rawScore, remappedScore in expected.items():
            with self.subTest(rawScore=rawScore):
                self.assertAlmostEqual(remapFusedScore(rawScore), remappedScore, places=7)

    def testLowersScoresAndStretchesTargetInterval(self) -> None:
        rawScores = (0.60, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
        remapped = tuple(remapFusedScore(score) for score in rawScores)
        oldRemapped = (0.10, 0.40, 0.55, 0.70, 0.80, 0.90, 0.95, 0.98)

        self.assertEqual(remapped, tuple(sorted(remapped)))
        self.assertTrue(all(new < old for new, old in zip(remapped, oldRemapped, strict=True)))
        self.assertGreater(remapFusedScore(0.95) - remapFusedScore(0.80), 0.50)

    def testCreatesNewObservationsWithOnlyFusedScoreChanged(self) -> None:
        original = _observation(0.85)

        (remapped,) = remapLocalObservationFusedScores((original,))

        self.assertIsNot(remapped, original)
        self.assertEqual(original.fusedScore, 0.85)
        self.assertAlmostEqual(remapped.fusedScore, 0.26728930, places=7)
        self.assertEqual(remapped.bbox, original.bbox)
        self.assertEqual(remapped.appearanceScore, original.appearanceScore)


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
