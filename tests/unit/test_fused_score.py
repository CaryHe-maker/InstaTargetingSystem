import unittest

from instatarget.controller import remapFusedScore, remapLocalObservationFusedScores
from instatarget.core.types import BBoxXYWH, LocalObservation


class FusedScoreRemappingTest(unittest.TestCase):
    def testMapsRequestedIntervalsContinuously(self) -> None:
        expected = {
            0.00: 0.00,
            0.30: 0.05,
            0.60: 0.10,
            0.70: 0.25,
            0.80: 0.40,
            0.85: 0.55,
            0.90: 0.70,
            0.925: 0.80,
            0.95: 0.90,
            0.975: 0.95,
            1.00: 1.00,
        }

        for rawScore, remappedScore in expected.items():
            with self.subTest(rawScore=rawScore):
                self.assertAlmostEqual(remapFusedScore(rawScore), remappedScore)

    def testPreservesOrderingWhileStretchingHighValueDifferences(self) -> None:
        rawScores = (0.50, 0.60, 0.80, 0.85, 0.90, 0.95, 1.00)
        remapped = tuple(remapFusedScore(score) for score in rawScores)

        self.assertEqual(remapped, tuple(sorted(remapped)))
        self.assertGreater(remapped[5] - remapped[2], rawScores[5] - rawScores[2])
        self.assertLessEqual(remapFusedScore(0.60), 0.10)

    def testCreatesNewObservationsWithOnlyFusedScoreChanged(self) -> None:
        original = _observation(0.85)

        (remapped,) = remapLocalObservationFusedScores((original,))

        self.assertIsNot(remapped, original)
        self.assertEqual(original.fusedScore, 0.85)
        self.assertAlmostEqual(remapped.fusedScore, 0.55)
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
