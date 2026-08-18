import tempfile
import unittest
from pathlib import Path

from instatarget.app.competition import (
    BfovResultSink,
    formatCompetitionResult,
    listSequences,
    loadInitialBfov,
)
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    ResultSource,
    SequenceId,
    TrackResult,
    TrackStatus,
)
from instatarget.geometry import makeSphericalPoint


class CompetitionSubmissionTest(unittest.TestCase):
    def testInitialBfovParsingUsesDegrees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "init.txt"
            path.write_text("-175,20,80,45\n", encoding="utf-8")

            bfov = loadInitialBfov(path)

        self.assertAlmostEqual(bfov.center.yawRad, -3.05432619099, places=6)
        self.assertAlmostEqual(bfov.verticalFovRad, 0.78539816339, places=6)

    def testSequenceDiscoveryHonorsSeqlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first").mkdir()
            (root / "second").mkdir()
            (root / "first" / "video.mp4").touch()
            (root / "first" / "init.txt").write_text("0,0,30,30\n", encoding="utf-8")
            (root / "second" / "video.mp4").touch()
            (root / "second" / "init.txt").write_text("0,0,30,30\n", encoding="utf-8")
            (root / "seqlist.txt").write_text("second\nfirst\n", encoding="utf-8")

            self.assertEqual(listSequences(root), ["second", "first"])

    def testSinkKeepsInitialBfovAndWritesZeroForLostFrame(self) -> None:
        initial = _bfov(yaw=-1.0, pitch=0.2, horizontal=0.7, vertical=0.5)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "sequence.txt"
            sink = BfovResultSink(initial)
            sink.open(str(destination))
            sink.write(_result(0, initial, valid=True))
            sink.write(
                _result(
                    1,
                    _bfov(yaw=0.5, pitch=0.0, horizontal=0.4, vertical=0.3),
                    valid=False,
                )
            )
            sink.finalize(2)

            self.assertEqual(
                destination.read_text(encoding="utf-8").splitlines(),
                ["-57.296,11.459,40.107,28.648", "0.000,0.000,0.000,0.000"],
            )

    def testFormatterUsesOfficialFourAngleOrder(self) -> None:
        result = _result(0, _bfov(yaw=0.0, pitch=-0.5, horizontal=1.0, vertical=0.25), valid=True)
        self.assertEqual(formatCompetitionResult(result), "0.000,-28.648,57.296,14.324")


def _bfov(*, yaw: float, pitch: float, horizontal: float, vertical: float) -> BFoV:
    return BFoV(
        center=makeSphericalPoint(yaw, pitch),
        horizontalFovRad=horizontal,
        verticalFovRad=vertical,
    )


def _result(frameIndex: int, bfov: BFoV, *, valid: bool) -> TrackResult:
    return TrackResult(
        sequenceId=SequenceId("sequence"),
        frameIndex=FrameIndex(frameIndex),
        bbox=BBoxXYWH(0.0, 0.0, 10.0, 10.0),
        bfov=bfov,
        confidence=0.9,
        status=TrackStatus.TRACKING if valid else TrackStatus.LOST,
        valid=valid,
        resultSource=ResultSource.OBSERVED_CONFIRMED,
    )


if __name__ == "__main__":
    unittest.main()
