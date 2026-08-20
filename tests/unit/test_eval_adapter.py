import unittest

from instatarget.adapters.competition_adapter import CompetitionAdapter
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    SequenceId,
    SphericalPoint,
    TrackResult,
    TrackStatus,
)
from instatarget.eval.otb_metrics import OtbMetrics, bboxIoU, trackingLossRate


class EvalAdapterTest(unittest.TestCase):
    def testCompetitionAdapterSplitsAcrossSeam(self) -> None:
        adapter = CompetitionAdapter(frameWidthPx=100, frameHeightPx=50, strategy="split")
        result = _result(BBoxXYWH(92.0, 4.0, 16.0, 10.0))

        boxes = adapter.adaptResult(result)

        self.assertEqual(len(boxes), 2)
        self.assertAlmostEqual(sum(box.widthPx for box in boxes), 16.0)

    def testOtbMetricsSummarizeSuccessRate(self) -> None:
        metrics = OtbMetrics()
        metrics.update(BBoxXYWH(0.0, 0.0, 10.0, 10.0), BBoxXYWH(0.0, 0.0, 10.0, 10.0))
        metrics.update(BBoxXYWH(0.0, 0.0, 5.0, 5.0), BBoxXYWH(0.0, 0.0, 10.0, 10.0))

        summary = metrics.summarize()

        self.assertAlmostEqual(summary["successRate@0.5"], 0.5)
        self.assertEqual(summary["lostFrameCount"], 0)
        self.assertEqual(summary["trackingLossRate"], 0.0)
        self.assertAlmostEqual(
            bboxIoU(
                BBoxXYWH(0.0, 0.0, 10.0, 10.0),
                BBoxXYWH(0.0, 0.0, 10.0, 10.0),
            ),
            1.0,
        )

    def testTrackingLossRateCountsOnlyZeroOverlapVisibleSamples(self) -> None:
        self.assertAlmostEqual(trackingLossRate([0.0, 0.2, 0.0, 1.0]), 0.5)
        self.assertEqual(trackingLossRate([]), 0.0)

        metrics = OtbMetrics(ious=[0.0, 0.2, 0.0, 1.0])
        summary = metrics.summarize()
        self.assertEqual(summary["lostFrameCount"], 2)
        self.assertAlmostEqual(summary["trackingLossRate"], 0.5)


def _result(bbox: BBoxXYWH) -> TrackResult:
    return TrackResult(
        sequenceId=SequenceId("sequence"),
        frameIndex=FrameIndex(0),
        bbox=bbox,
        bfov=BFoV(
            center=SphericalPoint(1.0, 0.0, 0.0, 0.0, 0.0),
            horizontalFovRad=0.5,
            verticalFovRad=0.5,
        ),
        confidence=0.9,
        status=TrackStatus.TRACKING,
        valid=True,
    )
