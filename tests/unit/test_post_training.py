import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from instatarget.eval.profiler import RuntimeProfiler
from instatarget.tracker.pytorch_hit_session import _outputsAreFinite
from tools.compare_evaluation_ab import main as compareEvaluations
from tools.eval_manifest_controller import _lossEpisodeReport
from tools.verify_release_artifacts import _verifyCalibrationPair


class RuntimeProfilerTest(unittest.TestCase):
    def testDisabledProfilerCollectsNothing(self) -> None:
        profiler = RuntimeProfiler(enabled=False)

        with profiler.track("crop"):
            pass
        profiler.startFrame(1)
        profiler.record("crop", 1_000_000)
        profiler.finishFrame()

        self.assertEqual(profiler.stats, {})
        self.assertEqual(profiler.frameRows, [])
        self.assertEqual(profiler.summarizeFrames(), {})

    def testPerFrameSummaryAggregatesRepeatedStageCalls(self) -> None:
        profiler = RuntimeProfiler()
        profiler.startFrame(1)
        profiler.record("crop", 1_000_000)
        profiler.record("crop", 2_000_000)
        profiler.finishFrame()
        profiler.startFrame(2)
        profiler.record("crop", 5_000_000)
        profiler.finishFrame()

        summary = profiler.summarizeFrames()["crop"]

        self.assertEqual(summary["count"], 2.0)
        self.assertEqual(summary["meanNs"], 4_000_000.0)
        self.assertEqual(summary["p50Ns"], 4_000_000.0)


class HiTFiniteOutputTest(unittest.TestCase):
    def testCheckCoversEveryFp16FallbackOutput(self) -> None:
        import torch

        output = {
            name: torch.ones(1)
            for name in (
                "predBoxes",
                "cornerHeatmapTl",
                "cornerHeatmapBr",
                "presenceLogit",
                "qualityLogit",
                "presenceProbability",
                "qualityProbability",
            )
        }

        self.assertTrue(_outputsAreFinite(output, torch))
        output["qualityLogit"] = torch.tensor([float("nan")])
        self.assertFalse(_outputsAreFinite(output, torch))


class LossEpisodeReportTest(unittest.TestCase):
    def testReportIncludesPreLossSignalsAndShadowAccuracy(self) -> None:
        frameRows = [
            {"frameIndex": 1, "visible": True, "circularErpIoU": 0.4},
            {"frameIndex": 2, "visible": True, "circularErpIoU": 0.2},
            {
                "frameIndex": 3,
                "visible": True,
                "circularErpIoU": 0.0,
                "centerErrorDeg": 10.0,
                "widthRelativeError": 0.2,
                "heightRelativeError": 0.3,
            },
            {"frameIndex": 4, "visible": True, "circularErpIoU": 0.0},
            {"frameIndex": 5, "visible": True, "circularErpIoU": 0.5},
        ]
        candidates = [
            _candidate(frameIndex=1, score=0.8, iou=0.4),
            _candidate(frameIndex=2, score=0.5, iou=0.2),
            _candidate(frameIndex=3, score=0.4, iou=0.0),
            _candidate(frameIndex=4, score=0.3, iou=0.0),
            _candidate(frameIndex=5, score=0.8, iou=0.5),
        ]

        report = _lossEpisodeReport(frameRows, candidates, candidateMinScore=0.6)

        self.assertEqual(report["count"], 1)
        self.assertEqual(report["maxLength"], 2)
        self.assertEqual(report["shadowLostCandidateFrames"], [3])
        self.assertEqual(report["shadowLostCandidatePrecision"], 1.0)
        self.assertEqual(report["shadowLostCandidateRecall"], 1.0)
        self.assertEqual(
            [row["frameIndex"] for row in report["episodes"][0]["preLossTrajectory"]],
            [1, 2, 3],
        )


class EvaluationComparisonTest(unittest.TestCase):
    def testExactComparisonAlignsCandidatesByStableKey(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            report = _report()
            baseline.write_text(json.dumps(report), encoding="utf-8")
            candidate.write_text(json.dumps(report), encoding="utf-8")
            rows = [
                _comparisonCandidate(frameIndex=1, roundIndex=1, viewId=1),
                _comparisonCandidate(frameIndex=1, roundIndex=2, viewId=5),
            ]
            _writeCandidates(baseline, rows)
            _writeCandidates(candidate, list(reversed(rows)))

            code = compareEvaluations(
                ["--baseline", str(baseline), "--candidate", str(candidate)]
            )

            self.assertEqual(code, 0)


class ReleaseArtifactTest(unittest.TestCase):
    def testCalibrationPairIsBoundToCheckpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pth"
            calibration = root / "model.calibration.json"
            checkpoint.write_bytes(b"checkpoint")
            calibration.write_text(
                json.dumps(
                    {"checkpointSha256": hashlib.sha256(b"checkpoint").hexdigest()}
                ),
                encoding="utf-8",
            )

            _verifyCalibrationPair(checkpoint, calibration, required=True)

            calibration.write_text(
                json.dumps({"checkpointSha256": "bad"}),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _verifyCalibrationPair(checkpoint, calibration, required=True)


def _candidate(*, frameIndex: int, score: float, iou: float) -> dict[str, float | int]:
    return {
        "frameIndex": frameIndex,
        "roundIndex": 1,
        "circularErpIoU": iou,
        "presenceProbability": score,
        "qualityProbability": score,
        "motionProbability": score,
        "singleScore": score,
    }


def _comparisonCandidate(*, frameIndex: int, roundIndex: int, viewId: int) -> dict:
    box = {"xPx": 1.0, "yPx": 2.0, "widthPx": 3.0, "heightPx": 4.0}
    bfov = {
        "center": {"x": 1.0, "y": 0.0, "z": 0.0, "yawRad": 0.0, "pitchRad": 0.0},
        "horizontalFovRad": 0.5,
        "verticalFovRad": 0.4,
        "rollRad": 0.0,
    }
    return {
        "frameIndex": frameIndex,
        "roundIndex": roundIndex,
        "viewId": viewId,
        **{name: 0.5 for name in (
            "modelScore",
            "presenceLogit",
            "qualityLogit",
            "presenceProbability",
            "qualityProbability",
            "predictedIoU",
            "cornerScore",
            "appearanceProbability",
            "motionProbability",
            "singleScore",
        )},
        "localBBox": box,
        "projectedBBox": box,
        "projectedBFoV": bfov,
    }


def _report() -> dict:
    return {
        "summary": {
            "sequence": "validation/sequence",
            "split": "validation",
            "frameCount": 2,
            "circularErpMeanIoU": 0.5,
            "successRateAt0.5": 0.5,
            "trackingLossRate": 0.0,
            "absentFalsePositiveRate": 0.0,
            "latencyP95Ms": 100.0,
        },
        "frames": [
            {
                "frameIndex": 0,
                "valid": True,
                "confidence": 1.0,
                "status": 1,
                "resultSource": 1,
                "bbox": {"xPx": 1.0, "yPx": 2.0, "widthPx": 3.0, "heightPx": 4.0},
            },
            {
                "frameIndex": 1,
                "valid": True,
                "confidence": 0.5,
                "status": 1,
                "resultSource": 2,
                "bbox": {"xPx": 2.0, "yPx": 3.0, "widthPx": 4.0, "heightPx": 5.0},
            },
        ],
    }


def _writeCandidates(report: Path, rows: list[dict]) -> None:
    path = report.with_name(f"{report.stem}.candidates.jsonl")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
