"""Evaluate the production Controller on one labeled manifest sequence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from math import pi
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import cv2
import numpy as np

from instatarget.app.driver import buildRuntime, closeRuntime, runTracking
from instatarget.core.config import loadConfig
from instatarget.core.types import FrameIndex, FramePacket, SequenceId, TrackResult
from instatarget.eval.otb_metrics import auc, circularBBoxIoU, trackingLossRate
from instatarget.eval.spherical_metrics import bfovSphericalIoU, centerAngularErrorRad
from instatarget.training.dataset import (
    ManifestRecord,
    _bfovToNormalizedLocal,
    _erpBoxToNormalizedLocal,
    loadManifest,
)


class ManifestVideoSource:
    def __init__(self, records: tuple[ManifestRecord, ...], maxFrames: int | None) -> None:
        self._records = records[:maxFrames] if maxFrames is not None else records
        self._capture: Any = None
        self._offset = 0

    def open(self, uri: str) -> None:
        del uri
        self._capture = cv2.VideoCapture(str(self._records[0].videoPath))
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open video: {self._records[0].videoPath}")

    def read(self) -> FramePacket | None:
        if self._offset >= len(self._records):
            return None
        record = self._records[self._offset]
        if int(self._capture.get(cv2.CAP_PROP_POS_FRAMES)) != record.frameIndex:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, record.frameIndex)
        ok, bgr = self._capture.read()
        if not ok or bgr is None:
            raise RuntimeError(f"cannot decode frame {record.frameIndex}")
        self._offset += 1
        return FramePacket(
            sequenceId=SequenceId(record.sequenceId),
            frameIndex=FrameIndex(record.frameIndex),
            timestampNs=int(round(record.timestamp * 1_000_000_000.0)),
            rgb=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
        )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class MemoryResultSink:
    def __init__(self) -> None:
        self.results: list[TrackResult] = []

    def open(self, destination: str) -> None:
        del destination

    def write(self, result: TrackResult) -> None:
        self.results.append(result)

    def finalize(self, expectedFrameCount: int) -> None:
        if len(self.results) != expectedFrameCount:
            raise RuntimeError("result count mismatch")

    def close(self) -> None:
        pass


class IntervalTimer:
    def __init__(self) -> None:
        self._started: int | None = None
        self.intervalsNs: list[int] = []

    def startProcessing(self) -> None:
        if self._started is not None:
            raise RuntimeError("processing interval already started")
        self._started = perf_counter_ns()

    def stopProcessing(self) -> None:
        if self._started is None:
            raise RuntimeError("processing interval not started")
        self.intervalsNs.append(max(0, perf_counter_ns() - self._started))
        self._started = None


class CandidateRecorder:
    def __init__(self, truth: dict[int, ManifestRecord]) -> None:
        self._truth = truth
        self._local: dict[tuple[int, int], Any] = {}
        self._views: dict[tuple[int, int], Any] = {}
        self.viewCounts: Counter[int] = Counter()
        self.forwardCounts: Counter[int] = Counter()
        self.rows: list[dict[str, Any]] = []

    def recordLocalRgb(self, frame: FramePacket, views: Any) -> None:
        self.viewCounts[int(frame.frameIndex)] += len(views)
        self.forwardCounts[int(frame.frameIndex)] += 1
        for view in views:
            self._views[(int(frame.frameIndex), view.spec.viewId)] = view.spec

    def recordBackendBoxes(self, frame: FramePacket, views: Any, observations: Any) -> None:
        del views
        for observation in observations:
            self._local[(int(frame.frameIndex), observation.viewId)] = observation

    def recordGeometryBoxes(self, frame: FramePacket, observations: Any) -> None:
        record = self._truth[int(frame.frameIndex)]
        for observation in observations:
            local = self._local.pop((int(frame.frameIndex), observation.viewId))
            spec = self._views.pop((int(frame.frameIndex), observation.viewId))
            targetLocalBox = None
            if record.visible:
                targetLocalBox = (
                    _bfovToNormalizedLocal(record.bfov, spec)
                    if record.bfov is not None
                    else _erpBoxToNormalizedLocal(
                        record.bbox,
                        spec,
                        record.width,
                        record.height,
                    )
                )
            iou = (
                circularBBoxIoU(observation.bbox, record.bbox, record.width)
                if record.visible and record.bbox is not None
                else 0.0
            )
            self.rows.append(
                {
                    "sequenceId": record.sequenceId,
                    "split": record.split,
                    "frameIndex": int(frame.frameIndex),
                    "viewId": observation.viewId,
                    "visible": record.visible,
                    "targetPresent": targetLocalBox is not None,
                    "targetLocalBoxCxCyWh": (
                        targetLocalBox.tolist() if targetLocalBox is not None else None
                    ),
                    "circularErpIoU": iou,
                    "hitAt0.5": bool(record.visible and iou >= 0.5),
                    "modelScore": local.modelScore,
                    "presenceLogit": local.presenceLogit,
                    "qualityLogit": local.qualityLogit,
                    "presenceProbability": local.presenceProbability,
                    "qualityProbability": local.qualityProbability,
                    "predictedIoU": local.predictedIoU,
                    "cornerScore": local.cornerScore,
                    "appearanceProbability": observation.appearanceProbability,
                    "motionProbability": observation.motionProbability,
                    "singleScore": observation.singleScore,
                    "normalizedRadius": observation.normalizedRadius,
                    "edgeMargin": observation.edgeMargin,
                    "envelopeInflation": observation.envelopeInflation,
                    "viewYawDeg": float(np.degrees(spec.bfov.center.yawRad)),
                    "viewPitchDeg": float(np.degrees(spec.bfov.center.pitchRad)),
                    "viewHorizontalFovDeg": float(
                        np.degrees(spec.bfov.horizontalFovRad)
                    ),
                    "viewVerticalFovDeg": float(np.degrees(spec.bfov.verticalFovRad)),
                    "targetPitchDeg": (
                        float(np.degrees(record.bfov.center.pitchRad))
                        if record.bfov is not None
                        else None
                    ),
                    "difficultType": record.difficultType,
                }
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:\NewDownload\train"),
        help="Canonical labeled dataset root; manifest and every video must be inside it.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "validation", "calibration", "holdout"),
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--spherical-samples-yaw", type=int, default=128)
    parser.add_argument("--spherical-samples-pitch", type=int, default=64)
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument(
        "--uncalibrated-stage3",
        action="store_true",
        help="Use raw Stage 3 presence*quality for pre-calibration E01 only.",
    )
    parser.add_argument(
        "--fusion-strategy",
        choices=("legacy", "presence_quality", "geometric_consensus", "weighted_box"),
        default=None,
        help="Override evaluator.fusionStrategy for this run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.split == "holdout" and not args.allow_holdout:
        raise RuntimeError("holdout evaluation requires --allow-holdout after model freeze")
    datasetRoot = args.dataset_root.expanduser().resolve()
    manifestPath = args.manifest.expanduser().resolve()
    if not manifestPath.is_relative_to(datasetRoot):
        raise RuntimeError(
            f"manifest must be inside the canonical dataset root {datasetRoot}: {manifestPath}"
        )
    records = tuple(
        record
        for record in loadManifest(manifestPath)
        if record.split == args.split and record.sequenceId == args.sequence
    )
    if not records:
        raise RuntimeError(f"sequence is not present in split: {args.sequence}")
    outside = sorted(
        {
            str(record.videoPath)
            for record in records
            if not record.videoPath.is_relative_to(datasetRoot)
        }
    )
    if outside:
        raise RuntimeError(f"manifest video is outside canonical dataset root: {outside[0]}")
    records = tuple(sorted(records, key=lambda item: item.frameIndex))
    if records[0].frameIndex != 0 or not records[0].visible or records[0].bbox is None:
        raise RuntimeError("sequence requires a visible frame-0 initialization")
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")

    appConfig = loadConfig(args.config)
    appConfig = replace(
        appConfig,
        model=replace(appConfig.model, weights=args.weights.resolve()),
        scoring=(
            replace(appConfig.scoring, calibrationArtifact=None)
            if args.uncalibrated_stage3
            else appConfig.scoring
        ),
        visualization=replace(appConfig.visualization, enabled=False),
    )
    if args.fusion_strategy is not None:
        appConfig = replace(
            appConfig,
            evaluator=replace(appConfig.evaluator, fusionStrategy=args.fusion_strategy),
        )
    selected = records[: args.max_frames] if args.max_frames is not None else records
    truth = {record.frameIndex: record for record in selected}
    source = ManifestVideoSource(records, args.max_frames)
    sink = MemoryResultSink()
    timer = IntervalTimer()
    recorder = CandidateRecorder(truth)
    runtime = buildRuntime(
        appConfig,
        allowUncalibratedScoring=args.uncalibrated_stage3,
    )
    source.open("")
    try:
        count = runTracking(
            source=source,
            initialBox=records[0].bbox,
            geometry=runtime.geometry,
            controller=runtime.controller,
            backend=runtime.backend,
            sink=sink,
            recorder=recorder,
            processingTimer=timer,
            scoreCalibration=runtime.scoreCalibration,
        )
        sink.finalize(count)
    finally:
        closeRuntime(runtime)
        source.close()

    report = _summarize(args, selected, sink.results, timer, recorder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    candidates = args.output.with_name(f"{args.output.stem}.candidates.jsonl")
    candidateTemporary = candidates.with_suffix(candidates.suffix + ".tmp")
    candidateTemporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in recorder.rows),
        encoding="utf-8",
    )
    candidateTemporary.replace(candidates)
    timings = args.output.with_name(f"{args.output.stem}.timings.jsonl")
    timingTemporary = timings.with_suffix(timings.suffix + ".tmp")
    timingTemporary.write_text(
        "".join(
            json.dumps(
                {
                    "sequenceId": args.sequence,
                    "frameIndex": int(record.frameIndex),
                    "totalProcessingMs": interval / 1_000_000.0,
                    "viewCount": recorder.viewCounts[int(record.frameIndex)],
                    "forwardCount": recorder.forwardCounts[int(record.frameIndex)],
                },
                separators=(",", ":"),
            )
            + "\n"
            for record, interval in zip(selected, timer.intervalsNs[:-1], strict=True)
        ),
        encoding="utf-8",
    )
    timingTemporary.replace(timings)
    print(json.dumps(report["summary"], indent=2))
    return 0


def _summarize(
    args: argparse.Namespace,
    records: tuple[ManifestRecord, ...],
    results: list[TrackResult],
    timer: IntervalTimer,
    recorder: CandidateRecorder,
) -> dict[str, Any]:
    candidates = recorder.rows
    erpIous: list[float] = []
    sphericalIous: list[float] = []
    centerErrors: list[float] = []
    widthErrors: list[float] = []
    heightErrors: list[float] = []
    absentCount = falsePositives = 0
    invalidSphericalPredictions = 0
    frameRows: list[dict[str, Any]] = []
    for record, result in zip(records[1:], results[1:], strict=True):
        if not record.visible or record.bbox is None or record.bfov is None:
            absentCount += 1
            falsePositives += int(result.valid)
            frameRows.append(
                {
                    "frameIndex": record.frameIndex,
                    "visible": False,
                    "valid": result.valid,
                    "circularErpIoU": 0.0,
                    "sphericalIoU": 0.0,
                }
            )
            continue
        erpIoU = circularBBoxIoU(result.bbox, record.bbox, record.width)
        erpIous.append(erpIoU)
        if not (
            0.0 < result.bfov.horizontalFovRad < pi
            and 0.0 < result.bfov.verticalFovRad < pi
        ):
            sphericalIous.append(0.0)
            invalidSphericalPredictions += 1
        else:
            sphericalIous.append(
                bfovSphericalIoU(
                    result.bfov,
                    record.bfov,
                    samplesYaw=args.spherical_samples_yaw,
                    samplesPitch=args.spherical_samples_pitch,
                )
            )
        centerErrors.append(centerAngularErrorRad(result.bfov, record.bfov))
        widthErrors.append(abs(result.bfov.horizontalFovRad / record.bfov.horizontalFovRad - 1.0))
        heightErrors.append(abs(result.bfov.verticalFovRad / record.bfov.verticalFovRad - 1.0))
        frameRows.append(
            {
                "frameIndex": record.frameIndex,
                "visible": True,
                "valid": result.valid,
                "circularErpIoU": erpIoU,
                "sphericalIoU": sphericalIous[-1],
                "centerErrorDeg": float(np.degrees(centerErrors[-1])),
                "widthRelativeError": widthErrors[-1],
                "heightRelativeError": heightErrors[-1],
                "seam": bool(record.bbox.xPx + record.bbox.widthPx > record.width),
                "highLatitude": bool(abs(record.bfov.center.pitchRad) >= pi / 3.0),
                "smallTarget": bool(
                    record.bbox.widthPx * record.bbox.heightPx
                    <= 0.01 * record.width * record.height
                ),
                "difficultType": record.difficultType,
            }
        )
    frameIntervals = timer.intervalsNs[1:-1]
    elapsedMs = np.asarray(frameIntervals, dtype=np.float64) / 1_000_000.0
    values = np.asarray(erpIous, dtype=np.float64)
    summary = {
        "sequence": args.sequence,
        "split": args.split,
        "weights": str(args.weights.resolve()),
        "frameCount": len(results),
        "evaluatedVisibleFrames": len(erpIous),
        "absentFrames": absentCount,
        "circularErpMeanIoU": float(values.mean()) if values.size else 0.0,
        "successAUC": auc(erpIous),
        "successRateAt0.5": float((values > 0.5).mean()) if values.size else 0.0,
        "lostFrameCount": int(np.sum(values <= 1e-12)) if values.size else 0,
        "trackingLossRate": trackingLossRate(erpIous),
        "sphericalMeanIoU": float(np.mean(sphericalIous)) if sphericalIous else 0.0,
        "meanCenterErrorDeg": float(np.degrees(np.mean(centerErrors))) if centerErrors else 0.0,
        "centerErrorP50Deg": _percentileDegrees(centerErrors, 50),
        "centerErrorP95Deg": _percentileDegrees(centerErrors, 95),
        "meanWidthRelativeError": float(np.mean(widthErrors)) if widthErrors else 0.0,
        "meanHeightRelativeError": float(np.mean(heightErrors)) if heightErrors else 0.0,
        "widthRelativeErrorP50": _percentile(widthErrors, 50),
        "widthRelativeErrorP95": _percentile(widthErrors, 95),
        "heightRelativeErrorP50": _percentile(heightErrors, 50),
        "heightRelativeErrorP95": _percentile(heightErrors, 95),
        "invalidSphericalPredictions": invalidSphericalPredictions,
        "absentFalsePositiveRate": falsePositives / absentCount if absentCount else 0.0,
        "validRate": (
            float(np.mean([result.valid for result in results[1:]]))
            if len(results) > 1
            else 0.0
        ),
        "statusCounts": dict(Counter(result.status.value for result in results[1:])),
        "resultSourceCounts": dict(Counter(result.resultSource.value for result in results[1:])),
        "latencyP50Ms": float(np.percentile(elapsedMs, 50)) if elapsedMs.size else 0.0,
        "latencyP95Ms": float(np.percentile(elapsedMs, 95)) if elapsedMs.size else 0.0,
        "latencyP99Ms": float(np.percentile(elapsedMs, 99)) if elapsedMs.size else 0.0,
        "candidateCount": len(candidates),
        "averageViewsPerFrame": (
            float(np.mean([recorder.viewCounts[int(record.frameIndex)] for record in records[1:]]))
            if len(records) > 1
            else 0.0
        ),
        "averageForwardsPerFrame": (
            float(
                np.mean([recorder.forwardCounts[int(record.frameIndex)] for record in records[1:]])
            )
            if len(records) > 1
            else 0.0
        ),
        "candidateHitRateAt0.5": (
            float(np.mean([row["hitAt0.5"] for row in candidates]))
            if candidates
            else 0.0
        ),
        "sphericalSamples": [args.spherical_samples_yaw, args.spherical_samples_pitch],
    }
    return {
        "format": "instatarget.manifest-controller-eval.v1",
        "summary": summary,
        "frames": [
            {
                "frameIndex": int(result.frameIndex),
                "valid": result.valid,
                "confidence": result.confidence,
                "status": result.status.value,
                "resultSource": result.resultSource.value,
                "bbox": asdict(result.bbox),
            }
            for result in results
        ],
        "frameMetrics": frameRows,
    }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


def _percentileDegrees(values: list[float], percentile: float) -> float:
    return float(np.degrees(_percentile(values, percentile)))


if __name__ == "__main__":
    raise SystemExit(main())
