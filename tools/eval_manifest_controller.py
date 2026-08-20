"""Evaluate the production Controller on one labeled manifest sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace
from math import pi
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import cv2
import numpy as np

from instatarget.app.driver import buildRuntime, closeBackend, runTracking
from instatarget.core.config import DEFAULT_VISUALIZATION_STAGES, VisualizationConfig, loadConfig
from instatarget.core.types import BBoxXYWH, FrameIndex, FramePacket, SequenceId, TrackResult
from instatarget.eval.otb_metrics import auc, bboxIoU, circularBBoxIoU, trackingLossRate
from instatarget.eval.profiler import RuntimeProfiler
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
    def __init__(self, truth: dict[int, ManifestRecord], visualRecorder: Any = None) -> None:
        self._truth = truth
        self._visualRecorder = visualRecorder
        self._local: dict[tuple[int, int], Any] = {}
        self._views: dict[tuple[int, int], Any] = {}
        self._rounds: dict[tuple[int, int], int] = {}
        self.viewCounts: Counter[int] = Counter()
        self.forwardCounts: Counter[int] = Counter()
        self.rows: list[dict[str, Any]] = []

    def recordLocalRgb(self, frame: FramePacket, views: Any) -> None:
        if self._visualRecorder is not None:
            self._visualRecorder.recordLocalRgb(frame, views)
        self.viewCounts[int(frame.frameIndex)] += len(views)
        self.forwardCounts[int(frame.frameIndex)] += 1
        roundIndex = self.forwardCounts[int(frame.frameIndex)]
        for view in views:
            self._views[(int(frame.frameIndex), view.spec.viewId)] = view.spec
            self._rounds[(int(frame.frameIndex), view.spec.viewId)] = roundIndex

    def recordBackendBoxes(self, frame: FramePacket, views: Any, observations: Any) -> None:
        if self._visualRecorder is not None:
            self._visualRecorder.recordBackendBoxes(frame, views, observations)
        for observation in observations:
            self._local[(int(frame.frameIndex), observation.viewId)] = observation

    def recordGeometryBoxes(self, frame: FramePacket, observations: Any) -> None:
        if self._visualRecorder is not None:
            self._visualRecorder.recordGeometryBoxes(frame, observations)
        record = self._truth[int(frame.frameIndex)]
        for observation in observations:
            local = self._local.pop((int(frame.frameIndex), observation.viewId))
            spec = self._views.pop((int(frame.frameIndex), observation.viewId))
            roundIndex = self._rounds.pop((int(frame.frameIndex), observation.viewId))
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
            localIoU = (
                bboxIoU(
                    local.bbox,
                    BBoxXYWH(
                        xPx=float(
                            (targetLocalBox[0] - targetLocalBox[2] / 2.0) * spec.outputWidthPx
                        ),
                        yPx=float(
                            (targetLocalBox[1] - targetLocalBox[3] / 2.0) * spec.outputHeightPx
                        ),
                        widthPx=float(targetLocalBox[2] * spec.outputWidthPx),
                        heightPx=float(targetLocalBox[3] * spec.outputHeightPx),
                    ),
                )
                if targetLocalBox is not None
                else 0.0
            )
            sphericalIoU = (
                bfovSphericalIoU(observation.bfov, record.bfov, samplesYaw=64, samplesPitch=32)
                if record.visible and record.bfov is not None
                else 0.0
            )
            self.rows.append(
                {
                    "sequenceId": record.sequenceId,
                    "split": record.split,
                    "frameIndex": int(frame.frameIndex),
                    "viewId": observation.viewId,
                    "roundIndex": roundIndex,
                    "visible": record.visible,
                    "targetPresent": targetLocalBox is not None,
                    "targetLocalBoxCxCyWh": (
                        targetLocalBox.tolist() if targetLocalBox is not None else None
                    ),
                    "circularErpIoU": iou,
                    "localIoU": localIoU,
                    "sphericalIoU": sphericalIoU,
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
                    "localBBox": asdict(local.bbox),
                    "projectedBBox": asdict(observation.bbox),
                    "projectedBFoV": asdict(observation.bfov),
                    "normalizedRadius": observation.normalizedRadius,
                    "edgeMargin": observation.edgeMargin,
                    "envelopeInflation": observation.envelopeInflation,
                    "viewYawDeg": float(np.degrees(spec.bfov.center.yawRad)),
                    "viewPitchDeg": float(np.degrees(spec.bfov.center.pitchRad)),
                    "viewHorizontalFovDeg": float(np.degrees(spec.bfov.horizontalFovRad)),
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
    parser.add_argument(
        "--visual-output-root",
        type=Path,
        help="Optional midVisual root; only backend_box and geometry_box are written.",
    )
    parser.add_argument(
        "--result-visual-root",
        type=Path,
        help="Optional root for one final ERP result image per frame.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "fp16"))
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--reuse-buffers", action="store_true")
    parser.add_argument("--pinned-nonblocking", action="store_true")
    parser.add_argument("--spherical-samples-yaw", type=int, default=128)
    parser.add_argument("--spherical-samples-pitch", type=int, default=64)
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument(
        "--uncalibrated-stage3",
        action="store_true",
        help="Use raw Stage 3 presence*quality for pre-calibration E01 only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for enabled, names in (
        (args.profile, ("INSTARGET_PROFILE",)),
        (args.cudnn_benchmark, ("INSTARGET_CUDNN_BENCHMARK",)),
        (args.channels_last, ("INSTARGET_CHANNELS_LAST",)),
        (args.reuse_buffers, ("INSTARGET_REUSE_BUFFERS",)),
        (
            args.pinned_nonblocking,
            ("INSTARGET_PINNED_MEMORY", "INSTARGET_NON_BLOCKING"),
        ),
    ):
        for name in names:
            os.environ[name] = "1" if enabled else "0"
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
        model=replace(
            appConfig.model,
            weights=args.weights.resolve(),
            precision=args.precision or appConfig.model.precision,
        ),
        scoring=(
            replace(appConfig.scoring, calibrationArtifact=None)
            if args.uncalibrated_stage3
            else appConfig.scoring
        ),
        visualization=replace(appConfig.visualization, enabled=False),
    )
    selected = records[: args.max_frames] if args.max_frames is not None else records
    truth = {record.frameIndex: record for record in selected}
    source = ManifestVideoSource(records, args.max_frames)
    sink = MemoryResultSink()
    timer = IntervalTimer()
    visualRecorder = None
    if args.visual_output_root is not None:
        from instatarget.visualization.recorder import VisualizationRecorder

        visualRecorder = VisualizationRecorder(
            VisualizationConfig(
                enabled=True,
                outputRoot=args.visual_output_root,
                stages=DEFAULT_VISUALIZATION_STAGES,
            )
        )
    recorder = CandidateRecorder(truth, visualRecorder)
    profiler = RuntimeProfiler(enabled=args.profile)
    resultVisualRecorder = None
    if args.result_visual_root is not None:
        from instatarget.visualization.result import ResultVisualizationRecorder

        resultVisualRecorder = ResultVisualizationRecorder(args.result_visual_root)
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
            resultRecorder=resultVisualRecorder,
            processingTimer=timer,
            profiler=profiler,
            scoreCalibration=runtime.scoreCalibration,
        )
        sink.finalize(count)
    finally:
        closeBackend(runtime.backend)
        source.close()

    if profiler.enabled:
        profiler.metadata.update(_runtimeProfileMetadata(profiler.frameRows))
    report = _summarize(
        args,
        selected,
        sink.results,
        timer,
        recorder,
        profiler,
        candidateMinScore=appConfig.tracking.candidateMinScore,
    )
    report["experiment"] = _experimentMetadata(args, appConfig)
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
    profiledRows = {int(row["frameIndex"]): row for row in profiler.frameRows}
    timingTemporary.write_text(
        "".join(
            json.dumps(
                {
                    "sequenceId": args.sequence,
                    "frameIndex": int(record.frameIndex),
                    "totalProcessingMs": interval / 1_000_000.0,
                    "viewCount": recorder.viewCounts[int(record.frameIndex)],
                    "forwardCount": recorder.forwardCounts[int(record.frameIndex)],
                    **profiledRows.get(int(record.frameIndex), {}),
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
    profiler: RuntimeProfiler,
    *,
    candidateMinScore: float,
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
    for offset, (record, result) in enumerate(zip(records[1:], results[1:], strict=True), 1):
        previous = records[offset - 1]
        speedDegPerSec = 0.0
        if record.bfov is not None and previous.bfov is not None:
            deltaSeconds = max(record.timestamp - previous.timestamp, 1e-9)
            speedDegPerSec = float(
                np.degrees(centerAngularErrorRad(record.bfov, previous.bfov)) / deltaSeconds
            )
        if not record.visible or record.bbox is None or record.bfov is None:
            absentCount += 1
            falsePositives += int(result.valid)
            frameRows.append(
                {
                    "sequenceId": record.sequenceId,
                    "frameIndex": record.frameIndex,
                    "visible": False,
                    "valid": result.valid,
                    "circularErpIoU": 0.0,
                    "sphericalIoU": 0.0,
                    "difficultType": record.difficultType,
                    "domain": "real" if "real" in record.sequenceId.lower() else "sim",
                    "occluded": record.occluded,
                    "truncated": record.truncated,
                }
            )
            continue
        erpIoU = circularBBoxIoU(result.bbox, record.bbox, record.width)
        erpIous.append(erpIoU)
        if not (0.0 < result.bfov.horizontalFovRad < pi and 0.0 < result.bfov.verticalFovRad < pi):
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
                "sequenceId": record.sequenceId,
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
                "targetSizeBand": _targetSizeBand(record),
                "difficultType": record.difficultType,
                "domain": "real" if "real" in record.sequenceId.lower() else "sim",
                "occluded": record.occluded,
                "truncated": record.truncated,
                "speedDegPerSec": speedDegPerSec,
                "speedBand": (
                    "fast"
                    if speedDegPerSec >= 90.0
                    else "medium"
                    if speedDegPerSec >= 30.0
                    else "slow"
                ),
                "horizontalFovDeg": float(np.degrees(record.bfov.horizontalFovRad)),
                "fovBand": _fovBand(float(np.degrees(record.bfov.horizontalFovRad))),
                "viewEdge": bool(
                    any(
                        row["frameIndex"] == record.frameIndex and row["edgeMargin"] <= 0.02
                        for row in candidates
                    )
                ),
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
            float(np.mean([result.valid for result in results[1:]])) if len(results) > 1 else 0.0
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
            float(np.mean([row["hitAt0.5"] for row in candidates])) if candidates else 0.0
        ),
        "sphericalSamples": [args.spherical_samples_yaw, args.spherical_samples_pitch],
        "profiler": profiler.summarize(),
        "profilerPerFrame": profiler.summarizeFrames(),
        "profilerMetadata": dict(profiler.metadata),
        "profilerEnabled": profiler.enabled,
        "lossEpisodes": _lossEpisodeReport(
            frameRows,
            candidates,
            candidateMinScore=candidateMinScore,
        ),
        "coverageStrata": _coverageStrata(frameRows),
        "geometryEvidence": _geometryEvidence(candidates),
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
                "bfov": asdict(result.bfov),
            }
            for result in results
        ],
        "frameMetrics": frameRows,
    }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


def _lossEpisodeReport(
    frameRows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    candidateMinScore: float,
) -> dict[str, Any]:
    byFrame: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        byFrame.setdefault(int(candidate["frameIndex"]), []).append(candidate)
    lost = [row for row in frameRows if row["visible"] and float(row["circularErpIoU"]) <= 1e-12]
    episodes: list[list[dict[str, Any]]] = []
    for row in lost:
        if not episodes or int(row["frameIndex"]) != int(episodes[-1][-1]["frameIndex"]) + 1:
            episodes.append([])
        episodes[-1].append(row)
    frameMetrics = {int(row["frameIndex"]): row for row in frameRows}
    details = []
    for episode in episodes:
        first = episode[0]
        firstFrame = int(first["frameIndex"])
        rows = byFrame.get(firstFrame, [])
        round1 = [row for row in rows if int(row.get("roundIndex", 0)) == 1]
        round2 = [row for row in rows if int(row.get("roundIndex", 0)) == 2]
        details.append(
            {
                "startFrame": int(first["frameIndex"]),
                "endFrame": int(episode[-1]["frameIndex"]),
                "length": len(episode),
                "firstCenterErrorDeg": first.get("centerErrorDeg"),
                "firstWidthRelativeError": first.get("widthRelativeError"),
                "firstHeightRelativeError": first.get("heightRelativeError"),
                "round1Covered": any(float(row["circularErpIoU"]) > 0.0 for row in round1),
                "round2Covered": any(float(row["circularErpIoU"]) > 0.0 for row in round2),
                "maxPresence": max(
                    (float(row["presenceProbability"]) for row in rows), default=0.0
                ),
                "maxQuality": max((float(row["qualityProbability"]) for row in rows), default=0.0),
                "maxMotion": max((float(row["motionProbability"]) for row in rows), default=0.0),
                "maxSingleScore": max((float(row["singleScore"]) for row in rows), default=0.0),
                "preLossTrajectory": [
                    _signalRow(
                        frameIndex,
                        frameMetrics.get(frameIndex),
                        byFrame.get(frameIndex, []),
                    )
                    for frameIndex in range(max(1, firstFrame - 5), firstFrame + 1)
                ],
            }
        )
    lengths = [item["length"] for item in details]
    shadowTriggers: list[int] = []
    lowRun = 0
    for row in sorted(frameRows, key=lambda item: int(item["frameIndex"])):
        candidatesForFrame = byFrame.get(int(row["frameIndex"]), [])
        maxScore = max((float(item["singleScore"]) for item in candidatesForFrame), default=0.0)
        lowRun = lowRun + 1 if maxScore < candidateMinScore else 0
        if lowRun == 2:
            shadowTriggers.append(int(row["frameIndex"]))
    matchedEpisodes = 0
    trueTriggers = 0
    for trigger in shadowTriggers:
        matching = [
            episode
            for episode in details
            if int(episode["startFrame"]) - 2 <= trigger <= int(episode["endFrame"])
        ]
        trueTriggers += bool(matching)
    for episode in details:
        if any(
            int(episode["startFrame"]) - 2 <= trigger <= int(episode["endFrame"])
            for trigger in shadowTriggers
        ):
            matchedEpisodes += 1
    return {
        "count": len(details),
        "lengthP50": _percentile(lengths, 50),
        "lengthP95": _percentile(lengths, 95),
        "maxLength": max(lengths, default=0),
        "shadowLostCandidateTriggers": len(shadowTriggers),
        "shadowLostCandidatePrecision": trueTriggers / len(shadowTriggers)
        if shadowTriggers
        else 0.0,
        "shadowLostCandidateRecall": matchedEpisodes / len(details) if details else 0.0,
        "shadowLostCandidateThreshold": candidateMinScore,
        "shadowLostCandidateFrames": shadowTriggers,
        "episodes": details,
    }


def _coverageStrata(frameRows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = (
        "domain",
        "smallTarget",
        "targetSizeBand",
        "speedBand",
        "occluded",
        "seam",
        "highLatitude",
        "difficultType",
        "viewEdge",
        "fovBand",
    )
    result: dict[str, Any] = {}
    visible = [row for row in frameRows if row["visible"]]
    for dimension in dimensions:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in visible:
            groups.setdefault(str(row.get(dimension)), []).append(row)
        result[dimension] = {
            key: {
                "frameCount": len(rows),
                "meanIoU": float(np.mean([float(row["circularErpIoU"]) for row in rows])),
                "lossRate": float(np.mean([float(row["circularErpIoU"]) <= 1e-12 for row in rows])),
            }
            for key, rows in groups.items()
        }
    return result


def _geometryEvidence(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    visible = [row for row in candidates if row.get("targetPresent")]
    inflation = [float(row["envelopeInflation"]) for row in visible]
    local = [float(row.get("localIoU", 0.0)) for row in visible]
    circular = [float(row["circularErpIoU"]) for row in visible]
    spherical = [float(row.get("sphericalIoU", 0.0)) for row in visible]
    return {
        "sampleCount": len(visible),
        "localIoUMean": _mean(local),
        "circularErpIoUMean": _mean(circular),
        "sphericalIoUMean": _mean(spherical),
        "inflationCorrelationWithLocalToErpGap": _correlation(
            inflation, [a - b for a, b in zip(local, circular, strict=True)]
        ),
        "inflationCorrelationWithLocalToSphericalGap": _correlation(
            inflation, [a - b for a, b in zip(local, spherical, strict=True)]
        ),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) < 2 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _percentileDegrees(values: list[float], percentile: float) -> float:
    return float(np.degrees(_percentile(values, percentile)))


def _signalRow(
    frameIndex: int,
    metric: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "frameIndex": frameIndex,
        "visible": bool(metric and metric.get("visible")),
        "circularErpIoU": float(metric.get("circularErpIoU", 0.0)) if metric else None,
        "maxPresence": max(
            (float(row["presenceProbability"]) for row in candidates), default=0.0
        ),
        "maxQuality": max((float(row["qualityProbability"]) for row in candidates), default=0.0),
        "maxMotion": max((float(row["motionProbability"]) for row in candidates), default=0.0),
        "maxSingleScore": max((float(row["singleScore"]) for row in candidates), default=0.0),
    }


def _targetSizeBand(record: ManifestRecord) -> str:
    if record.bbox is None:
        return "absent"
    ratio = record.bbox.widthPx * record.bbox.heightPx / (record.width * record.height)
    if ratio <= 0.01:
        return "small"
    if ratio <= 0.05:
        return "medium"
    return "large"


def _fovBand(horizontalFovDeg: float) -> str:
    if horizontalFovDeg <= 30.0:
        return "narrow"
    if horizontalFovDeg <= 75.0:
        return "medium"
    return "wide"


def _runtimeProfileMetadata(frameRows: list[dict[str, Any]]) -> dict[str, Any]:
    backendRows = [
        batch
        for row in frameRows
        for batch in row.get("backendBatches", [])
        if isinstance(batch, dict)
    ]
    metadata: dict[str, Any] = {
        "oomCount": max((int(row.get("oomCount", 0)) for row in backendRows), default=0),
        "fp16FallbackCount": max(
            (int(row.get("fp16FallbackCount", 0)) for row in backendRows),
            default=0,
        ),
    }
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            metadata.update(
                {
                    "device": torch.cuda.get_device_name(device),
                    "cudaVersion": torch.version.cuda,
                    "torchVersion": torch.__version__,
                    "maxMemoryAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
                    "maxMemoryReservedBytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
    except Exception as error:
        metadata["cudaMetadataError"] = str(error)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        metadata["gpuTemperatureC"] = float(result.stdout.splitlines()[0].strip())
    except Exception as error:
        metadata["temperatureError"] = str(error)
    return metadata


def _experimentMetadata(args: argparse.Namespace, appConfig: Any) -> dict[str, Any]:
    calibration = appConfig.scoring.calibrationArtifact
    paths = {
        "config": args.config.expanduser().resolve(),
        "checkpoint": args.weights.expanduser().resolve(),
        "calibration": calibration,
    }
    return {
        "precision": appConfig.model.precision,
        "optimizations": {
            "cudnnBenchmark": bool(args.cudnn_benchmark),
            "channelsLast": bool(args.channels_last),
            "reuseBuffers": bool(args.reuse_buffers),
            "pinnedNonBlocking": bool(args.pinned_nonblocking),
        },
        "profilerEnabled": bool(args.profile),
        "artifacts": {
            name: {
                "path": str(path) if path is not None else None,
                "sha256": _sha256(path) if path is not None and path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "gitCommit": _gitCommit(args.config.expanduser().resolve().parents[1]),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gitCommit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
