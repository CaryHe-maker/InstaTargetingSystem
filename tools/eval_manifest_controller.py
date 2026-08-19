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

from instatarget.app.driver import buildRuntime, closeBackend, runTracking
from instatarget.core.config import loadConfig
from instatarget.core.types import FrameIndex, FramePacket, SequenceId, TrackResult
from instatarget.eval.otb_metrics import circularBBoxIoU
from instatarget.eval.spherical_metrics import bfovSphericalIoU, centerAngularErrorRad
from instatarget.training.dataset import ManifestRecord, loadManifest


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
        self.rows: list[dict[str, Any]] = []

    def recordLocalRgb(self, frame: FramePacket, views: Any) -> None:
        del frame, views

    def recordBackendBoxes(self, frame: FramePacket, views: Any, observations: Any) -> None:
        del views
        for observation in observations:
            self._local[(int(frame.frameIndex), observation.viewId)] = observation

    def recordGeometryBoxes(self, frame: FramePacket, observations: Any) -> None:
        record = self._truth[int(frame.frameIndex)]
        for observation in observations:
            local = self._local.pop((int(frame.frameIndex), observation.viewId))
            iou = (
                circularBBoxIoU(observation.bbox, record.bbox, record.width)
                if record.visible and record.bbox is not None
                else 0.0
            )
            self.rows.append(
                {
                    "frameIndex": int(frame.frameIndex),
                    "viewId": observation.viewId,
                    "visible": record.visible,
                    "circularErpIoU": iou,
                    "hitAt0.5": bool(record.visible and iou >= 0.5),
                    "modelScore": local.modelScore,
                    "presenceLogit": local.presenceLogit,
                    "qualityLogit": local.qualityLogit,
                    "presenceProbability": local.presenceProbability,
                    "qualityProbability": local.qualityProbability,
                    "cornerScore": local.cornerScore,
                    "appearanceProbability": observation.appearanceProbability,
                    "motionProbability": observation.motionProbability,
                    "singleScore": observation.singleScore,
                    "normalizedRadius": observation.normalizedRadius,
                    "edgeMargin": observation.edgeMargin,
                    "envelopeInflation": observation.envelopeInflation,
                }
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.split == "holdout" and not args.allow_holdout:
        raise RuntimeError("holdout evaluation requires --allow-holdout after model freeze")
    records = tuple(
        record
        for record in loadManifest(args.manifest)
        if record.split == args.split and record.sequenceId == args.sequence
    )
    if not records:
        raise RuntimeError(f"sequence is not present in split: {args.sequence}")
    records = tuple(sorted(records, key=lambda item: item.frameIndex))
    if records[0].frameIndex != 0 or not records[0].visible or records[0].bbox is None:
        raise RuntimeError("sequence requires a visible frame-0 initialization")
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")

    appConfig = loadConfig(args.config)
    appConfig = replace(
        appConfig,
        model=replace(appConfig.model, weights=args.weights.resolve()),
        visualization=replace(appConfig.visualization, enabled=False),
    )
    selected = records[: args.max_frames] if args.max_frames is not None else records
    truth = {record.frameIndex: record for record in selected}
    source = ManifestVideoSource(records, args.max_frames)
    sink = MemoryResultSink()
    timer = IntervalTimer()
    recorder = CandidateRecorder(truth)
    runtime = buildRuntime(appConfig)
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
        )
        sink.finalize(count)
    finally:
        closeBackend(runtime.backend)
        source.close()

    report = _summarize(args, selected, sink.results, timer, recorder.rows)
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
    print(json.dumps(report["summary"], indent=2))
    return 0


def _summarize(
    args: argparse.Namespace,
    records: tuple[ManifestRecord, ...],
    results: list[TrackResult],
    timer: IntervalTimer,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    erpIous: list[float] = []
    sphericalIous: list[float] = []
    centerErrors: list[float] = []
    widthErrors: list[float] = []
    heightErrors: list[float] = []
    absentCount = falsePositives = 0
    invalidSphericalPredictions = 0
    for record, result in zip(records[1:], results[1:], strict=True):
        if not record.visible or record.bbox is None or record.bfov is None:
            absentCount += 1
            falsePositives += int(result.valid)
            continue
        erpIous.append(circularBBoxIoU(result.bbox, record.bbox, record.width))
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
        "successRateAt0.5": float((values > 0.5).mean()) if values.size else 0.0,
        "sphericalMeanIoU": float(np.mean(sphericalIous)) if sphericalIous else 0.0,
        "meanCenterErrorDeg": float(np.degrees(np.mean(centerErrors))) if centerErrors else 0.0,
        "meanWidthRelativeError": float(np.mean(widthErrors)) if widthErrors else 0.0,
        "meanHeightRelativeError": float(np.mean(heightErrors)) if heightErrors else 0.0,
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
    }


if __name__ == "__main__":
    raise SystemExit(main())
