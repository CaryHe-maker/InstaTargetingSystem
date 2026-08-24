"""Render ERP and local-crop diagnostics for one manifest evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from instatarget.core.types import BBoxXYWH, BFoV, FrameIndex, FramePacket, SequenceId, ViewSpec
from instatarget.geometry import SphericalGeometryImpl
from instatarget.geometry.projection_math import makeSphericalPoint
from instatarget.training.dataset import _bfovToNormalizedLocal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["sequenceId"] == "train_real/seq_0005":
            records[int(row["frameIndex"])] = row
    report = json.loads(args.report.read_text(encoding="utf-8"))
    resultFrames = {int(row["frameIndex"]): row for row in report["frames"]}
    candidates = [
        json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines()
    ]
    byFrame = {}
    for row in candidates:
        byFrame.setdefault(int(row["frameIndex"]), []).append(row)
    candidateFrames = set(byFrame)
    selected = {0, 1, 2, 20, 50, 100, 200, 400, 600, 800, 834}
    if candidateFrames:
        selected.add(min(candidateFrames, key=lambda i: float(byFrame[i][0]["circularErpIoU"])))
    metrics = {int(row["frameIndex"]): row for row in report["frameMetrics"]}
    selected.update(
        row["frameIndex"]
        for row in sorted(
            report["frameMetrics"], key=lambda row: row["centerErrorDeg"], reverse=True
        )[:3]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    geometry = SphericalGeometryImpl(boundarySamplesPerEdge=65)
    try:
        for index in sorted(selected):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok or bgr is None or index not in records:
                continue
            frame = bgr[:, :, ::-1].copy()
            truth = _bbox(records[index]["bbox"])
            result = resultFrames.get(index)
            rows = byFrame.get(index, [])
            best = (
                max(rows, key=lambda row: float(row.get("circularErpIoU", 0.0))) if rows else None
            )
            crop = None
            if best is not None:
                crop = _localCrop(frame, best, result, geometry, index)
            canvas = frame.copy()
            _draw(canvas, truth, (0, 220, 0), "truth")
            if result is not None:
                _draw(canvas, _bbox(result["bbox"]), (230, 40, 40), "ARTrack/controller")
            text = [
                f"frame={index}",
                f"candidate_iou={float(best['circularErpIoU']):.3f}"
                if best
                else "candidate_iou=n/a",
                f"score={float(best['modelScore']):.3f}" if best else "score=n/a",
                f"view={int(best['viewId'])} fov={float(best['viewHorizontalFovDeg']):.1f}deg"
                if best
                else "view=n/a",
                f"center_error={float(metrics[index]['centerErrorDeg']):.2f}deg"
                if index in metrics
                else "center_error=n/a",
            ]
            for lineNo, line in enumerate(text):
                cv2.putText(
                    canvas,
                    line,
                    (16, 28 + 28 * lineNo),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(args.output / f"frame_{index:04d}_erp.jpg"), canvas[:, :, ::-1])
            if crop is not None:
                cv2.imwrite(str(args.output / f"frame_{index:04d}_local.jpg"), crop[:, :, ::-1])
    finally:
        capture.release()
    (args.output / "README.txt").write_text(
        "Green=manifest truth, red=ARTrack/Controller prediction. "
        "Local images are exact 256x256 crops; green is truth and red is prediction.\n",
        encoding="utf-8",
    )
    return 0


def _bbox(values: list[float]) -> BBoxXYWH:
    if isinstance(values, dict):
        return BBoxXYWH(
            xPx=float(values["xPx"]),
            yPx=float(values["yPx"]),
            widthPx=float(values["widthPx"]),
            heightPx=float(values["heightPx"]),
        )
    return BBoxXYWH(
        xPx=float(values[0]),
        yPx=float(values[1]),
        widthPx=float(values[2]),
        heightPx=float(values[3]),
    )


def _viewSpec(row: dict) -> ViewSpec:
    center = makeSphericalPoint(
        np.radians(float(row["viewYawDeg"])), np.radians(float(row["viewPitchDeg"]))
    )
    return ViewSpec(
        viewId=int(row["viewId"]),
        bfov=BFoV(
            center=center,
            horizontalFovRad=np.radians(float(row["viewHorizontalFovDeg"])),
            verticalFovRad=np.radians(float(row["viewVerticalFovDeg"])),
        ),
        outputWidthPx=256,
        outputHeightPx=256,
    )


def _localCrop(
    frame: np.ndarray, row: dict, result: dict | None, geometry: SphericalGeometryImpl, index: int
) -> np.ndarray:
    spec = _viewSpec(row)
    packet = FramePacket(
        sequenceId=SequenceId("train_real/seq_0005"),
        frameIndex=FrameIndex(index),
        timestampNs=index,
        rgb=frame,
    )
    crop = geometry.cropViews(packet, (spec,))[0].rgb.copy()
    truth = np.asarray(row["targetLocalBoxCxCyWh"], dtype=np.float64)
    _drawNormalized(crop, truth, (0, 220, 0), "truth")
    if result is not None:
        predictionBfov = geometry.bboxToBfov(_bbox(result["bbox"]), frame.shape[1], frame.shape[0])
        predicted = _bfovToNormalizedLocal(predictionBfov, spec)
        _drawNormalized(crop, predicted, (230, 40, 40), "prediction")
    return crop


def _drawNormalized(
    image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], label: str
) -> None:
    cx, cy, width, height = (float(value) for value in box)
    x1, y1 = int((cx - width / 2) * 256), int((cy - height / 2) * 256)
    x2, y2 = int((cx + width / 2) * 256), int((cy + height / 2) * 256)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (max(2, x1), max(16, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw(image: np.ndarray, box: BBoxXYWH, color: tuple[int, int, int], label: str) -> None:
    x1, y1 = int(round(box.xPx)), int(round(box.yPx))
    x2, y2 = int(round(box.xPx + box.widthPx)), int(round(box.yPx + box.heightPx))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
    )


if __name__ == "__main__":
    raise SystemExit(main())
