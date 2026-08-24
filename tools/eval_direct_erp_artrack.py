"""Evaluate ARTrack by feeding each full ERP frame directly as the search image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from instatarget.core.config import loadConfig
from instatarget.core.types import BBoxXYWH
from instatarget.eval.otb_metrics import circularBBoxIoU
from instatarget.tracker.artrack_model import PyTorchARTrackV2Session
from instatarget.training.dataset import loadManifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = tuple(
        sorted(
            (row for row in loadManifest(args.manifest) if row.sequenceId == args.sequence),
            key=lambda row: row.frameIndex,
        )
    )
    if not records or records[0].bbox is None:
        raise RuntimeError("direct ERP evaluation requires a frame-0 target bbox")
    args.output.mkdir(parents=True, exist_ok=True)
    framesRoot = args.output / "frames"
    framesRoot.mkdir(parents=True, exist_ok=True)

    config = loadConfig(args.config)
    session = PyTorchARTrackV2Session(config.model)
    capture = cv2.VideoCapture(str(records[0].videoPath))
    if not capture.isOpened():
        session.close()
        raise RuntimeError(f"cannot open video: {records[0].videoPath}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    writer = None
    rows: list[dict[str, object]] = []
    try:
        ok, bgr0 = capture.read()
        if not ok or bgr0 is None:
            raise RuntimeError("video has no readable frame 0")
        rgb0 = np.ascontiguousarray(bgr0[:, :, ::-1])
        template = session.encodeTemplate(rgb0, records[0].bbox)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for record in records:
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f"cannot decode frame {record.frameIndex}")
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            prediction, score = _inferFullErp(session, rgb, template.tensor)
            iou = (
                circularBBoxIoU(prediction, record.bbox, record.width)
                if record.visible and record.bbox is not None
                else 0.0
            )
            row = {
                "frameIndex": int(record.frameIndex),
                "score": score,
                "bbox": _boxDict(prediction),
                "truthBbox": _boxDict(record.bbox) if record.bbox is not None else None,
                "visible": record.visible,
                "circularErpIoU": iou,
            }
            rows.append(row)
            annotated = _annotate(bgr, prediction, record.bbox, score, iou, record.frameIndex)
            cv2.imwrite(str(framesRoot / f"frame_{record.frameIndex:04d}.jpg"), annotated)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(args.output / "direct_erp.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (annotated.shape[1], annotated.shape[0]),
                )
            writer.write(annotated)
            if record.frameIndex % 50 == 0:
                print(
                    json.dumps({"frameIndex": record.frameIndex, "score": score, "iou": iou}),
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        session.close()

    (args.output / "results.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    visible = [row for row in rows if row["visible"]]
    report = {
        "sequence": args.sequence,
        "mode": "full_erp_direct_no_geometry_no_fusor",
        "frameCount": len(rows),
        "meanScore": float(np.mean([row["score"] for row in rows])),
        "minScore": float(np.min([row["score"] for row in rows])),
        "maxScore": float(np.max([row["score"] for row in rows])),
        "circularErpMeanIoU": float(np.mean([row["circularErpIoU"] for row in visible])),
        "hitRateAt0.5": float(np.mean([row["circularErpIoU"] >= 0.5 for row in visible])),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _inferFullErp(
    session: PyTorchARTrackV2Session, rgb: np.ndarray, templateTensor: object
) -> tuple[BBoxXYWH, float]:
    torch = session._torch
    resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    search = session._preprocessBatch((np.ascontiguousarray(resized),))
    template = torch.stack((templateTensor, templateTensor), dim=0)[:, None]
    with torch.inference_mode():
        output = session._model(template=template, search=search)
    # ARTrack decodes box coordinates around the search-center origin: the
    # model range is [-0.5, 0.5], not [0, 1]. Map each axis independently from
    # the square resized ERP back to the original 2:1 ERP dimensions.
    seq = output["seqs"][0, :4].float() / 399.0 - 0.5
    x0, y0, x1, y1 = (float(value) for value in seq.tolist())
    widthPx = float(rgb.shape[1])
    heightPx = float(rgb.shape[0])
    x0Px = float(np.clip(x0 * widthPx, 0.0, widthPx - 1.0))
    y0Px = float(np.clip(y0 * heightPx, 0.0, heightPx - 1.0))
    x1Px = float(np.clip(x1 * widthPx, x0Px + 1.0, widthPx))
    y1Px = float(np.clip(y1 * heightPx, y0Px + 1.0, heightPx))
    prediction = BBoxXYWH(x0Px, y0Px, x1Px - x0Px, y1Px - y0Px)
    score = float(torch.sigmoid(output["score"][0].reshape(-1)[0]).item())
    return prediction, score


def _boxDict(box: BBoxXYWH) -> dict[str, float]:
    return {
        "xPx": box.xPx,
        "yPx": box.yPx,
        "widthPx": box.widthPx,
        "heightPx": box.heightPx,
    }


def _annotate(
    bgr: np.ndarray,
    prediction: BBoxXYWH,
    truth: BBoxXYWH | None,
    score: float,
    iou: float,
    frameIndex: int,
) -> np.ndarray:
    image = bgr.copy()
    if truth is not None:
        _drawBox(image, truth, (0, 220, 0), "truth")
    _drawBox(image, prediction, (0, 0, 255), "direct ERP prediction")
    lines = (f"frame={frameIndex}", f"score={score:.6f}", f"ERP IoU={iou:.3f}")
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (18, 32 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def _drawBox(image: np.ndarray, box: BBoxXYWH, color: tuple[int, int, int], label: str) -> None:
    x1, y1 = int(round(box.xPx)), int(round(box.yPx))
    x2, y2 = int(round(box.xPx + box.widthPx)), int(round(box.yPx + box.heightPx))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise
