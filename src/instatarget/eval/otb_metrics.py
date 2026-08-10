"""OTB-style metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from instatarget.core.types import BBoxXYWH
from instatarget.geometry.seam import splitSeamBox, wrapPixelX
from instatarget.io.result_writer import TextResultWriter


def bboxIoU(first: BBoxXYWH, second: BBoxXYWH) -> float:
    x0 = max(first.xPx, second.xPx)
    y0 = max(first.yPx, second.yPx)
    x1 = min(first.xPx + first.widthPx, second.xPx + second.widthPx)
    y1 = min(first.yPx + first.heightPx, second.yPx + second.heightPx)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = first.widthPx * first.heightPx + second.widthPx * second.heightPx - intersection
    if union <= 0.0:
        return 0.0
    return float(np.clip(intersection / union, 0.0, 1.0))


def circularBBoxIoU(first: BBoxXYWH, second: BBoxXYWH, frameWidthPx: int) -> float:
    """IoU for ERP boxes whose horizontal interval wraps at the seam."""
    firstParts = splitSeamBox(
        BBoxXYWH(wrapPixelX(first.xPx, frameWidthPx), first.yPx, first.widthPx, first.heightPx),
        frameWidthPx,
    )
    secondParts = splitSeamBox(
        BBoxXYWH(
            wrapPixelX(second.xPx, frameWidthPx),
            second.yPx,
            second.widthPx,
            second.heightPx,
        ),
        frameWidthPx,
    )
    intersection = 0.0
    for left in firstParts:
        for right in secondParts:
            xOverlap = max(
                0.0,
                min(left.xPx + left.widthPx, right.xPx + right.widthPx)
                - max(left.xPx, right.xPx),
            )
            yOverlap = max(
                0.0,
                min(left.yPx + left.heightPx, right.yPx + right.heightPx)
                - max(left.yPx, right.yPx),
            )
            intersection += xOverlap * yOverlap
    firstArea = first.widthPx * first.heightPx
    secondArea = second.widthPx * second.heightPx
    union = firstArea + secondArea - intersection
    return float(np.clip(intersection / union, 0.0, 1.0)) if union > 0.0 else 0.0


def successCurve(
    ious: list[float], thresholds: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    values = np.asarray(ious, dtype=np.float64)
    if values.size == 0:
        return thresholds, np.zeros_like(thresholds)
    curve = np.asarray([(values > threshold).mean() for threshold in thresholds], dtype=np.float64)
    return thresholds, curve


def auc(ious: list[float]) -> float:
    thresholds, curve = successCurve(ious)
    return float(np.trapezoid(curve, thresholds))


@dataclass(slots=True)
class OtbMetrics:
    ious: list[float] = field(default_factory=list)

    def update(self, prediction: BBoxXYWH, target: BBoxXYWH) -> None:
        self.ious.append(bboxIoU(prediction, target))

    def summarize(self) -> dict[str, float]:
        if not self.ious:
            return {"successRate@0.5": 0.0, "auc": 0.0, "meanIoU": 0.0}
        values = np.asarray(self.ious, dtype=np.float64)
        return {
            "successRate@0.5": float((values > 0.5).mean()),
            "auc": auc(self.ious),
            "meanIoU": float(values.mean()),
        }


def readResultFile(path: str | Path) -> list[BBoxXYWH]:
    reader = TextResultWriter()
    boxes: list[BBoxXYWH] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        boxes.append(reader.parseLine(line))
    return boxes


__all__ = [
    "OtbMetrics",
    "auc",
    "bboxIoU",
    "circularBBoxIoU",
    "readResultFile",
    "successCurve",
]
