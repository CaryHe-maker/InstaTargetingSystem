"""Pure RGB annotation helpers for visualization output."""

from __future__ import annotations

from math import ceil, floor

import numpy as np
from numpy.typing import NDArray

from instatarget.core.errors import ProtocolError
from instatarget.core.types import BBoxXYWH

FLUORESCENT_GREEN_RGB = (57, 255, 20)
BOX_THICKNESS_PX = 2


def drawBoxRgb(
    rgb: NDArray[np.uint8],
    bbox: BBoxXYWH,
    *,
    wrapHorizontal: bool = False,
) -> NDArray[np.uint8]:
    """Return an RGB copy with one fluorescent-green XYWH box."""
    _requireRgb(rgb)
    heightPx, widthPx = rgb.shape[:2]
    output = rgb.copy()
    y0 = max(0, min(heightPx - 1, floor(bbox.yPx)))
    y1 = max(0, min(heightPx - 1, ceil(bbox.yPx + bbox.heightPx) - 1))
    if y1 < y0:
        raise ProtocolError(f"visualization bbox has no visible vertical area: {bbox}")

    if wrapHorizontal:
        _drawWrappedBox(output, bbox, y0, y1)
    else:
        _drawClippedBox(output, bbox, y0, y1)
    return output


def _drawClippedBox(
    output: NDArray[np.uint8],
    bbox: BBoxXYWH,
    y0: int,
    y1: int,
) -> None:
    widthPx = output.shape[1]
    x0 = max(0, min(widthPx - 1, floor(bbox.xPx)))
    x1 = max(0, min(widthPx - 1, ceil(bbox.xPx + bbox.widthPx) - 1))
    if x1 < x0:
        raise ProtocolError(f"visualization bbox has no visible horizontal area: {bbox}")
    _drawEdges(output, np.arange(x0, x1 + 1), x0, x1, y0, y1)


def _drawWrappedBox(
    output: NDArray[np.uint8],
    bbox: BBoxXYWH,
    y0: int,
    y1: int,
) -> None:
    widthPx = output.shape[1]
    if bbox.widthPx > widthPx:
        raise ProtocolError("wrapped visualization bbox cannot be wider than the RGB frame")
    startX = floor(bbox.xPx)
    endX = ceil(bbox.xPx + bbox.widthPx) - 1
    horizontalX = np.mod(np.arange(startX, endX + 1), widthPx)
    _drawEdges(output, horizontalX, startX % widthPx, endX % widthPx, y0, y1)


def _drawEdges(
    output: NDArray[np.uint8],
    horizontalX: NDArray[np.int64],
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> None:
    heightPx, widthPx = output.shape[:2]
    color = np.asarray(FLUORESCENT_GREEN_RGB, dtype=np.uint8)
    for offsetPx in range(BOX_THICKNESS_PX):
        topY = min(y0 + offsetPx, heightPx - 1)
        bottomY = max(y1 - offsetPx, 0)
        leftX = (x0 + offsetPx) % widthPx
        rightX = (x1 - offsetPx) % widthPx
        output[topY, horizontalX] = color
        output[bottomY, horizontalX] = color
        output[y0 : y1 + 1, leftX] = color
        output[y0 : y1 + 1, rightX] = color


def _requireRgb(rgb: NDArray[np.uint8]) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ProtocolError("visualization RGB must be a uint8 NumPy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0:
        raise ProtocolError(
            f"visualization RGB must have non-empty shape [H, W, 3], actual={rgb.shape}"
        )


__all__ = ["FLUORESCENT_GREEN_RGB", "drawBoxRgb"]
