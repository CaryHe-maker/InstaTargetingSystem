"""Construction helpers for backend-local observations."""

from __future__ import annotations

from time import perf_counter_ns

from instatarget.core.errors import ModelError
from instatarget.core.types import BBoxXYWH, LocalObservation, LocalView
from instatarget.tracker.hit_backend import HiTPrediction


def buildRgbObservation(
    view: LocalView,
    prediction: HiTPrediction,
    latencyNs: int,
) -> LocalObservation:
    """Convert a HiT prediction into the stable RGB-only output contract."""
    if latencyNs < 0:
        raise ModelError(f"latencyNs must be non-negative, actual={latencyNs}")
    bbox = clipLocalBox(prediction.bbox, view.spec.outputWidthPx, view.spec.outputHeightPx)
    return LocalObservation(
        viewId=view.spec.viewId,
        bbox=bbox,
        modelScore=prediction.modelScore,
        appearanceScore=prediction.appearanceScore,
        fusedScore=prediction.appearanceScore,
        latencyNs=latencyNs,
    )


def clipLocalBox(bbox: BBoxXYWH, widthPx: int, heightPx: int) -> BBoxXYWH:
    """Clip a model box to a local view, rejecting boxes with no area."""
    x0 = max(0.0, min(float(widthPx), bbox.xPx))
    y0 = max(0.0, min(float(heightPx), bbox.yPx))
    x1 = max(0.0, min(float(widthPx), bbox.xPx + bbox.widthPx))
    y1 = max(0.0, min(float(heightPx), bbox.yPx + bbox.heightPx))
    if x1 <= x0 or y1 <= y0:
        raise ModelError(f"HiT returned a box outside the local view: {bbox}")
    return BBoxXYWH(xPx=x0, yPx=y0, widthPx=x1 - x0, heightPx=y1 - y0)


def startTimingNs() -> int:
    """Return the monotonic timestamp used by backend latency measurements."""
    return perf_counter_ns()


__all__ = ["LocalObservation", "buildRgbObservation", "clipLocalBox", "startTimingNs"]
