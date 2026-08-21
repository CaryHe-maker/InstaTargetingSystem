"""Construction helpers for backend-local observations."""

from __future__ import annotations

from dataclasses import replace
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
        presenceLogit=prediction.presenceLogit,
        qualityLogit=prediction.qualityLogit,
        presenceProbability=prediction.presenceProbability,
        qualityProbability=prediction.qualityProbability,
        predictedIoU=prediction.predictedIoU,
        cornerScore=prediction.cornerScore,
    )


def refineObservations(
    observations: tuple[LocalObservation, ...],
    views: tuple[LocalView, ...],
) -> tuple[LocalObservation, ...]:
    """Temporary heuristic refinement used by the isolated IoU/BBox experiment."""
    if len(observations) != len(views):
        raise ModelError("refinement observations and views must have equal length")
    refined: list[LocalObservation] = []
    for observation, view in zip(observations, views, strict=True):
        score = _refinementScore(observation)
        scale = 1.0 + 0.10 * (0.55 - score)
        scale = max(0.90, min(1.08, scale))
        centerX = observation.bbox.xPx + observation.bbox.widthPx / 2.0
        centerY = observation.bbox.yPx + observation.bbox.heightPx / 2.0
        widthPx = observation.bbox.widthPx * scale
        heightPx = observation.bbox.heightPx * scale
        refinedBox = clipLocalBox(
            BBoxXYWH(
                xPx=centerX - widthPx / 2.0,
                yPx=centerY - heightPx / 2.0,
                widthPx=widthPx,
                heightPx=heightPx,
            ),
            view.spec.outputWidthPx,
            view.spec.outputHeightPx,
        )
        refined.append(
            replace(
                observation,
                bbox=refinedBox,
                fusedScore=min(1.0, max(observation.fusedScore, score)),
            )
        )
    return tuple(refined)


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


__all__ = [
    "LocalObservation",
    "buildRgbObservation",
    "clipLocalBox",
    "refineObservations",
    "startTimingNs",
]


def _refinementScore(observation: LocalObservation) -> float:
    values = [
        observation.appearanceProbability,
        observation.predictedIoU,
        observation.cornerScore,
        observation.fusedScore,
    ]
    filtered = [float(item) for item in values if item is not None]
    if not filtered:
        return 0.5
    return float(sum(filtered) / len(filtered))
