"""Seam-aware, bounded same-frame candidate fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from enum import StrEnum
from math import isfinite, pi, sqrt

import numpy as np

from instatarget.controller.state_model import EvaluatedCandidate
from instatarget.core.errors import ProtocolError
from instatarget.core.protocols import SphericalGeometry
from instatarget.core.types import BBoxXYWH, BFoV, ProjectedObservation
from instatarget.geometry.projection_math import erpPixelToSphericalPoint

FUSION_OVERLAP_RATE = 0.70
FUSION_AGREEMENT_BONUS_WEIGHT = 0.15
FUSION_MAX_SCORE_GAIN = 0.03
FUSION_SCORE_CAP = 0.99


class FusionBoxMode(StrEnum):
    """Geometry used for a two-source fused candidate's output box."""

    MAX_INTERSECTION = "max_intersection"
    MIN_UNION = "min_union"
    REFERENCE_ADAPTIVE = "reference_adaptive"
    BEST_SOURCE = "best_source"
    WEIGHTED_BOX = "weighted_box"
    ROBUST_SPHERICAL_CONSENSUS = "robust_spherical_consensus"


class Fusor:
    """Return exactly one best single or two-source fused candidate."""

    def __init__(
        self,
        geometry: SphericalGeometry,
        *,
        overlapRate: float = FUSION_OVERLAP_RATE,
        sourceMinConfidence: float = 0.80,
        boxMode: FusionBoxMode | str = FusionBoxMode.MAX_INTERSECTION,
    ) -> None:
        if not 0.0 <= overlapRate <= 1.0:
            raise ValueError("fusion overlapRate must be in [0, 1]")
        if not 0.0 <= sourceMinConfidence <= 1.0:
            raise ValueError("fusion sourceMinConfidence must be in [0, 1]")
        self._geometry = geometry
        self._overlapRate = float(overlapRate)
        self._sourceMinConfidence = float(sourceMinConfidence)
        try:
            self._boxMode = FusionBoxMode(boxMode)
        except ValueError as error:
            raise ValueError(
                "fusion boxMode must be 'max_intersection', 'min_union', "
                "'reference_adaptive', 'best_source', 'weighted_box', or "
                "'robust_spherical_consensus'"
            ) from error

    def fuse(
        self,
        observations: Sequence[ProjectedObservation],
        *,
        frameWidthPx: int,
        frameHeightPx: int,
        referenceBoxAreaPx: float | None = None,
    ) -> EvaluatedCandidate | None:
        if not observations:
            return None
        if frameWidthPx <= 0 or frameHeightPx <= 0:
            raise ProtocolError("fusion frame dimensions must be positive")
        if len({item.viewId for item in observations}) != len(observations):
            raise ProtocolError("fusion observations must have unique viewIds")
        if self._boxMode is FusionBoxMode.REFERENCE_ADAPTIVE and (
            referenceBoxAreaPx is None
            or not isfinite(referenceBoxAreaPx)
            or referenceBoxAreaPx <= 0.0
        ):
            raise ProtocolError("reference-adaptive fusion requires a positive reference area")

        candidates = [_singleCandidate(item) for item in observations]
        for firstIndex, first in enumerate(observations):
            for second in observations[firstIndex + 1 :]:
                firstScore = _singleScore(first)
                secondScore = _singleScore(second)
                if min(firstScore, secondScore) < self._sourceMinConfidence:
                    continue
                overlap = _overlapRate(first.bbox, second.bbox, frameWidthPx)
                if overlap < self._overlapRate:
                    continue
                candidate = self._fusedCandidate(
                    first,
                    second,
                    overlap,
                    _agreementIou(first.bbox, second.bbox, frameWidthPx),
                    frameWidthPx,
                    frameHeightPx,
                )
                if candidate is not None:
                    candidates.append(candidate)
        best = max(candidates, key=_candidateRank)
        observationsById = {item.viewId: item for item in observations}
        if self._boxMode is FusionBoxMode.WEIGHTED_BOX and best.fused:
            best = self._weightedGeometry(best, observationsById, frameWidthPx, frameHeightPx)
        if self._boxMode is FusionBoxMode.ROBUST_SPHERICAL_CONSENSUS and best.fused:
            best = self._robustGeometry(best, observations, frameWidthPx, frameHeightPx)
        if self._boxMode in {FusionBoxMode.WEIGHTED_BOX, FusionBoxMode.ROBUST_SPHERICAL_CONSENSUS}:
            return best
        if self._boxMode is not FusionBoxMode.REFERENCE_ADAPTIVE or not best.fused:
            return best
        observationsById = {item.viewId: item for item in observations}
        first, second = (observationsById[viewId] for viewId in best.sourceViewIds)
        bbox = _referenceAdaptiveBox(
            first.bbox,
            second.bbox,
            float(referenceBoxAreaPx),
            frameWidthPx,
            frameHeightPx,
        )
        if bbox is None:
            return best
        return replace(
            best,
            bbox=bbox,
            bfov=_intersectionBfov(bbox, self._geometry, frameWidthPx, frameHeightPx),
        )

    def _weightedGeometry(
        self,
        candidate: EvaluatedCandidate,
        observationsById: dict[int, ProjectedObservation],
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> EvaluatedCandidate:
        first, second = (observationsById[item] for item in candidate.sourceViewIds)
        weights = np.asarray([_singleScore(first), _singleScore(second)], dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 1e-9:
            return candidate
        weights /= float(weights.sum())
        reference = first.bbox.xPx + first.bbox.widthPx / 2.0
        centers = np.asarray([
            _unwrapCenter(item.bbox.xPx + item.bbox.widthPx / 2.0, reference, frameWidthPx)
            for item in (first, second)
        ], dtype=np.float64)
        y = float(np.dot(weights, [item.bbox.yPx + item.bbox.heightPx / 2.0 for item in (first, second)]))
        width = float(np.exp(np.dot(weights, [np.log(item.bbox.widthPx) for item in (first, second)])))
        height = float(np.exp(np.dot(weights, [np.log(item.bbox.heightPx) for item in (first, second)])))
        x = float(np.dot(weights, centers) - width / 2.0) % frameWidthPx
        y = float(np.clip(y - height / 2.0, 0.0, max(0.0, frameHeightPx - height)))
        bbox = BBoxXYWH(xPx=x, yPx=y, widthPx=min(width, float(frameWidthPx)), heightPx=min(height, float(frameHeightPx)))
        try:
            bfov = self._geometry.bboxToBfov(bbox, frameWidthPx, frameHeightPx)
        except Exception:
            return candidate
        return replace(candidate, bbox=bbox, bfov=bfov)

    def _robustGeometry(
        self,
        candidate: EvaluatedCandidate,
        observations: Sequence[ProjectedObservation],
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> EvaluatedCandidate:
        seeds = [item for item in observations if item.viewId in candidate.sourceViewIds]
        if len(seeds) < 2:
            return candidate
        support = [item for item in observations if _singleScore(item) >= self._sourceMinConfidence and any(_overlapRate(item.bbox, seed.bbox, frameWidthPx) >= self._overlapRate for seed in seeds)]
        if len(support) < 2:
            return candidate
        # A bounded Huber-like center consensus; dimensions use weighted medians.
        reference = seeds[0].bbox.xPx + seeds[0].bbox.widthPx / 2.0
        centers = np.asarray([_unwrapCenter(item.bbox.xPx + item.bbox.widthPx / 2.0, reference, frameWidthPx) for item in support], dtype=np.float64)
        weights = np.asarray([_singleScore(item) for item in support], dtype=np.float64)
        if not np.isfinite(centers).all() or not np.isfinite(weights).all() or float(weights.sum()) <= 1e-9:
            return candidate
        weights /= float(weights.sum())
        center = float(np.dot(weights, centers))
        for _ in range(3):
            residual = np.abs(centers - center)
            delta = np.clip(0.25 * max(seeds[0].bfov.horizontalFovRad, seeds[0].bfov.verticalFovRad), np.deg2rad(1.0), np.deg2rad(15.0)) * frameWidthPx / (2.0 * np.pi)
            robust = np.minimum(1.0, delta / np.maximum(residual, 1e-9))
            effective = weights * robust
            center = float(np.dot(effective, centers) / max(float(effective.sum()), 1e-9))
        width = _weightedMedian([item.bbox.widthPx for item in support], weights)
        height = _weightedMedian([item.bbox.heightPx for item in support], weights)
        y = float(np.dot(weights, [item.bbox.yPx + item.bbox.heightPx / 2.0 for item in support]) - height / 2.0)
        bbox = BBoxXYWH(xPx=float(center - width / 2.0) % frameWidthPx, yPx=float(np.clip(y, 0.0, max(0.0, frameHeightPx - height))), widthPx=min(width, float(frameWidthPx)), heightPx=min(height, float(frameHeightPx)))
        try:
            bfov = self._geometry.bboxToBfov(bbox, frameWidthPx, frameHeightPx)
        except Exception:
            return candidate
        return replace(candidate, bbox=bbox, bfov=bfov)

    def _fusedCandidate(
        self,
        first: ProjectedObservation,
        second: ProjectedObservation,
        overlap: float,
        agreementIou: float,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> EvaluatedCandidate | None:
        if self._boxMode is FusionBoxMode.MIN_UNION:
            bbox = _unionBox(first.bbox, second.bbox, frameWidthPx, frameHeightPx)
        else:
            bbox = _intersectionBox(first.bbox, second.bbox, frameWidthPx, frameHeightPx)
        if bbox is None:
            return None
        firstScore = _singleScore(first)
        secondScore = _singleScore(second)
        base = sqrt(firstScore * secondScore)
        consistency = 1.0 - abs(firstScore - secondScore)
        bonus = (
            FUSION_AGREEMENT_BONUS_WEIGHT
            * agreementIou
            * consistency
            * (1.0 - base)
        )
        confidence = min(
            base + bonus,
            max(firstScore, secondScore) + FUSION_MAX_SCORE_GAIN,
            FUSION_SCORE_CAP,
        )
        representative = max(
            (first, second),
            key=lambda item: (_singleScore(item), -item.viewId),
        )
        if self._boxMode is FusionBoxMode.BEST_SOURCE:
            bbox = representative.bbox
            bfov = representative.bfov
        else:
            bfov = _intersectionBfov(bbox, self._geometry, frameWidthPx, frameHeightPx)
        return EvaluatedCandidate(
            bfov=bfov,
            bbox=bbox,
            confidence=confidence,
            sourceViewIds=tuple(sorted((first.viewId, second.viewId))),
            fused=True,
            overlapRate=overlap,
            minSourceConfidence=min(firstScore, secondScore),
            sourceConfidencePassed=True,
            representativeViewId=representative.viewId,
            representativeLocalBox=representative.localBox,
        )


def fuse(
    observations: Sequence[ProjectedObservation],
    geometry: SphericalGeometry,
    *,
    frameWidthPx: int,
    frameHeightPx: int,
    overlapRate: float = FUSION_OVERLAP_RATE,
    sourceMinConfidence: float = 0.80,
    boxMode: FusionBoxMode | str = FusionBoxMode.MAX_INTERSECTION,
    referenceBoxAreaPx: float | None = None,
) -> EvaluatedCandidate | None:
    return Fusor(
        geometry,
        overlapRate=overlapRate,
        sourceMinConfidence=sourceMinConfidence,
        boxMode=boxMode,
    ).fuse(
        observations,
        frameWidthPx=frameWidthPx,
        frameHeightPx=frameHeightPx,
        referenceBoxAreaPx=referenceBoxAreaPx,
    )


def _singleScore(observation: ProjectedObservation) -> float:
    return float(
        np.clip(
            observation.singleScore
            if observation.singleScore is not None
            else observation.fusedScore,
            0.0,
            1.0,
        )
    )


def _singleCandidate(observation: ProjectedObservation) -> EvaluatedCandidate:
    return EvaluatedCandidate(
        bfov=observation.bfov,
        bbox=observation.bbox,
        confidence=_singleScore(observation),
        sourceViewIds=(observation.viewId,),
        fused=False,
        overlapRate=None,
        minSourceConfidence=None,
        sourceConfidencePassed=True,
        representativeViewId=observation.viewId,
        representativeLocalBox=observation.localBox,
    )


def _candidateRank(candidate: EvaluatedCandidate) -> tuple[float, bool, int]:
    return candidate.confidence, candidate.fused, -candidate.representativeViewId


def _unwrapCenter(value: float, reference: float, width: int) -> float:
    while value - reference > width / 2.0:
        value -= width
    while value - reference < -width / 2.0:
        value += width
    return value


def _weightedMedian(values: Sequence[float], weights: np.ndarray) -> float:
    order = np.argsort(np.asarray(values, dtype=np.float64))
    orderedValues = np.asarray(values, dtype=np.float64)[order]
    orderedWeights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(orderedWeights)
    index = int(np.searchsorted(cumulative, 0.5, side="left"))
    return float(orderedValues[min(index, len(orderedValues) - 1)])


def _xSegments(box: BBoxXYWH, frameWidthPx: int) -> tuple[tuple[float, float], ...]:
    width = min(float(frameWidthPx), box.widthPx)
    start = box.xPx % frameWidthPx
    end = start + width
    if end <= frameWidthPx:
        return ((start, end),)
    return ((start, float(frameWidthPx)), (0.0, end - frameWidthPx))


def _overlapRate(first: BBoxXYWH, second: BBoxXYWH, frameWidthPx: int) -> float:
    intersectionArea = _intersectionArea(first, second, frameWidthPx)
    smallerArea = min(_boxArea(first, frameWidthPx), _boxArea(second, frameWidthPx))
    if smallerArea <= 0.0:
        return 0.0
    return float(np.clip(intersectionArea / smallerArea, 0.0, 1.0))


def _agreementIou(first: BBoxXYWH, second: BBoxXYWH, frameWidthPx: int) -> float:
    intersectionArea = _intersectionArea(first, second, frameWidthPx)
    unionArea = (
        _boxArea(first, frameWidthPx)
        + _boxArea(second, frameWidthPx)
        - intersectionArea
    )
    if unionArea <= 0.0:
        return 0.0
    return float(np.clip(intersectionArea / unionArea, 0.0, 1.0))


def _intersectionArea(first: BBoxXYWH, second: BBoxXYWH, frameWidthPx: int) -> float:
    yStart = max(first.yPx, second.yPx)
    yEnd = min(first.yPx + first.heightPx, second.yPx + second.heightPx)
    if yEnd <= yStart:
        return 0.0
    horizontal = sum(
        max(0.0, min(firstEnd, secondEnd) - max(firstStart, secondStart))
        for firstStart, firstEnd in _xSegments(first, frameWidthPx)
        for secondStart, secondEnd in _xSegments(second, frameWidthPx)
    )
    return horizontal * (yEnd - yStart)


def _boxArea(box: BBoxXYWH, frameWidthPx: int) -> float:
    return min(float(frameWidthPx), box.widthPx) * box.heightPx


def _intersectionBox(
    first: BBoxXYWH,
    second: BBoxXYWH,
    frameWidthPx: int,
    frameHeightPx: int,
) -> BBoxXYWH | None:
    intersections = sorted(
        (
            max(firstStart, secondStart),
            min(firstEnd, secondEnd),
        )
        for firstStart, firstEnd in _xSegments(first, frameWidthPx)
        for secondStart, secondEnd in _xSegments(second, frameWidthPx)
        if min(firstEnd, secondEnd) > max(firstStart, secondStart)
    )
    if not intersections:
        return None
    merged: list[list[float]] = []
    for start, end in intersections:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if len(merged) > 1 and merged[0][0] == 0.0 and merged[-1][1] == frameWidthPx:
        xPx = merged[-1][0]
        width = merged[0][1] + frameWidthPx - xPx
    else:
        start, end = max(merged, key=lambda item: item[1] - item[0])
        xPx, width = start, end - start
    yStart = max(0.0, first.yPx, second.yPx)
    yEnd = min(float(frameHeightPx), first.yPx + first.heightPx, second.yPx + second.heightPx)
    if yEnd <= yStart or width <= 0.0:
        return None
    return BBoxXYWH(xPx=xPx, yPx=yStart, widthPx=width, heightPx=yEnd - yStart)


def _unionBox(
    first: BBoxXYWH,
    second: BBoxXYWH,
    frameWidthPx: int,
    frameHeightPx: int,
) -> BBoxXYWH | None:
    """Return the smallest circular ERP box containing both source boxes."""
    firstStart = first.xPx % frameWidthPx
    firstWidth = min(float(frameWidthPx), first.widthPx)
    secondStart = second.xPx % frameWidthPx
    secondWidth = min(float(frameWidthPx), second.widthPx)
    best: tuple[float, float] | None = None
    for shift in (-float(frameWidthPx), 0.0, float(frameWidthPx)):
        start = min(firstStart, secondStart + shift)
        end = max(firstStart + firstWidth, secondStart + shift + secondWidth)
        width = end - start
        if best is None or width < best[1]:
            best = (start, width)
    if best is None or best[1] <= 0.0:
        return None
    yStart = max(0.0, min(first.yPx, second.yPx))
    yEnd = min(
        float(frameHeightPx),
        max(first.yPx + first.heightPx, second.yPx + second.heightPx),
    )
    if yEnd <= yStart:
        return None
    return BBoxXYWH(
        xPx=best[0] % frameWidthPx,
        yPx=yStart,
        widthPx=min(float(frameWidthPx), best[1]),
        heightPx=yEnd - yStart,
    )


def _referenceAdaptiveBox(
    first: BBoxXYWH,
    second: BBoxXYWH,
    referenceAreaPx: float,
    frameWidthPx: int,
    frameHeightPx: int,
) -> BBoxXYWH | None:
    intersection = _intersectionBox(first, second, frameWidthPx, frameHeightPx)
    union = _unionBox(first, second, frameWidthPx, frameHeightPx)
    if intersection is None or union is None:
        return None
    intersectionArea = _boxArea(intersection, frameWidthPx)
    unionArea = _boxArea(union, frameWidthPx)
    if referenceAreaPx <= intersectionArea:
        return intersection
    if referenceAreaPx < unionArea:
        expandedIntersection = _resizeCenteredToArea(
            intersection,
            1.5 * referenceAreaPx,
            frameWidthPx,
            frameHeightPx,
        )
        return (
            _intersectionBox(expandedIntersection, union, frameWidthPx, frameHeightPx)
            or intersection
        )
    return union


def _resizeCenteredToArea(
    box: BBoxXYWH,
    targetAreaPx: float,
    frameWidthPx: int,
    frameHeightPx: int,
) -> BBoxXYWH:
    frameWidth = float(frameWidthPx)
    frameHeight = float(frameHeightPx)
    targetArea = min(float(targetAreaPx), frameWidth * frameHeight)
    sourceWidth = min(frameWidth, box.widthPx)
    aspectRatio = sourceWidth / box.heightPx
    width = sqrt(targetArea * aspectRatio)
    height = sqrt(targetArea / aspectRatio)
    if width > frameWidth:
        width = frameWidth
        height = targetArea / width
    if height > frameHeight:
        height = frameHeight
        width = min(frameWidth, targetArea / height)
    centerX = (box.xPx + sourceWidth / 2.0) % frameWidth
    centerY = min(frameHeight, max(0.0, box.yPx + box.heightPx / 2.0))
    return BBoxXYWH(
        xPx=(centerX - width / 2.0) % frameWidth,
        yPx=min(max(0.0, centerY - height / 2.0), frameHeight - height),
        widthPx=width,
        heightPx=height,
    )


def _intersectionBfov(
    bbox: BBoxXYWH,
    geometry: SphericalGeometry,
    frameWidthPx: int,
    frameHeightPx: int,
) -> BFoV:
    horizontalSpan = 2.0 * pi * min(bbox.widthPx, frameWidthPx) / frameWidthPx
    if horizontalSpan < pi:
        return geometry.bboxToBfov(bbox, frameWidthPx, frameHeightPx)
    center = erpPixelToSphericalPoint(
        (bbox.xPx + bbox.widthPx / 2.0) % frameWidthPx,
        min(float(frameHeightPx), max(0.0, bbox.yPx + bbox.heightPx / 2.0)),
        frameWidthPx,
        frameHeightPx,
    )
    return BFoV(
        center=center,
        horizontalFovRad=min(horizontalSpan, float(np.nextafter(2.0 * pi, 0.0))),
        verticalFovRad=min(
            pi * min(bbox.heightPx, frameHeightPx) / frameHeightPx,
            float(np.nextafter(pi, 0.0)),
        ),
    )


__all__ = [
    "FUSION_AGREEMENT_BONUS_WEIGHT",
    "FUSION_MAX_SCORE_GAIN",
    "FUSION_OVERLAP_RATE",
    "FUSION_SCORE_CAP",
    "FusionBoxMode",
    "Fusor",
    "fuse",
]
