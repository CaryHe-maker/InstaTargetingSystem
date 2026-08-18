"""Seam-aware, bounded same-frame candidate fusion."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from math import pi, sqrt

import numpy as np

from instatarget.controller.state_model import EvaluatedCandidate
from instatarget.core.errors import ProtocolError
from instatarget.core.protocols import SphericalGeometry
from instatarget.core.types import BBoxXYWH, BFoV, DepthSummary, ProjectedObservation
from instatarget.geometry.projection_math import erpPixelToSphericalPoint

FUSION_OVERLAP_RATE = 0.70
FUSION_AGREEMENT_BONUS_WEIGHT = 0.15
FUSION_MAX_SCORE_GAIN = 0.03
FUSION_SCORE_CAP = 0.99


class FusionBoxMode(StrEnum):
    """Geometry used for a two-source fused candidate's output box."""

    MAX_INTERSECTION = "max_intersection"
    MIN_UNION = "min_union"


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
                "fusion boxMode must be 'max_intersection' or 'min_union'"
            ) from error

    def fuse(
        self,
        observations: Sequence[ProjectedObservation],
        *,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> EvaluatedCandidate | None:
        if not observations:
            return None
        if frameWidthPx <= 0 or frameHeightPx <= 0:
            raise ProtocolError("fusion frame dimensions must be positive")
        if len({item.viewId for item in observations}) != len(observations):
            raise ProtocolError("fusion observations must have unique viewIds")

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
        return max(candidates, key=_candidateRank)

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
        return EvaluatedCandidate(
            bfov=_intersectionBfov(bbox, self._geometry, frameWidthPx, frameHeightPx),
            bbox=bbox,
            confidence=confidence,
            sourceViewIds=tuple(sorted((first.viewId, second.viewId))),
            fused=True,
            overlapRate=overlap,
            minSourceConfidence=min(firstScore, secondScore),
            sourceConfidencePassed=True,
            representativeViewId=representative.viewId,
            representativeLocalBox=representative.localBox,
            depthSummary=_mergeDepth(first, second),
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
) -> EvaluatedCandidate | None:
    return Fusor(
        geometry,
        overlapRate=overlapRate,
        sourceMinConfidence=sourceMinConfidence,
        boxMode=boxMode,
    ).fuse(observations, frameWidthPx=frameWidthPx, frameHeightPx=frameHeightPx)


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
        depthSummary=observation.depthSummary,
    )


def _candidateRank(candidate: EvaluatedCandidate) -> tuple[float, bool, int]:
    return candidate.confidence, candidate.fused, -candidate.representativeViewId


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


def _mergeDepth(first: ProjectedObservation, second: ProjectedObservation) -> DepthSummary | None:
    if first.depthSummary is None:
        return second.depthSummary
    if second.depthSummary is None:
        return first.depthSummary
    firstWeight = max(_singleScore(first), 1e-6)
    secondWeight = max(_singleScore(second), 1e-6)
    total = firstWeight + secondWeight
    return DepthSummary(
        medianDepth=(
            first.depthSummary.medianDepth * firstWeight
            + second.depthSummary.medianDepth * secondWeight
        ) / total,
        meanDepth=(
            first.depthSummary.meanDepth * firstWeight
            + second.depthSummary.meanDepth * secondWeight
        ) / total,
        validRatio=(
            first.depthSummary.validRatio * firstWeight
            + second.depthSummary.validRatio * secondWeight
        ) / total,
        minDepth=min(first.depthSummary.minDepth, second.depthSummary.minDepth),
        maxDepth=max(first.depthSummary.maxDepth, second.depthSummary.maxDepth),
        confidence=(
            first.depthSummary.confidence * firstWeight
            + second.depthSummary.confidence * secondWeight
        ) / total,
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
