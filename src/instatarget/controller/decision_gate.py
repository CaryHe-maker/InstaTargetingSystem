"""Candidate scoring, clustering and single-frame aggregation for DTC."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, log, pi
from typing import TYPE_CHECKING

import numpy as np

from instatarget.core.config import DecisionGateConfig, TrackingConfig
from instatarget.core.errors import ProtocolError
from instatarget.core.protocols import SphericalGeometry
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    DepthSummary,
    ProjectedObservation,
    SphericalPoint,
)
from instatarget.geometry.projection_math import makeSphericalPoint

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ScoredObservation:
    observation: ProjectedObservation
    decisionScore: float
    depthConsistencyScore: float


@dataclass(frozen=True, slots=True)
class FrameAggregate:
    """The best geometrically consistent cluster in one frame."""

    bfov: BFoV
    bbox: BBoxXYWH
    confidence: float
    decisionScore: float
    sourceViewIds: tuple[int, ...]
    representativeViewId: int
    localBox: BBoxXYWH | None
    depthSummary: DepthSummary | None
    supported: bool


class DecisionGate:
    """Apply lightweight control weights without reimplementing backend fusion."""

    def __init__(self, gateConfig: DecisionGateConfig, trackingConfig: TrackingConfig) -> None:
        self._gateConfig = gateConfig
        self._trackingConfig = trackingConfig

    def score(self, observation: ProjectedObservation) -> ScoredObservation:
        depthAvailable = observation.depthSummary is not None
        depthScore = (
            observation.depthScore * observation.depthSummary.confidence
            if depthAvailable and observation.depthSummary is not None
            else 0.0
        )
        weights = [
            max(
                0.0,
                1.0
                - self._gateConfig.motionScoreWeight
                - self._gateConfig.scaleScoreWeight
                - (self._gateConfig.depthConsistencyWeight if depthAvailable else 0.0),
            ),
            self._gateConfig.motionScoreWeight,
            self._gateConfig.scaleScoreWeight,
        ]
        values = [observation.fusedScore, observation.motionScore, observation.scaleScore]
        if depthAvailable:
            weights.append(self._gateConfig.depthConsistencyWeight)
            values.append(depthScore)
        totalWeight = sum(weights)
        if totalWeight <= 0.0:
            raise ProtocolError("decision gate must have a positive effective weight")
        decisionScore = float(np.clip(np.dot(weights, values) / totalWeight, 0.0, 1.0))
        return ScoredObservation(
            observation=observation,
            decisionScore=decisionScore,
            depthConsistencyScore=float(np.clip(depthScore, 0.0, 1.0)),
        )

    def aggregate(
        self,
        observations: Sequence[ProjectedObservation],
        geometry: SphericalGeometry,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> FrameAggregate | None:
        scored = [self.score(observation) for observation in observations]
        eligible = [
            item
            for item in scored
            if item.observation.fusedScore >= self._trackingConfig.candidateMinScore
        ]
        if not eligible:
            return None

        clusters: list[list[ScoredObservation]] = []
        for candidate in eligible:
            matching = next(
                (
                    cluster
                    for cluster in clusters
                    if _compatible(
                        candidate,
                        cluster[0],
                        self._trackingConfig.scaleClusterTolerance,
                    )
                ),
                None,
            )
            if matching is None:
                clusters.append([candidate])
            else:
                matching.append(candidate)

        best = max(clusters, key=lambda cluster: _clusterRank(cluster))
        return _aggregateCluster(
            best,
            geometry,
            frameWidthPx,
            frameHeightPx,
            self._trackingConfig.minViewsForCommit,
        )


def _compatible(
    candidate: ScoredObservation,
    reference: ScoredObservation,
    scaleTolerance: float,
) -> bool:
    angle = _centerAngle(candidate.observation.bfov.center, reference.observation.bfov.center)
    angleLimit = max(
        0.25,
        min(
            pi * 0.95,
            max(
                candidate.observation.bfov.horizontalFovRad,
                reference.observation.bfov.horizontalFovRad,
            )
            * 0.55,
        ),
    )
    candidateArea = candidate.observation.bbox.widthPx * candidate.observation.bbox.heightPx
    referenceArea = reference.observation.bbox.widthPx * reference.observation.bbox.heightPx
    scaleDelta = abs(log(max(candidateArea, 1e-6) / max(referenceArea, 1e-6)))
    return angle <= angleLimit and scaleDelta <= scaleTolerance


def _clusterRank(cluster: list[ScoredObservation]) -> tuple[float, int, float]:
    weight = sum(item.decisionScore for item in cluster)
    peak = max(item.decisionScore for item in cluster)
    return weight, len({item.observation.viewId for item in cluster}), peak


def _aggregateCluster(
    cluster: list[ScoredObservation],
    geometry: SphericalGeometry,
    frameWidthPx: int,
    frameHeightPx: int,
    minViewsForCommit: int,
) -> FrameAggregate:
    weights = np.asarray([max(item.decisionScore, 1e-6) for item in cluster], dtype=np.float64)
    weights /= float(np.sum(weights))
    vectors = np.asarray(
        [
            (
                item.observation.bfov.center.x,
                item.observation.bfov.center.y,
                item.observation.bfov.center.z,
            )
            for item in cluster
        ],
        dtype=np.float64,
    )
    centerVector = _normalize(np.sum(vectors * weights[:, np.newaxis], axis=0))
    yawRad, pitchRad = _vectorToYawPitch(centerVector)
    horizontalFov = max(item.observation.bfov.horizontalFovRad for item in cluster)
    verticalFov = max(item.observation.bfov.verticalFovRad for item in cluster)
    spread = max(
        (
            _centerAngle(cluster[0].observation.bfov.center, item.observation.bfov.center)
            for item in cluster
        ),
        default=0.0,
    )
    fusedBfov = BFoV(
        center=makeSphericalPoint(yawRad, pitchRad),
        horizontalFovRad=min(pi - 1e-5, max(1e-4, horizontalFov + 2.0 * spread)),
        verticalFovRad=min(pi - 1e-5, max(1e-4, verticalFov + 2.0 * spread)),
    )
    bbox = geometry.bfovToBbox(fusedBfov, frameWidthPx, frameHeightPx)
    confidence = float(np.clip(np.dot(weights, [item.decisionScore for item in cluster]), 0.0, 1.0))
    representative = max(cluster, key=lambda item: item.decisionScore).observation
    depthSummary = _aggregateDepth(cluster, weights)
    viewIds = tuple(sorted({item.observation.viewId for item in cluster}))
    return FrameAggregate(
        bfov=fusedBfov,
        bbox=bbox,
        confidence=confidence,
        decisionScore=confidence,
        sourceViewIds=viewIds,
        representativeViewId=representative.viewId,
        localBox=representative.localBox,
        depthSummary=depthSummary,
        supported=len(viewIds) >= minViewsForCommit,
    )


def _aggregateDepth(
    cluster: list[ScoredObservation],
    weights: np.ndarray,
) -> DepthSummary | None:
    valid = [
        (item.observation.depthSummary, weight)
        for item, weight in zip(cluster, weights, strict=True)
        if item.observation.depthSummary is not None
    ]
    if not valid:
        return None
    total = sum(weight for _, weight in valid)
    values = [
        sum(summary.medianDepth * weight for summary, weight in valid) / total,
        sum(summary.meanDepth * weight for summary, weight in valid) / total,
        sum(summary.validRatio * weight for summary, weight in valid) / total,
        sum(summary.minDepth * weight for summary, weight in valid) / total,
        sum(summary.maxDepth * weight for summary, weight in valid) / total,
        sum(summary.confidence * weight for summary, weight in valid) / total,
    ]
    return DepthSummary(*values)


def _centerAngle(first: SphericalPoint, second: SphericalPoint) -> float:
    dot = np.clip(first.x * second.x + first.y * second.y + first.z * second.z, -1.0, 1.0)
    return float(acos(float(dot)))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ProtocolError("candidate center vectors must not cancel out")
    return vector / norm


def _vectorToYawPitch(vector: np.ndarray) -> tuple[float, float]:
    return float(np.arctan2(vector[0], vector[2])), float(np.arcsin(np.clip(vector[1], -1.0, 1.0)))


__all__ = ["DecisionGate", "FrameAggregate", "ScoredObservation"]
