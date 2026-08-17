"""Deterministic spherical clustering for same-frame search refinement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import acos

import numpy as np

from instatarget.core.errors import ProtocolError
from instatarget.core.types import ProjectedObservation, SphericalPoint

CLASSIFY_RADIUS_RAD = np.deg2rad(30.0)


@dataclass(frozen=True, slots=True)
class ClusterCenter:
    """A refinement target and the evidence that selected it."""

    center: SphericalPoint
    memberViewIds: tuple[int, ...]
    memberCount: int
    meanSingleScore: float


@dataclass(slots=True)
class _Cluster:
    observations: list[ProjectedObservation]

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(item.viewId for item in self.observations))


class Classifier:
    """Cluster projected observation centers with a hard 30 degree radius."""

    def __init__(self, radiusRad: float = CLASSIFY_RADIUS_RAD, maxClusters: int = 3) -> None:
        if not np.isfinite(radiusRad) or radiusRad <= 0.0 or radiusRad >= np.pi:
            raise ValueError("classifier radius must be in (0, pi)")
        if maxClusters <= 0:
            raise ValueError("classifier maxClusters must be positive")
        self._radius = float(radiusRad)
        self._maxClusters = int(maxClusters)

    def classify(
        self,
        observations: Sequence[ProjectedObservation],
    ) -> tuple[ClusterCenter, ...]:
        if not observations:
            return ()
        if len({item.viewId for item in observations}) != len(observations):
            raise ProtocolError("classifier observations must have unique viewIds")

        # The highest scoring item is the deterministic seed.  The small search rounds
        # contain at most six items, so an exhaustive merge pass is both cheap and clear.
        ordered = sorted(
            observations,
            key=lambda item: (-_singleScore(item), item.viewId),
        )
        clusters = [_Cluster([item]) for item in ordered]
        while True:
            bestPair: tuple[float, int, int] | None = None
            for firstIndex, first in enumerate(clusters):
                for secondIndex in range(firstIndex + 1, len(clusters)):
                    merged = _Cluster(first.observations + clusters[secondIndex].observations)
                    if not _withinRadius(merged.observations, self._radius):
                        continue
                    key = (
                        -float(len(merged.observations)),
                        min(first.ids + clusters[secondIndex].ids),
                        max(first.ids + clusters[secondIndex].ids),
                    )
                    if bestPair is None or key < bestPair:
                        bestPair = (key[0], firstIndex, secondIndex)
            if bestPair is None:
                break
            _, firstIndex, secondIndex = bestPair
            clusters[firstIndex].observations.extend(clusters[secondIndex].observations)
            del clusters[secondIndex]

        ranked = sorted(
            clusters,
            key=lambda cluster: (
                -len(cluster.observations),
                -float(np.mean([_singleScore(item) for item in cluster.observations])),
                cluster.ids,
            ),
        )
        return tuple(_toCenter(cluster) for cluster in ranked[: self._maxClusters])


def classify(
    observations: Sequence[ProjectedObservation],
    *,
    radiusRad: float = CLASSIFY_RADIUS_RAD,
    maxClusters: int = 3,
) -> tuple[ClusterCenter, ...]:
    return Classifier(radiusRad=radiusRad, maxClusters=maxClusters).classify(observations)


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


def _vector(observation: ProjectedObservation) -> np.ndarray:
    return np.asarray(
        (observation.bfov.center.x, observation.bfov.center.y, observation.bfov.center.z),
        dtype=np.float64,
    )


def _centerVector(observations: Sequence[ProjectedObservation]) -> np.ndarray:
    vectors = np.asarray([_vector(item) for item in observations], dtype=np.float64)
    weights = np.asarray([max(_singleScore(item), 1e-6) for item in observations])
    result = np.sum(vectors * weights[:, np.newaxis], axis=0)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        result = vectors[0]
        norm = float(np.linalg.norm(result))
    return result / norm


def _withinRadius(observations: Sequence[ProjectedObservation], radius: float) -> bool:
    center = _centerVector(observations)
    return all(
        acos(float(np.clip(np.dot(center, _vector(item)), -1.0, 1.0))) <= radius + 1e-10
        for item in observations
    )


def _toCenter(cluster: _Cluster) -> ClusterCenter:
    center = _centerVector(cluster.observations)
    yaw = float(np.arctan2(center[0], center[2]))
    pitch = float(np.arcsin(np.clip(center[1], -1.0, 1.0)))
    from instatarget.geometry.projection_math import makeSphericalPoint

    return ClusterCenter(
        center=makeSphericalPoint(yaw, pitch),
        memberViewIds=cluster.ids,
        memberCount=len(cluster.observations),
        meanSingleScore=float(np.mean([_singleScore(item) for item in cluster.observations])),
    )


__all__ = ["CLASSIFY_RADIUS_RAD", "ClusterCenter", "Classifier", "classify"]
