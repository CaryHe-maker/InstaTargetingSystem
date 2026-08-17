"""Adapter for the second HiT branch and a small dependency-free fallback encoder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from instatarget.core.errors import ModelError, ProtocolError
from instatarget.core.types import BBoxXYWH


@dataclass(frozen=True, slots=True)
class DepthFeatures:
    """Compact depth representation used by the fusion head/template cache."""

    vector: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class DepthPrediction:
    """Normalized output of a depth HiT adapter."""

    depthScore: float
    modelScore: float | None = None
    bbox: BBoxXYWH | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.depthScore) or not 0.0 <= self.depthScore <= 1.0:
            raise ModelError("depthScore must be finite and in [0, 1]")


class DepthEncoder:
    """Encode pseudo-colour depth RGB and optionally delegate to a HiT session.

    The fallback path is intentionally deterministic and useful for CPU tests;
    production deployments can pass a session implementing ``encodeTemplate``
    and ``infer``.
    """

    def __init__(self, session: Any | None = None) -> None:
        self._session = session
        self._closed = False

    @property
    def supportsOnlineTemplates(self) -> bool:
        return bool(getattr(self._session, "supportsOnlineTemplates", True))

    def encode(self, depthRgb: NDArray[np.uint8]) -> DepthFeatures:
        self._requireOpen()
        _requireDepthRgb(depthRgb)
        gray = depthRgb.astype(np.float32).mean(axis=2) / 255.0
        # Global moments plus a coarse spatial grid are stable across resolutions.
        h, w = gray.shape
        cells = [gray]
        for ys in np.array_split(np.arange(h), 2):
            for xs in np.array_split(np.arange(w), 2):
                cells.append(gray[np.ix_(ys, xs)])
        vector = np.asarray(
            [float(np.mean(cell)) if cell.size else 0.0 for cell in cells]
            + [float(np.std(gray)), float(np.percentile(gray, 90) - np.percentile(gray, 10))],
            dtype=np.float32,
        )
        return DepthFeatures(vector=vector)

    def encodeTemplate(
        self,
        depthRgb: NDArray[np.uint8],
        bbox: BBoxXYWH | None = None,
    ) -> object:
        self._requireOpen()
        _requireDepthRgb(depthRgb)
        if self._session is not None and hasattr(self._session, "encodeTemplate"):
            try:
                return (
                    self._session.encodeTemplate(depthRgb)
                    if bbox is None
                    else self._session.encodeTemplate(depthRgb, bbox)
                )
            except Exception as error:
                raise ModelError(f"depth HiT template encoding failed: {error}") from error
        return self.encode(depthRgb)

    def infer(
        self,
        depthRgb: NDArray[np.uint8],
        templateFeatures: Sequence[object] = (),
    ) -> DepthPrediction:
        self._requireOpen()
        _requireDepthRgb(depthRgb)
        if self._session is not None:
            try:
                raw = self._session.infer(depthRgb, templateFeatures)
            except Exception as error:
                raise ModelError(f"depth HiT inference failed: {error}") from error
            return _coercePrediction(raw)
        current = self.encode(depthRgb).vector
        if templateFeatures:
            vectors = [_featureVector(feature) for feature in templateFeatures]
            reference = np.mean(np.stack(vectors), axis=0)
            distance = float(np.linalg.norm(current - reference) / np.sqrt(current.size))
            score = float(np.exp(-4.0 * distance))
        else:
            score = float(np.clip(np.mean(current), 0.0, 1.0))
        return DepthPrediction(depthScore=score, modelScore=score)

    def inferBatch(
        self,
        depthRgbs: Sequence[NDArray[np.uint8]],
        templateFeatures: Sequence[object] = (),
    ) -> tuple[DepthPrediction, ...]:
        self._requireOpen()
        images = tuple(depthRgbs)
        for image in images:
            _requireDepthRgb(image)
        if not images:
            return ()
        batchInfer = getattr(self._session, "inferBatch", None)
        if callable(batchInfer):
            try:
                rawPredictions = tuple(batchInfer(images, templateFeatures))
            except Exception as error:
                raise ModelError(f"depth HiT batch inference failed: {error}") from error
            if len(rawPredictions) != len(images):
                raise ModelError(
                    "depth HiT returned an invalid batch size: "
                    f"expected={len(images)}, actual={len(rawPredictions)}"
                )
            return tuple(_coercePrediction(value) for value in rawPredictions)
        return tuple(self.infer(image, templateFeatures) for image in images)

    def close(self) -> None:
        if self._closed:
            return
        if self._session is not None and hasattr(self._session, "close"):
            self._session.close()
        self._closed = True

    def _requireOpen(self) -> None:
        if self._closed:
            raise ProtocolError("depth encoder is closed")


def _requireDepthRgb(depthRgb: NDArray[np.uint8]) -> None:
    if not isinstance(depthRgb, np.ndarray) or depthRgb.dtype != np.uint8:
        raise ProtocolError("depth encoder input must be uint8 RGB")
    if depthRgb.ndim != 3 or depthRgb.shape[2] != 3 or 0 in depthRgb.shape[:2]:
        raise ProtocolError("depth encoder input must have shape [H, W, 3]")


def _featureVector(value: object) -> NDArray[np.float32]:
    if isinstance(value, DepthFeatures):
        return value.vector
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).reshape(-1)
    raise ModelError(f"unsupported depth template feature type: {type(value).__name__}")


def _coercePrediction(value: object) -> DepthPrediction:
    if isinstance(value, DepthPrediction):
        return value
    if isinstance(value, (float, int, np.floating, np.integer)):
        return DepthPrediction(float(value))
    score = getattr(value, "depthScore", getattr(value, "appearanceScore", None))
    if score is None:
        raise ModelError("depth HiT returned no depth score")
    return DepthPrediction(
        depthScore=float(score),
        modelScore=getattr(value, "modelScore", None),
        bbox=getattr(value, "bbox", None),
    )


__all__ = ["DepthEncoder", "DepthFeatures", "DepthPrediction"]
