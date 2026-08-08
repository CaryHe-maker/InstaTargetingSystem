"""Deterministic NumPy depth processing used by the RGB-D tracker branch.

The preprocessor deliberately owns no tracking state.  It converts an aligned
``DepthPlane`` into robust normalized values, a local background estimate and
an RGB pseudo-colour image suitable for a second visual encoder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from instatarget.core.errors import DepthError, ProtocolError
from instatarget.core.protocols import DepthProcessor
from instatarget.core.types import BBoxXYWH, DepthPlane, DepthSummary, FramePacket, LocalView


@dataclass(frozen=True, slots=True)
class DepthPreprocessResult:
    normalized: NDArray[np.float32]
    validMask: NDArray[np.bool_]
    backgroundPlane: NDArray[np.float32]
    relief: NDArray[np.float32]
    edge: NDArray[np.float32]
    depthRgb: NDArray[np.uint8]

    @property
    def normalizedDepth(self) -> NDArray[np.float32]:
        return self.normalized


class DepthPreprocessor(DepthProcessor):
    """Normalize, repair and pseudo-colour local depth maps."""

    def __init__(
        self,
        minValidRatio: float | object = 0.35,
        maxDepthJumpRatio: float = 0.60,
        colorizationMode: str = "relief",
        nearBrightness: float = 0.95,
        farBrightness: float = 0.20,
        reliefGain: float = 1.0,
        edgeGain: float = 0.35,
        smoothingKernel: int = 7,
    ) -> None:
        # Accept the immutable core ``DepthConfig`` directly as a convenience
        # for application wiring while keeping scalar arguments usable in tests.
        if not isinstance(minValidRatio, (int, float)) and hasattr(minValidRatio, "minValidRatio"):
            config = minValidRatio
            colorization = getattr(config, "colorization", None)
            minValidRatio = config.minValidRatio
            maxDepthJumpRatio = config.maxDepthJumpRatio
            if colorization is not None:
                colorizationMode = colorization.mode
                nearBrightness = colorization.nearBrightness
                farBrightness = colorization.farBrightness
                reliefGain = colorization.reliefGain
                edgeGain = colorization.edgeGain
                smoothingKernel = colorization.smoothingKernel
        if not 0.0 <= minValidRatio <= 1.0 or not 0.0 <= maxDepthJumpRatio <= 1.0:
            raise DepthError("depth validity thresholds must be in [0, 1]")
        if colorizationMode not in {"relief", "grayscale"}:
            raise DepthError(f"unsupported depth colorization mode: {colorizationMode}")
        for name, value in (
            ("nearBrightness", nearBrightness),
            ("farBrightness", farBrightness),
            ("reliefGain", reliefGain),
            ("edgeGain", edgeGain),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise DepthError(f"{name} must be finite and non-negative")
        if smoothingKernel < 1 or smoothingKernel % 2 == 0:
            raise DepthError("smoothingKernel must be a positive odd integer")
        self.minValidRatio = float(minValidRatio)
        self.maxDepthJumpRatio = float(maxDepthJumpRatio)
        self.colorizationMode = colorizationMode
        self.nearBrightness = float(nearBrightness)
        self.farBrightness = float(farBrightness)
        self.reliefGain = float(reliefGain)
        self.edgeGain = float(edgeGain)
        self.smoothingKernel = int(smoothingKernel)

    def preprocess(self, depth: DepthPlane | NDArray[np.float32]) -> DepthPreprocessResult:
        values, mask = _coerceDepth(depth)
        normalized = self.normalize(values, mask)
        background = self.estimateBackgroundPlane(values, mask)
        maxDepth = float(np.nanmax(values[mask])) if mask.any() else 1.0
        relief = np.clip((background - values) / max(maxDepth, 1e-6), -1.0, 1.0)
        relief[~mask] = 0.0
        edge = _edgeMagnitude(normalized, mask)
        depthRgb = self._colorizeArrays(normalized, mask, relief, edge)
        return DepthPreprocessResult(normalized, mask, background, relief, edge, depthRgb)

    def normalize(
        self,
        depth: DepthPlane | NDArray[np.float32],
        validMask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float32]:
        values, mask = _coerceDepth(depth, validMask)
        result = np.zeros(values.shape, dtype=np.float32)
        if not mask.any():
            return result
        valid = values[mask].astype(np.float64)
        low, high = np.percentile(valid, (2.0, 98.0))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(valid.min()), float(valid.max())
        scale = max(high - low, 1e-6)
        result[mask] = np.clip((values[mask] - low) / scale, 0.0, 1.0)
        return result

    def estimateBackgroundPlane(
        self,
        depth: DepthPlane | NDArray[np.float32],
        validMask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float32]:
        values, mask = _coerceDepth(depth, validMask)
        if not mask.any():
            return np.zeros(values.shape, dtype=np.float32)
        height, width = values.shape
        yy, xx = np.mgrid[:height, :width]
        valid = mask & np.isfinite(values)
        if int(valid.sum()) < 3:
            return np.full(values.shape, float(np.median(values[valid])), dtype=np.float32)
        design = np.column_stack((xx[valid], yy[valid], np.ones(int(valid.sum()))))
        coefficients, *_ = np.linalg.lstsq(design, values[valid].astype(np.float64), rcond=None)
        plane = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
        median = float(np.median(values[valid]))
        plane = np.clip(plane, 0.0, max(float(np.max(values[valid])) * 1.5, median))
        return _boxSmooth(plane.astype(np.float32), self.smoothingKernel)

    estimateLocalBackground = estimateBackgroundPlane

    def colorize(
        self,
        views: DepthPlane | NDArray[np.float32] | Mapping[int, LocalView] | Sequence[LocalView],
    ) -> NDArray[np.uint8] | dict[int, NDArray[np.uint8]]:
        """Return RGB uint8 pseudo-colour output, preserving view IDs when supplied."""
        if isinstance(views, DepthPlane) or isinstance(views, np.ndarray):
            return self.preprocess(views).depthRgb
        if isinstance(views, Mapping):
            return {int(viewId): self._colorizeView(view) for viewId, view in views.items()}
        return {view.spec.viewId: self._colorizeView(view) for view in views}

    def summarize(self, frame: FramePacket, bbox: BBoxXYWH) -> DepthSummary | None:
        if frame.depth is None:
            return None
        return self.summarizePlane(frame.depth, bbox)

    def summarizeLocal(self, view: LocalView, localBox: BBoxXYWH) -> DepthSummary | None:
        if view.depth is None:
            return None
        return self.summarizePlane(view.depth, localBox)

    def summarizePlane(self, depth: DepthPlane, bbox: BBoxXYWH) -> DepthSummary | None:
        values, mask = _coerceDepth(depth)
        x0 = max(0, int(np.floor(bbox.xPx)))
        y0 = max(0, int(np.floor(bbox.yPx)))
        x1 = min(values.shape[1], int(np.ceil(bbox.xPx + bbox.widthPx)))
        y1 = min(values.shape[0], int(np.ceil(bbox.yPx + bbox.heightPx)))
        if x1 <= x0 or y1 <= y0:
            raise ProtocolError("depth summary box has no intersection with the view")
        regionMask = mask[y0:y1, x0:x1]
        region = values[y0:y1, x0:x1][regionMask]
        validRatio = float(regionMask.mean())
        if region.size == 0 or validRatio < self.minValidRatio:
            return None
        median = float(np.median(region))
        mean = float(np.mean(region))
        minDepth = float(np.min(region))
        maxDepth = float(np.max(region))
        spread = (maxDepth - minDepth) / max(median, 1e-6)
        consistency = float(np.clip(1.0 - spread / max(self.maxDepthJumpRatio, 1e-6), 0.0, 1.0))
        confidence = float(np.clip(validRatio * (0.5 + 0.5 * consistency), 0.0, 1.0))
        return DepthSummary(median, mean, validRatio, minDepth, maxDepth, confidence)

    def score(self, summary: DepthSummary | None) -> float:
        return 0.0 if summary is None else float(np.clip(summary.confidence, 0.0, 1.0))

    def _colorizeView(self, view: LocalView) -> NDArray[np.uint8]:
        if view.depth is None:
            return np.zeros((*view.rgb.shape[:2], 3), dtype=np.uint8)
        return self.preprocess(view.depth).depthRgb

    def _colorizeArrays(
        self,
        normalized: NDArray[np.float32],
        mask: NDArray[np.bool_],
        relief: NDArray[np.float32],
        edge: NDArray[np.float32],
    ) -> NDArray[np.uint8]:
        # Inverse depth is bright/near and dark/far; relief and edges add shape.
        brightness = self.farBrightness + (
            self.nearBrightness - self.farBrightness
        ) * (1.0 - normalized)
        if self.colorizationMode == "relief":
            brightness = brightness + self.reliefGain * 0.15 * np.maximum(relief, 0.0)
        brightness = np.clip(brightness + self.edgeGain * edge, 0.0, 1.0)
        brightness[~mask] = 0.0
        gray = np.rint(brightness * 255.0).astype(np.uint8)
        # A neutral RGB image is intentional: the depth encoder must not infer a rainbow palette.
        return np.repeat(gray[..., None], 3, axis=2)


def _coerceDepth(
    depth: DepthPlane | NDArray[np.float32],
    validMask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    if isinstance(depth, DepthPlane):
        values, mask = depth.values, depth.validMask
    elif isinstance(depth, np.ndarray):
        if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
            raise DepthError("depth must be a floating-point array with shape [H, W]")
        values = depth.astype(np.float32, copy=False)
        mask = np.isfinite(values) & (values >= 0.0)
        if validMask is not None:
            mask &= validMask
    else:
        raise DepthError("unsupported depth input")
    if values.ndim != 2 or mask.shape != values.shape:
        raise DepthError("depth values and mask must have matching [H, W] shapes")
    return values, mask.astype(np.bool_, copy=False)


def _edgeMagnitude(normalized: NDArray[np.float32], mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    if min(normalized.shape) < 2:
        return np.zeros(normalized.shape, dtype=np.float32)
    gy, gx = np.gradient(normalized.astype(np.float32, copy=False))
    edge = np.sqrt(gx * gx + gy * gy)
    edge[~mask] = 0.0
    maximum = float(edge.max(initial=0.0))
    return np.clip(edge / max(maximum, 1e-6), 0.0, 1.0).astype(np.float32)


def _boxSmooth(values: NDArray[np.float32], kernel: int) -> NDArray[np.float32]:
    if kernel <= 1 or min(values.shape) < 2:
        return values
    radius = kernel // 2
    padded = np.pad(values.astype(np.float64), radius, mode="edge")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)))
    height, width = values.shape
    y0 = np.arange(height)
    y1 = y0 + kernel
    x0 = np.arange(width)
    x1 = x0 + kernel
    total = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    return (total / float(kernel * kernel)).astype(np.float32)


__all__ = ["DepthPreprocessResult", "DepthPreprocessor"]
