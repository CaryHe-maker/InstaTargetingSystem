"""Depth-edge prediction and RGB edge enhancement for the RGB-D mode."""

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
    edgeMask: NDArray[np.bool_]

    @property
    def normalizedDepth(self) -> NDArray[np.float32]:
        return self.normalized

    @property
    def edgeRgb(self) -> NDArray[np.uint8]:
        gray = self.edgeMask.astype(np.uint8) * 255
        return np.repeat(gray[..., None], 3, axis=2)


class DepthPreprocessor(DepthProcessor):
    """Normalize, repair and pseudo-colour local depth maps."""

    def __init__(
        self,
        minValidRatio: float | object = 0.35,
        maxDepthJumpRatio: float = 0.60,
        edgeThreshold: float = 0.20,
        edgeWidthPx: int = 2,
        minContrast: int = 160,
    ) -> None:
        # Accept the immutable core ``DepthConfig`` directly as a convenience
        # for application wiring while keeping scalar arguments usable in tests.
        if not isinstance(minValidRatio, (int, float)) and hasattr(minValidRatio, "minValidRatio"):
            config = minValidRatio
            edge = getattr(config, "edge", None)
            minValidRatio = config.minValidRatio
            maxDepthJumpRatio = config.maxDepthJumpRatio
            if edge is not None:
                edgeThreshold = edge.threshold
                edgeWidthPx = edge.widthPx
                minContrast = edge.minContrast
        if not 0.0 <= minValidRatio <= 1.0 or not 0.0 <= maxDepthJumpRatio <= 1.0:
            raise DepthError("depth validity thresholds must be in [0, 1]")
        if not np.isfinite(edgeThreshold) or not 0.0 <= edgeThreshold <= 1.0:
            raise DepthError("edgeThreshold must be in [0, 1]")
        if edgeWidthPx < 1:
            raise DepthError("edgeWidthPx must be positive")
        if not 0 <= minContrast <= 255:
            raise DepthError("minContrast must be in [0, 255]")
        self.minValidRatio = float(minValidRatio)
        self.maxDepthJumpRatio = float(maxDepthJumpRatio)
        self.edgeThreshold = float(edgeThreshold)
        self.edgeWidthPx = int(edgeWidthPx)
        self.minContrast = int(minContrast)

    def preprocess(self, depth: DepthPlane | NDArray[np.float32]) -> DepthPreprocessResult:
        values, mask = _coerceDepth(depth)
        normalized = self.normalize(values, mask)
        background = self.estimateBackgroundPlane(values, mask)
        maxDepth = float(np.nanmax(values[mask])) if mask.any() else 1.0
        relief = np.clip((background - values) / max(maxDepth, 1e-6), -1.0, 1.0)
        relief[~mask] = 0.0
        # Relative depth changes are more meaningful than absolute metre changes:
        # log depth keeps a large distant background from dominating the edge map.
        logDepth = np.log1p(np.where(mask, values, 0.0))
        edge = _edgeMagnitude(self.normalize(logDepth, mask), mask)
        edgeMask = _dilate(edge >= self.edgeThreshold, self.edgeWidthPx) & mask
        return DepthPreprocessResult(normalized, mask, background, relief, edge, edgeMask)

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
        return plane.astype(np.float32)

    estimateLocalBackground = estimateBackgroundPlane

    def predictEdges(
        self,
        views: DepthPlane | NDArray[np.float32] | Mapping[int, LocalView] | Sequence[LocalView],
    ) -> NDArray[np.bool_] | dict[int, NDArray[np.bool_]]:
        """Return the depth-predicted edge mask, preserving view IDs when supplied."""
        if isinstance(views, DepthPlane) or isinstance(views, np.ndarray):
            return self.preprocess(views).edgeMask
        if isinstance(views, Mapping):
            return {int(viewId): self._edgeView(view) for viewId, view in views.items()}
        return {view.spec.viewId: self._edgeView(view) for view in views}

    def enhanceRgb(self, rgb: NDArray[np.uint8], depth: DepthPlane) -> NDArray[np.uint8]:
        """Change only predicted edge pixels to a high-contrast RGB value."""
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ProtocolError("RGB edge enhancement requires uint8 [H, W, 3]")
        if rgb.shape[:2] != depth.values.shape:
            raise ProtocolError("RGB and depth must be aligned before edge enhancement")
        edgeMask = self.preprocess(depth).edgeMask
        enhanced = rgb.copy()
        if not edgeMask.any():
            return enhanced
        pixels = enhanced[edgeMask]
        inverse = 255 - pixels
        delta = inverse.astype(np.float32) - pixels.astype(np.float32)
        contrast = np.linalg.norm(delta, axis=1)
        toBlack = np.linalg.norm(pixels.astype(np.float32), axis=1)
        toWhite = np.linalg.norm(255.0 - pixels.astype(np.float32), axis=1)
        fallback = np.where((toBlack >= toWhite)[:, None], 0, 255).astype(np.uint8)
        enhanced[edgeMask] = np.where((contrast >= self.minContrast)[:, None], inverse, fallback)
        return enhanced

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

    def _edgeView(self, view: LocalView) -> NDArray[np.bool_]:
        if view.depth is None:
            return np.zeros(view.rgb.shape[:2], dtype=np.bool_)
        return self.preprocess(view.depth).edgeMask


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
    validEdges = edge[mask & np.isfinite(edge)]
    if validEdges.size == 0:
        return np.zeros(normalized.shape, dtype=np.float32)
    scale = float(np.percentile(validEdges, 99.0))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(validEdges.max(initial=0.0))
    scale = max(scale, 1e-6)
    return np.clip(edge / scale, 0.0, 1.0).astype(np.float32)


def _dilate(mask: NDArray[np.bool_], widthPx: int) -> NDArray[np.bool_]:
    if widthPx <= 1 or not mask.any():
        return mask.copy()
    radius = widthPx - 1
    padded = np.pad(mask, radius, mode="constant")
    result = np.zeros_like(mask)
    height, width = mask.shape
    for yOffset in range(2 * radius + 1):
        for xOffset in range(2 * radius + 1):
            result |= padded[yOffset : yOffset + height, xOffset : xOffset + width]
    return result


__all__ = ["DepthPreprocessResult", "DepthPreprocessor"]
