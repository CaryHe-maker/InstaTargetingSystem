"""Concrete spherical geometry facade built on top of the projection helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from instatarget.core.errors import GeometryError
from instatarget.core.protocols import SphericalGeometry as SphericalGeometryProtocol
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    FramePacket,
    LocalBoxProjection,
    LocalView,
    ViewSpec,
)
from instatarget.geometry.bfov_projector import BfovProjector
from instatarget.geometry.projection_math import (
    cameraBasis,
    erpPixelToSphericalPoint,
    localPixelsToUnitVectors,
    makeSphericalPoint,
    unitVectorsToErpPixels,
    unitVectorToYawPitch,
)
from instatarget.geometry.seam import minimalCircularInterval, wrapPixelX


@dataclass(slots=True)
class SphericalGeometryImpl(SphericalGeometryProtocol):
    """Default geometry implementation for ERP crops and BFoV envelopes."""

    boundarySamplesPerEdge: int = 65
    _projector: BfovProjector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _requireBoundarySamplesPerEdge(self.boundarySamplesPerEdge)
        self._projector = BfovProjector(self.boundarySamplesPerEdge)

    def bboxToBfov(self, bbox: BBoxXYWH, frameWidthPx: int, frameHeightPx: int) -> BFoV:
        _requireFrameDimensions(frameWidthPx, frameHeightPx)
        _requireErpBox(bbox, frameWidthPx, frameHeightPx)
        sampleX, sampleY = _sampleErpBoxBoundary(
            bbox.xPx,
            bbox.yPx,
            bbox.widthPx,
            bbox.heightPx,
            self.boundarySamplesPerEdge,
        )
        vectors = _erpSamplesToVectors(sampleX, sampleY, frameWidthPx, frameHeightPx)
        return _fitBfovFromVectors(vectors)

    def cropViews(self, frame: FramePacket, specs: Sequence[ViewSpec]) -> list[LocalView]:
        return self._projector.cropViews(frame, specs)

    def localBoxToBfov(self, localBox: BBoxXYWH, spec: ViewSpec) -> BFoV:
        _requireViewSpec(spec)
        _requireLocalBox(localBox, spec.outputWidthPx, spec.outputHeightPx)
        sampleX, sampleY = _sampleErpBoxBoundary(
            localBox.xPx,
            localBox.yPx,
            localBox.widthPx,
            localBox.heightPx,
            self.boundarySamplesPerEdge,
        )
        vectors = localPixelsToUnitVectors(
            sampleX,
            sampleY,
            spec.bfov,
            spec.outputWidthPx,
            spec.outputHeightPx,
        )
        return _fitBfovFromVectors(vectors)

    def projectLocalBoxBoundary(
        self,
        localBox: BBoxXYWH,
        spec: ViewSpec,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> LocalBoxProjection:
        """Project a local boundary once and fit both output envelopes from it."""
        _requireViewSpec(spec)
        _requireLocalBox(localBox, spec.outputWidthPx, spec.outputHeightPx)
        _requireFrameDimensions(frameWidthPx, frameHeightPx)
        sampleX, sampleY = _sampleErpBoxBoundary(
            localBox.xPx,
            localBox.yPx,
            localBox.widthPx,
            localBox.heightPx,
            self.boundarySamplesPerEdge,
        )
        vectors = localPixelsToUnitVectors(
            sampleX,
            sampleY,
            spec.bfov,
            spec.outputWidthPx,
            spec.outputHeightPx,
        )
        bfov = _fitBfovFromVectors(vectors)
        erpX, erpY = unitVectorsToErpPixels(vectors, frameWidthPx, frameHeightPx)
        xPx, widthPx = minimalCircularInterval(erpX, frameWidthPx)
        yMin = float(np.min(erpY))
        yMax = float(np.max(erpY))
        bbox = BBoxXYWH(
            xPx=xPx,
            yPx=yMin,
            widthPx=widthPx,
            heightPx=max(yMax - yMin, float(np.finfo(np.float64).eps)),
        )
        indirectBbox = self.bfovToBbox(bfov, frameWidthPx, frameHeightPx)
        directArea = bbox.widthPx * bbox.heightPx
        indirectArea = indirectBbox.widthPx * indirectBbox.heightPx
        sphericalBoundary = tuple(
            makeSphericalPoint(
                float(np.arctan2(vector[0], vector[2])),
                float(np.arcsin(np.clip(vector[1], -1.0, 1.0))),
            )
            for vector in vectors
        )
        return LocalBoxProjection(
            bfov=bfov,
            bbox=bbox,
            sphericalBoundary=sphericalBoundary,
            erpBoundary=tuple(
                (float(xValue), float(yValue))
                for xValue, yValue in zip(erpX, erpY, strict=True)
            ),
            indirectBbox=indirectBbox,
            envelopeInflation=float(indirectArea / max(directArea, 1e-12)),
        )

    def bfovToBbox(self, bfov: BFoV, frameWidthPx: int, frameHeightPx: int) -> BBoxXYWH:
        _requireFrameDimensions(frameWidthPx, frameHeightPx)
        sampleX, sampleY = _sampleCanonicalRectBoundary(self.boundarySamplesPerEdge)
        vectors = localPixelsToUnitVectors(sampleX, sampleY, bfov, 1, 1)
        erpX, erpY = unitVectorsToErpPixels(vectors, frameWidthPx, frameHeightPx)
        xPx, widthPx = minimalCircularInterval(erpX, frameWidthPx)
        yMin = float(np.min(erpY))
        yMax = float(np.max(erpY))
        bbox = BBoxXYWH(xPx=xPx, yPx=yMin, widthPx=widthPx, heightPx=max(yMax - yMin, 0.0))
        _requireErpBox(bbox, frameWidthPx, frameHeightPx)
        return bbox


def _fitBfovFromVectors(vectors: NDArray[np.float64]) -> BFoV:
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.isfinite(vectors).all():
        raise GeometryError("boundary samples must be a finite array with final dimension 3")
    meanVector = np.mean(vectors, axis=0)
    meanNorm = float(np.linalg.norm(meanVector))
    if meanNorm == 0.0:
        meanVector = vectors[0]
        meanNorm = float(np.linalg.norm(meanVector))
    if meanNorm == 0.0:
        raise GeometryError("boundary samples must contain at least one non-zero vector")
    center = makeSphericalPoint(*unitVectorToYawPitch(tuple(meanVector)))
    horizontalAngles = np.empty(0, dtype=np.float64)
    verticalAngles = np.empty(0, dtype=np.float64)
    for _ in range(4):
        forward, right, up = cameraBasis(
            BFoV(center=center, horizontalFovRad=1.0, verticalFovRad=1.0)
        )
        forwardDots = vectors @ forward
        horizontalAngles = np.arctan2(vectors @ right, forwardDots)
        verticalAngles = np.arctan2(vectors @ up, forwardDots)
        horizontalOffset = float(
            (np.min(horizontalAngles) + np.max(horizontalAngles)) / 2.0
        )
        verticalOffset = float((np.min(verticalAngles) + np.max(verticalAngles)) / 2.0)
        if max(abs(horizontalOffset), abs(verticalOffset)) < 1e-10:
            break
        shifted = (
            forward
            + np.tan(horizontalOffset) * right
            + np.tan(verticalOffset) * up
        )
        shifted /= np.linalg.norm(shifted)
        center = makeSphericalPoint(*unitVectorToYawPitch(tuple(shifted)))

    forward, right, up = cameraBasis(
        BFoV(center=center, horizontalFovRad=1.0, verticalFovRad=1.0)
    )
    forwardDots = vectors @ forward
    horizontalAngles = np.arctan2(vectors @ right, forwardDots)
    verticalAngles = np.arctan2(vectors @ up, forwardDots)
    horizontalFovRad = float(np.max(horizontalAngles) - np.min(horizontalAngles))
    verticalFovRad = float(np.max(verticalAngles) - np.min(verticalAngles))
    if not 0.0 < horizontalFovRad < np.pi:
        raise GeometryError(f"horizontal BFoV span is invalid: {horizontalFovRad}")
    if not 0.0 < verticalFovRad < np.pi:
        raise GeometryError(f"vertical BFoV span is invalid: {verticalFovRad}")
    return BFoV(
        center=center,
        horizontalFovRad=horizontalFovRad,
        verticalFovRad=verticalFovRad,
        rollRad=0.0,
    )


def _sampleErpBoxBoundary(
    xPx: float,
    yPx: float,
    widthPx: float,
    heightPx: float,
    samplesPerEdge: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    _requireBoundarySamplesPerEdge(samplesPerEdge)
    edge = np.linspace(0.0, 1.0, samplesPerEdge, dtype=np.float64)
    topX = xPx + edge * widthPx
    rightX = np.full_like(edge, xPx + widthPx)
    bottomX = xPx + (1.0 - edge) * widthPx
    leftX = np.full_like(edge, xPx)
    topY = np.full_like(edge, yPx)
    rightY = yPx + edge * heightPx
    bottomY = np.full_like(edge, yPx + heightPx)
    leftY = yPx + (1.0 - edge) * heightPx
    sampleX = np.concatenate((topX, rightX, bottomX, leftX))
    sampleY = np.concatenate((topY, rightY, bottomY, leftY))
    return sampleX, sampleY


def _sampleCanonicalRectBoundary(
    samplesPerEdge: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    edge = np.linspace(0.0, 1.0, samplesPerEdge, dtype=np.float64)
    topX = edge
    rightX = np.ones_like(edge)
    bottomX = 1.0 - edge
    leftX = np.zeros_like(edge)
    topY = np.zeros_like(edge)
    rightY = edge
    bottomY = np.ones_like(edge)
    leftY = 1.0 - edge
    return (
        np.concatenate((topX, rightX, bottomX, leftX)),
        np.concatenate((topY, rightY, bottomY, leftY)),
    )


def _erpSamplesToVectors(
    sampleX: NDArray[np.float64],
    sampleY: NDArray[np.float64],
    frameWidthPx: int,
    frameHeightPx: int,
) -> NDArray[np.float64]:
    if sampleX.shape != sampleY.shape:
        raise GeometryError(
            f"sample coordinate shapes must match: {sampleX.shape} != {sampleY.shape}"
        )
    points = [
        erpPixelToSphericalPoint(
            float(wrapPixelX(float(xPx), frameWidthPx)),
            float(yPx),
            frameWidthPx,
            frameHeightPx,
        )
        for xPx, yPx in zip(sampleX, sampleY, strict=True)
    ]
    return np.asarray([(point.x, point.y, point.z) for point in points], dtype=np.float64)


def _requireBoundarySamplesPerEdge(samples: int) -> None:
    if isinstance(samples, bool) or samples < 2:
        raise GeometryError(
            f"boundarySamplesPerEdge must be an integer >= 2, actual={samples}"
        )


def _requireFrameDimensions(frameWidthPx: int, frameHeightPx: int) -> None:
    if isinstance(frameWidthPx, bool) or isinstance(frameHeightPx, bool):
        raise GeometryError("frame dimensions must be integers")
    if frameWidthPx <= 0 or frameHeightPx <= 0:
        raise GeometryError(
            f"frame dimensions must be positive, actual=({frameWidthPx}, {frameHeightPx})"
        )


def _requireErpBox(bbox: BBoxXYWH, frameWidthPx: int, frameHeightPx: int) -> None:
    if bbox.widthPx <= 0.0 or bbox.heightPx <= 0.0:
        raise GeometryError("bbox dimensions must be positive")
    if bbox.widthPx > frameWidthPx or bbox.heightPx > frameHeightPx:
        raise GeometryError("bbox dimensions cannot exceed the frame size")
    if not np.isfinite([bbox.xPx, bbox.yPx, bbox.widthPx, bbox.heightPx]).all():
        raise GeometryError("bbox coordinates must be finite")
    if bbox.yPx < 0.0 or bbox.yPx + bbox.heightPx > frameHeightPx:
        raise GeometryError("bbox must stay within the vertical frame range")


def _requireLocalBox(localBox: BBoxXYWH, viewWidthPx: int, viewHeightPx: int) -> None:
    if localBox.widthPx <= 0.0 or localBox.heightPx <= 0.0:
        raise GeometryError("local box dimensions must be positive")
    if localBox.widthPx > viewWidthPx or localBox.heightPx > viewHeightPx:
        raise GeometryError("local box dimensions cannot exceed the view size")
    if not np.isfinite([localBox.xPx, localBox.yPx, localBox.widthPx, localBox.heightPx]).all():
        raise GeometryError("local box coordinates must be finite")
    if localBox.xPx < 0.0 or localBox.yPx < 0.0:
        raise GeometryError("local box must start within the view bounds")
    if (
        localBox.xPx + localBox.widthPx > viewWidthPx
        or localBox.yPx + localBox.heightPx > viewHeightPx
    ):
        raise GeometryError("local box must stay within the local view bounds")


def _requireViewSpec(spec: ViewSpec) -> None:
    if spec.outputWidthPx <= 0 or spec.outputHeightPx <= 0:
        raise GeometryError("view dimensions must be positive")


__all__ = ["SphericalGeometryImpl"]
