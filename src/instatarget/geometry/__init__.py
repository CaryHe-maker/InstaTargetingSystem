"""Geometry package public surface."""

from instatarget.geometry.bfov_projector import BfovProjector
from instatarget.geometry.projection_math import (
    angleToPixelOffsetPx,
    cameraBasis,
    clampPitch,
    erpPixelToSphericalPoint,
    focalLengthPxToFov,
    fovToFocalLengthPx,
    localPixelsToUnitVectors,
    makeSphericalPoint,
    pixelOffsetToAngleRad,
    sphericalPointToErpPixel,
    unitVectorsToErpPixels,
    unitVectorToYawPitch,
    wrapYaw,
)
from instatarget.geometry.seam import (
    containsCircularX,
    minimalCircularInterval,
    splitSeamBox,
    wrapPixelX,
)
from instatarget.geometry.spherical_geometry import SphericalGeometryImpl

__all__ = [
    "BfovProjector",
    "SphericalGeometryImpl",
    "angleToPixelOffsetPx",
    "cameraBasis",
    "clampPitch",
    "containsCircularX",
    "erpPixelToSphericalPoint",
    "focalLengthPxToFov",
    "fovToFocalLengthPx",
    "localPixelsToUnitVectors",
    "makeSphericalPoint",
    "minimalCircularInterval",
    "pixelOffsetToAngleRad",
    "sphericalPointToErpPixel",
    "splitSeamBox",
    "unitVectorToYawPitch",
    "unitVectorsToErpPixels",
    "wrapPixelX",
    "wrapYaw",
]
