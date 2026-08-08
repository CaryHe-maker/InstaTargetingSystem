"""Stage-oriented PNG recorder for manual tracking diagnostics."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from instatarget.core.config import VisualizationConfig
from instatarget.core.errors import ProtocolError
from instatarget.core.types import FramePacket, LocalObservation, LocalView, ProjectedObservation
from instatarget.visualization.image import drawBoxRgb
from instatarget.visualization.png import writeRgbPng

LOCAL_RGB_STAGE = "local_rgb"
DEPTH_RGB_STAGE = "depth_rgb"
BACKEND_BOX_STAGE = "backend_box"
GEOMETRY_BOX_STAGE = "geometry_box"


@dataclass(frozen=True, slots=True)
class VisualizationRecorder:
    """Write selected intermediate results without changing tracking decisions."""

    config: VisualizationConfig

    def recordLocalRgb(
        self,
        frame: FramePacket,
        views: Sequence[LocalView],
    ) -> tuple[Path, ...]:
        """Write the raw RGB crop for each local view."""
        if not self._active(LOCAL_RGB_STAGE):
            return ()
        _requireUniqueViewIds(views)
        return tuple(
            writeRgbPng(self._artifactPath(frame, LOCAL_RGB_STAGE, view.spec.viewId), view.rgb)
            for view in views
        )

    def recordDepthRgb(
        self,
        frame: FramePacket,
        depthRgbByViewId: Mapping[int, NDArray[np.uint8]],
    ) -> tuple[Path, ...]:
        """Write already-converted depth RGB images without transforming them."""
        if not self._active(DEPTH_RGB_STAGE):
            return ()
        _requireViewIds(depthRgbByViewId)
        return tuple(
            writeRgbPng(self._artifactPath(frame, DEPTH_RGB_STAGE, viewId), depthRgb)
            for viewId, depthRgb in sorted(depthRgbByViewId.items())
        )

    def recordBackendBoxes(
        self,
        frame: FramePacket,
        views: Sequence[LocalView],
        observations: Sequence[LocalObservation],
    ) -> tuple[Path, ...]:
        """Write each backend-local target box over its source local RGB view."""
        if not self._active(BACKEND_BOX_STAGE):
            return ()
        viewById = _indexViews(views)
        _requireUniqueObservationIds(observations)
        artifacts: list[Path] = []
        for observation in observations:
            view = _requireView(viewById, observation.viewId)
            annotatedRgb = drawBoxRgb(view.rgb, observation.bbox)
            artifacts.append(
                writeRgbPng(
                    self._artifactPath(frame, BACKEND_BOX_STAGE, observation.viewId),
                    annotatedRgb,
                )
            )
        return tuple(artifacts)

    def recordGeometryBoxes(
        self,
        frame: FramePacket,
        observations: Sequence[ProjectedObservation],
    ) -> tuple[Path, ...]:
        """Write each geometry-projected target box over the original ERP RGB frame."""
        if not self._active(GEOMETRY_BOX_STAGE):
            return ()
        _requireUniqueObservationIds(observations)
        return tuple(
            writeRgbPng(
                self._artifactPath(frame, GEOMETRY_BOX_STAGE, observation.viewId),
                drawBoxRgb(frame.rgb, observation.bbox, wrapHorizontal=True),
            )
            for observation in observations
        )

    def _active(self, stage: str) -> bool:
        return self.config.enabled and stage in self.config.stages

    def _artifactPath(self, frame: FramePacket, stage: str, viewId: int) -> Path:
        sequenceName = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(frame.sequenceId)).strip("._")
        if not sequenceName:
            sequenceName = "sequence"
        return (
            self.config.outputRoot
            / sequenceName
            / f"frame_{int(frame.frameIndex):06d}"
            / stage
            / f"view_{viewId:04d}.png"
        )


def _indexViews(views: Sequence[LocalView]) -> dict[int, LocalView]:
    _requireUniqueViewIds(views)
    return {view.spec.viewId: view for view in views}


def _requireUniqueViewIds(views: Sequence[LocalView]) -> None:
    viewIds = [view.spec.viewId for view in views]
    if len(viewIds) != len(set(viewIds)):
        raise ProtocolError("visualization local viewIds must be unique")


def _requireUniqueObservationIds(
    observations: Sequence[LocalObservation] | Sequence[ProjectedObservation],
) -> None:
    viewIds = [observation.viewId for observation in observations]
    if len(viewIds) != len(set(viewIds)):
        raise ProtocolError("visualization observation viewIds must be unique")


def _requireViewIds(depthRgbByViewId: Mapping[int, NDArray[np.uint8]]) -> None:
    invalidViewId = any(
        isinstance(viewId, bool) or not isinstance(viewId, int) or viewId < 0
        for viewId in depthRgbByViewId
    )
    if invalidViewId:
        raise ProtocolError("visualization depth RGB viewIds must be non-negative integers")


def _requireView(viewById: Mapping[int, LocalView], viewId: int) -> LocalView:
    try:
        return viewById[viewId]
    except KeyError as error:
        message = f"visualization observation has no local view: viewId={viewId}"
        raise ProtocolError(message) from error


__all__ = ["VisualizationRecorder"]
