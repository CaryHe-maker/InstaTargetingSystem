"""Lightweight runtime orchestration for the smoke-test pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from instatarget.controller.simple_geometry_controller import SimpleGeometryTrackController
from instatarget.core.config import VisualizationConfig
from instatarget.core.errors import ProtocolError
from instatarget.core.types import BBoxXYWH, TrackResult
from instatarget.data.image_sequence_source import DirectoryFrameSource
from instatarget.geometry import SphericalGeometryImpl
from instatarget.tracker import HiTBackend, HiTPrediction, TrackerBackendImpl
from instatarget.visualization import VisualizationRecorder


@dataclass
class SmokeHiTSession:
    """Deterministic tracker session used for controller smoke tests."""

    supportsOnlineTemplates: bool = True
    _templateSize: tuple[float, float] | None = None

    def encodeTemplate(self, rgb: np.ndarray, bbox: BBoxXYWH) -> object:
        self._templateSize = (bbox.widthPx, bbox.heightPx)
        return {
            "templateSize": self._templateSize,
            "mean": float(rgb.mean()),
        }

    def infer(self, rgb: np.ndarray, templateFeatures: tuple[object, ...]) -> HiTPrediction:
        if not templateFeatures:
            raise ProtocolError("smoke session expects at least one template feature")
        feature = templateFeatures[0]
        if isinstance(feature, dict):
            templateSize = feature.get("templateSize", self._templateSize)
        else:
            templateSize = self._templateSize
        if templateSize is None:
            templateSize = (max(8.0, rgb.shape[1] * 0.4), max(8.0, rgb.shape[0] * 0.4))
        widthPx = min(float(rgb.shape[1]), float(templateSize[0]))
        heightPx = min(float(rgb.shape[0]), float(templateSize[1]))
        bbox = BBoxXYWH(
            xPx=max(0.0, (rgb.shape[1] - widthPx) / 2.0),
            yPx=max(0.0, (rgb.shape[0] - heightPx) / 2.0),
            widthPx=max(1.0, widthPx),
            heightPx=max(1.0, heightPx),
        )
        brightness = float(np.clip(rgb.mean() / 255.0, 0.0, 1.0))
        appearanceScore = 0.55 + 0.4 * brightness
        return HiTPrediction(
            bbox=bbox,
            modelScore=0.75,
            appearanceScore=min(1.0, appearanceScore),
        )

    def close(self) -> None:
        return None


def runSmokePipeline(
    inputPath: str | Path,
    outputRoot: str | Path,
    initialBox: BBoxXYWH,
    *,
    sequenceId: str | None = None,
    viewWidthPx: int = 256,
    viewHeightPx: int = 256,
    recursive: bool = False,
) -> list[TrackResult]:
    source = DirectoryFrameSource(recursive=recursive, sequenceId=sequenceId)
    source.open(str(inputPath))
    geometry = SphericalGeometryImpl()
    backend = TrackerBackendImpl(HiTBackend(SmokeHiTSession()))
    visualization = VisualizationRecorder(
        VisualizationConfig(
            enabled=True,
            outputRoot=Path(outputRoot),
            stages=frozenset({"local_rgb", "backend_box", "geometry_box"}),
        )
    )
    controller = SimpleGeometryTrackController(
        geometry=geometry,
        tracker=backend,
        visualization=visualization,
        viewWidthPx=viewWidthPx,
        viewHeightPx=viewHeightPx,
    )
    try:
        results: list[TrackResult] = []
        frame = source.read()
        if frame is None:
            raise ProtocolError("input sequence is empty")
        results.append(controller.initialize(frame, initialBox))
        while True:
            frame = source.read()
            if frame is None:
                break
            results.append(controller.step(frame))
        return results
    finally:
        controller.close()
        source.close()


__all__ = ["SmokeHiTSession", "runSmokePipeline"]
