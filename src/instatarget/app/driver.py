"""Runtime driver orchestration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

from instatarget.controller import DepthAwareTrackController
from instatarget.core.config import AppConfig
from instatarget.core.errors import DecodeError, OutputError, ProtocolError
from instatarget.core.protocols import FrameSource as FrameSourceProtocol
from instatarget.core.protocols import ResultSink as ResultSinkProtocol
from instatarget.core.protocols import SphericalGeometry, TrackerBackend
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    DepthSummary,
    FramePacket,
    LocalObservation,
    LocalView,
    MotionState3D,
    ProjectedObservation,
    TrackResult,
)
from instatarget.geometry import SphericalGeometryImpl, makeSphericalPoint
from instatarget.io.result_sink import FileResultSink
from instatarget.tracker import DepthEncoder, DepthPreprocessor, FusionHead, HiTBackend, TrackerBackendImpl
from instatarget.tracker.hit_backend import HiTPrediction, HiTSession

if TYPE_CHECKING:
    from instatarget.core.protocols import DepthProcessor
    from instatarget.visualization.recorder import VisualizationRecorder


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    geometry: SphericalGeometry
    controller: DepthAwareTrackController
    backend: TrackerBackend
    sink: ResultSinkProtocol
    depthProcessor: DepthProcessor | None = None
    recorder: "VisualizationRecorder | None" = None


class FallbackHiTSession:
    """A deterministic local backend that keeps the CLI runnable without weights."""

    supportsOnlineTemplates = True

    def encodeTemplate(self, rgb: np.ndarray, bbox: BBoxXYWH) -> object:
        crop = _cropRgb(rgb, bbox)
        return {
            "bbox": (float(bbox.xPx), float(bbox.yPx), float(bbox.widthPx), float(bbox.heightPx)),
            "mean": crop.mean(axis=(0, 1)).astype(np.float32),
        }

    def infer(self, rgb: np.ndarray, templateFeatures: Iterable[object]) -> HiTPrediction:
        templates = [_coerceTemplate(feature) for feature in templateFeatures]
        if not templates:
            raise ProtocolError("fallback HiT session requires at least one template")
        templateBox = _meanBox(template["bbox"] for template in templates)
        templateMean = np.mean(np.stack([template["mean"] for template in templates]), axis=0)
        currentMean = rgb.mean(axis=(0, 1)).astype(np.float32)
        colorDelta = (currentMean - templateMean) / 255.0
        widthPx, heightPx = rgb.shape[1], rgb.shape[0]
        centerX = templateBox.xPx + templateBox.widthPx / 2.0 + float(colorDelta[0]) * 0.35 * widthPx
        centerY = templateBox.yPx + templateBox.heightPx / 2.0 - float(colorDelta[1]) * 0.35 * heightPx
        width = max(2.0, min(float(widthPx), templateBox.widthPx))
        height = max(2.0, min(float(heightPx), templateBox.heightPx))
        bbox = BBoxXYWH(
            xPx=_clamp(centerX - width / 2.0, 0.0, max(0.0, widthPx - width)),
            yPx=_clamp(centerY - height / 2.0, 0.0, max(0.0, heightPx - height)),
            widthPx=width,
            heightPx=height,
        )
        appearanceScore = float(np.clip(1.0 - np.mean(np.abs(colorDelta)), 0.0, 1.0))
        return HiTPrediction(bbox=bbox, modelScore=appearanceScore, appearanceScore=appearanceScore)

    def close(self) -> None:
        return None


def buildRuntime(config: AppConfig) -> RuntimeBundle:
    geometry = SphericalGeometryImpl(
        boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge,
    )
    depthProcessor: DepthPreprocessor | None = (
        DepthPreprocessor(config.depth) if config.depth.enabled else None
    )
    backend = TrackerBackendImpl(
        HiTBackend(FallbackHiTSession()),
        depthProcessor=depthProcessor,
        depthEncoder=DepthEncoder(),
        fusionHead=FusionHead(config.fusionHead, depthScoreWeight=config.backendFusion.depthScoreWeight),
        depthEnabled=config.depth.enabled,
    )
    controller = DepthAwareTrackController(geometry, config)
    sink = FileResultSink()
    return RuntimeBundle(geometry, controller, backend, sink, depthProcessor, None)


def runTracking(
    *,
    source: FrameSourceProtocol,
    initialBox: BBoxXYWH,
    geometry: SphericalGeometry,
    controller: DepthAwareTrackController,
    backend: TrackerBackend,
    sink: ResultSinkProtocol,
    depthProcessor: DepthProcessor | None = None,
    recorder: "VisualizationRecorder | None" = None,
) -> int:
    """Run the sequential tracking pipeline and publish one result per frame."""
    try:
        frame0 = _requireFrame(source.read())
        initPlan = controller.buildInitialization(frame0, initialBox)
        templateView = geometry.cropViews(frame0, [initPlan.templateView])[0]
        backend.initialize(templateView, initPlan.templateBox)
        initDepth = (
            depthProcessor.summarize(frame0, initialBox) if depthProcessor is not None else None
        )
        sink.write(controller.commitInitialization(initPlan, initDepth))
        resultCount = 1
        if recorder is not None:
            recorder.recordLocalRgb(frame0, [templateView])

        while True:
            frame = source.read()
            if frame is None:
                break
            plan = controller.plan(frame)
            views = tuple(geometry.cropViews(frame, plan.views))
            observations = tuple(backend.infer(views, plan.templateCommand))
            projected = tuple(
                _projectObservation(
                    frame=frame,
                    view=view,
                    observation=observation,
                    predictedMotion=plan.predictedMotion,
                    geometry=geometry,
                )
                for view, observation in zip(views, observations, strict=True)
            )
            if recorder is not None:
                recorder.recordLocalRgb(frame, views)
                recorder.recordBackendBoxes(frame, views, observations)
                recorder.recordGeometryBoxes(frame, projected)
            sink.write(controller.update(plan, projected))
            resultCount += 1
        return resultCount
    except Exception:
        if hasattr(sink, "close"):
            try:
                sink.close()  # type: ignore[call-arg]
            except Exception:
                pass
        raise


def finalizeSink(sink: ResultSinkProtocol, expectedFrameCount: int) -> None:
    sink.finalize(expectedFrameCount)


def openSink(sink: ResultSinkProtocol, destination: str) -> None:
    sink.open(destination)


def closeBackend(backend: TrackerBackend) -> None:
    backend.close()


def _projectObservation(
    *,
    frame: FramePacket,
    view: LocalView,
    observation: LocalObservation,
    predictedMotion: MotionState3D | None,
    geometry: SphericalGeometry,
) -> ProjectedObservation:
    candidateBfov = geometry.localBoxToBfov(observation.bbox, view.spec)
    bbox = geometry.bfovToBbox(candidateBfov, frame.rgb.shape[1], frame.rgb.shape[0])
    motionScore = _motionScore(candidateBfov.center, predictedMotion)
    scaleScore = _scaleScore(observation.bbox, view)
    return ProjectedObservation(
        viewId=view.spec.viewId,
        bfov=candidateBfov,
        bbox=bbox,
        modelScore=observation.modelScore,
        appearanceScore=observation.appearanceScore,
        motionScore=motionScore,
        scaleScore=scaleScore,
        depthScore=observation.depthScore,
        fusedScore=observation.fusedScore,
        depthSummary=observation.depthSummary,
        localBox=observation.bbox,
    )


def _motionScore(center, motion: MotionState3D | None) -> float:
    if motion is None:
        return 1.0
    motionPoint = makeSphericalPoint(math.atan2(motion.position[0], motion.position[2]), math.asin(max(-1.0, min(1.0, motion.position[1]))))
    dot = max(-1.0, min(1.0, center.x * motionPoint.x + center.y * motionPoint.y + center.z * motionPoint.z))
    return float(np.clip((dot + 1.0) / 2.0 * motion.confidence, 0.0, 1.0))


def _scaleScore(box: BBoxXYWH, view: LocalView) -> float:
    viewArea = max(float(view.spec.outputWidthPx * view.spec.outputHeightPx), 1.0)
    boxArea = max(float(box.widthPx * box.heightPx), 1e-6)
    return float(np.clip(1.0 - abs(math.log(boxArea / viewArea)) / 4.0, 0.0, 1.0))


def _requireFrame(frame: FramePacket | None) -> FramePacket:
    if frame is None:
        raise DecodeError("input source is empty")
    return frame


def _cropRgb(rgb: np.ndarray, box: BBoxXYWH) -> np.ndarray:
    x0 = max(0, int(math.floor(box.xPx)))
    y0 = max(0, int(math.floor(box.yPx)))
    x1 = min(rgb.shape[1], int(math.ceil(box.xPx + box.widthPx)))
    y1 = min(rgb.shape[0], int(math.ceil(box.yPx + box.heightPx)))
    if x1 <= x0 or y1 <= y0:
        return rgb[:1, :1].copy()
    return rgb[y0:y1, x0:x1].copy()


def _coerceTemplate(feature: object) -> dict[str, np.ndarray | tuple[float, float, float, float]]:
    if isinstance(feature, dict) and "bbox" in feature and "mean" in feature:
        bbox = tuple(float(value) for value in feature["bbox"])
        mean = np.asarray(feature["mean"], dtype=np.float32).reshape(3)
        return {"bbox": bbox, "mean": mean}
    if isinstance(feature, tuple) and len(feature) == 2:
        bbox, mean = feature
        return {
            "bbox": tuple(float(value) for value in bbox),
            "mean": np.asarray(mean, dtype=np.float32).reshape(3),
        }
    raise ProtocolError("fallback HiT feature has an unsupported shape")


def _meanBox(boxes: Iterable[tuple[float, float, float, float]]) -> BBoxXYWH:
    items = list(boxes)
    if not items:
        return BBoxXYWH(0.0, 0.0, 2.0, 2.0)
    array = np.asarray(items, dtype=np.float64)
    return BBoxXYWH(
        xPx=float(np.mean(array[:, 0])),
        yPx=float(np.mean(array[:, 1])),
        widthPx=float(np.mean(array[:, 2])),
        heightPx=float(np.mean(array[:, 3])),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


__all__ = [
    "FallbackHiTSession",
    "RuntimeBundle",
    "buildRuntime",
    "closeBackend",
    "finalizeSink",
    "openSink",
    "runTracking",
]
