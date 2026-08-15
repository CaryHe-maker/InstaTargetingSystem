"""Runtime driver orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from instatarget.controller import DepthAwareTrackController, remapLocalObservationFusedScores
from instatarget.core.config import AppConfig, ModelConfig
from instatarget.core.errors import DecodeError
from instatarget.core.protocols import FrameSource as FrameSourceProtocol
from instatarget.core.protocols import MoreViewsRequired, SphericalGeometry, TrackerBackend
from instatarget.core.protocols import ResultSink as ResultSinkProtocol
from instatarget.core.types import (
    BBoxXYWH,
    FramePacket,
    LocalObservation,
    LocalView,
    MotionState3D,
    ProjectedObservation,
)
from instatarget.geometry import SphericalGeometryImpl, makeSphericalPoint
from instatarget.io.result_sink import FileResultSink
from instatarget.tracker import (
    DepthEncoder,
    DepthPreprocessor,
    FusionHead,
    HiTBackend,
    PyTorchHiTSession,
    TrackerBackendImpl,
)

if TYPE_CHECKING:
    from instatarget.core.protocols import DepthProcessor
    from instatarget.tracker.hit_backend import HiTSession
    from instatarget.visualization.recorder import VisualizationRecorder
    from instatarget.visualization.result import ResultVisualizationRecorder
    from instatarget.visualization.time_counter import TimeCounter


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    geometry: SphericalGeometry
    controller: DepthAwareTrackController
    backend: TrackerBackend
    sink: ResultSinkProtocol
    depthProcessor: DepthProcessor | None = None
    recorder: VisualizationRecorder | None = None


def buildRuntime(
    config: AppConfig,
    *,
    hitSessionFactory: Callable[[ModelConfig], HiTSession] | None = None,
) -> RuntimeBundle:
    sessionFactory = hitSessionFactory or PyTorchHiTSession
    geometry = SphericalGeometryImpl(
        boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge,
    )
    depthProcessor: DepthPreprocessor | None = (
        DepthPreprocessor(config.depth) if config.depth.enabled else None
    )
    rgbSession = sessionFactory(config.model)
    try:
        depthEncoder = (
            DepthEncoder(session=sessionFactory(config.model)) if config.depth.enabled else None
        )
    except Exception:
        rgbSession.close()
        raise
    backend = TrackerBackendImpl(
        HiTBackend(rgbSession),
        depthProcessor=depthProcessor,
        depthEncoder=depthEncoder,
        fusionHead=FusionHead(
            config.fusionHead, depthScoreWeight=config.backendFusion.depthScoreWeight
        ),
        depthEnabled=config.depth.enabled,
    )
    controller = DepthAwareTrackController(geometry, config)
    sink = FileResultSink()
    recorder = None
    if config.visualization.enabled:
        from instatarget.visualization.recorder import VisualizationRecorder

        recorder = VisualizationRecorder(config.visualization)
    return RuntimeBundle(geometry, controller, backend, sink, depthProcessor, recorder)


def runTracking(
    *,
    source: FrameSourceProtocol,
    initialBox: BBoxXYWH,
    geometry: SphericalGeometry,
    controller: DepthAwareTrackController,
    backend: TrackerBackend,
    sink: ResultSinkProtocol,
    depthProcessor: DepthProcessor | None = None,
    recorder: VisualizationRecorder | None = None,
    resultRecorder: ResultVisualizationRecorder | None = None,
    processingTimer: TimeCounter | None = None,
) -> int:
    """Run the sequential tracking pipeline and publish one result per frame."""
    try:
        _startProcessing(processingTimer)
        try:
            frame0 = _requireFrame(source.read())
            initPlan = controller.buildInitialization(frame0, initialBox)
            templateView = geometry.cropViews(frame0, [initPlan.templateView])[0]
            backend.initialize(templateView, initPlan.templateBox)
            initDepth = (
                depthProcessor.summarize(frame0, initialBox) if depthProcessor is not None else None
            )
            initialResult = controller.commitInitialization(initPlan, initDepth)
        finally:
            _stopProcessing(processingTimer)
        sink.write(initialResult)
        if resultRecorder is not None:
            resultRecorder.record(frame0, initialResult, stateScore=None)
        resultCount = 1
        if recorder is not None:
            recorder.recordLocalRgb(frame0, [templateView])
            if depthProcessor is not None and frame0.depth is not None:
                recorder.recordDepthRgb(
                    frame0, {0: depthProcessor.preprocess(frame0.depth).depthRgb}
                )

        while True:
            _startProcessing(processingTimer)
            try:
                frame = source.read()
                if frame is not None:
                    plan = controller.beginFrame(frame)
                    visualizationBatches: list[
                        tuple[
                            tuple[LocalView, ...],
                            tuple[LocalObservation, ...],
                            tuple[ProjectedObservation, ...],
                        ]
                    ] = []
                    while True:
                        views = tuple(geometry.cropViews(frame, plan.views))
                        rawObservations = tuple(backend.infer(views, plan.templateCommand))
                        observations = remapLocalObservationFusedScores(rawObservations)
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
                        visualizationBatches.append((views, observations, projected))
                        step = controller.consume(plan, projected)
                        if isinstance(step, MoreViewsRequired):
                            plan = step.plan
                            continue
                        result = step.result
                        break
            finally:
                _stopProcessing(processingTimer)
            if frame is None:
                break
            if recorder is not None:
                if depthProcessor is not None and frame.depth is not None:
                    depthRgb = depthProcessor.preprocess(frame.depth).depthRgb
                    recorder.recordDepthRgb(frame, {0: depthRgb})
                for views, observations, projected in visualizationBatches:
                    recorder.recordLocalRgb(frame, views)
                    recorder.recordBackendBoxes(frame, views, observations)
                    recorder.recordGeometryBoxes(frame, projected)
            sink.write(result)
            if resultRecorder is not None:
                stateObservation = controller.lastStateObservation
                stateScore = (
                    stateObservation.stateScore
                    if stateObservation is not None
                    and stateObservation.frameIndex == frame.frameIndex
                    else None
                )
                resultRecorder.record(frame, result, stateScore=stateScore)
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
    motionPoint = makeSphericalPoint(
        math.atan2(motion.position[0], motion.position[2]),
        math.asin(max(-1.0, min(1.0, motion.position[1]))),
    )
    dot = max(
        -1.0,
        min(1.0, center.x * motionPoint.x + center.y * motionPoint.y + center.z * motionPoint.z),
    )
    return float(np.clip((dot + 1.0) / 2.0 * motion.confidence, 0.0, 1.0))


def _scaleScore(box: BBoxXYWH, view: LocalView) -> float:
    viewArea = max(float(view.spec.outputWidthPx * view.spec.outputHeightPx), 1.0)
    boxArea = max(float(box.widthPx * box.heightPx), 1e-6)
    return float(np.clip(1.0 - abs(math.log(boxArea / viewArea)) / 4.0, 0.0, 1.0))


def _requireFrame(frame: FramePacket | None) -> FramePacket:
    if frame is None:
        raise DecodeError("input source is empty")
    return frame


def _startProcessing(timer: TimeCounter | None) -> None:
    if timer is not None:
        timer.startProcessing()


def _stopProcessing(timer: TimeCounter | None) -> None:
    if timer is not None:
        timer.stopProcessing()


__all__ = [
    "RuntimeBundle",
    "buildRuntime",
    "closeBackend",
    "finalizeSink",
    "openSink",
    "runTracking",
]
