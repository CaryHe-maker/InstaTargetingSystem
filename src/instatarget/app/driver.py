"""Runtime driver orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from instatarget.controller import (
    UNCALIBRATED_STAGE3_SCORE_CALIBRATION,
    ScoreCalibration,
    SpeculativePipeline,
    TrackControllerImpl,
    calibrateBackendFusedScore,
    calibrateLocalAppearanceProbabilities,
    composeSingleScore,
    loadScoreCalibration,
    scoreViewCenterMotion,
)
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
from instatarget.geometry import SphericalGeometryImpl
from instatarget.io.result_sink import FileResultSink
from instatarget.tracker import HiTBackend, PyTorchHiTSession, TrackerBackendImpl

if TYPE_CHECKING:
    from instatarget.tracker.hit_backend import HiTSession
    from instatarget.visualization.recorder import VisualizationRecorder
    from instatarget.visualization.result import ResultVisualizationRecorder
    from instatarget.visualization.time_counter import TimeCounter


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    geometry: SphericalGeometry
    controller: TrackControllerImpl
    backend: TrackerBackend
    sink: ResultSinkProtocol
    scoreCalibration: ScoreCalibration
    recorder: VisualizationRecorder | None = None
    speculativePipeline: SpeculativePipeline | None = None


def buildRuntime(
    config: AppConfig,
    *,
    hitSessionFactory: Callable[[ModelConfig], HiTSession] | None = None,
    allowUncalibratedScoring: bool = False,
) -> RuntimeBundle:
    sessionFactory = hitSessionFactory or PyTorchHiTSession
    geometry = SphericalGeometryImpl(
        boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge,
    )
    rgbSession = sessionFactory(config.model)
    backend = TrackerBackendImpl(HiTBackend(rgbSession))
    controller = TrackControllerImpl(geometry, config)
    sink = FileResultSink()
    recorder = None
    if config.visualization.enabled:
        from instatarget.visualization.recorder import VisualizationRecorder

        recorder = VisualizationRecorder(config.visualization)
    speculativePipeline = SpeculativePipeline(config.speculativePipeline)
    if config.scoring.calibrationArtifact is not None:
        scoreCalibration = loadScoreCalibration(
            config.scoring.calibrationArtifact,
            checkpointPath=config.model.weights,
            candidateMinScore=config.tracking.candidateMinScore,
            fusionSourceMinConfidence=config.evaluator.fusionSourceMinConfidence,
            requireCheckpointHashMatch=config.scoring.requireCheckpointHashMatch,
        )
    elif allowUncalibratedScoring:
        scoreCalibration = UNCALIBRATED_STAGE3_SCORE_CALIBRATION
    else:
        raise ValueError("production runtime requires a Stage 3 calibration artifact")
    return RuntimeBundle(
        geometry=geometry,
        controller=controller,
        backend=backend,
        sink=sink,
        scoreCalibration=scoreCalibration,
        recorder=recorder,
        speculativePipeline=speculativePipeline,
    )


def runTracking(
    *,
    source: FrameSourceProtocol,
    initialBox: BBoxXYWH,
    geometry: SphericalGeometry,
    controller: TrackControllerImpl,
    backend: TrackerBackend,
    sink: ResultSinkProtocol,
    recorder: VisualizationRecorder | None = None,
    resultRecorder: ResultVisualizationRecorder | None = None,
    processingTimer: TimeCounter | None = None,
    scoreCalibration: ScoreCalibration,
) -> int:
    """Run the sequential tracking pipeline and publish one result per frame."""
    try:
        _startProcessing(processingTimer)
        try:
            frame0 = _requireFrame(source.read())
            initPlan = controller.buildInitialization(frame0, initialBox)
            templateView = geometry.cropViews(frame0, [initPlan.templateView])[0]
            backend.initialize(templateView, initPlan.templateBox)
            initialResult = controller.commitInitialization(initPlan)
        finally:
            _stopProcessing(processingTimer)
        sink.write(initialResult)
        if resultRecorder is not None:
            resultRecorder.record(frame0, initialResult, stateScore=None, roundCount=0)
        resultCount = 1
        if recorder is not None:
            recorder.recordLocalRgb(frame0, [templateView])

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
                        observations = calibrateLocalAppearanceProbabilities(
                            rawObservations,
                            scoreCalibration,
                        )
                        projected = tuple(
                            _projectObservation(
                                frame=frame,
                                view=view,
                                observation=observation,
                                predictedMotion=plan.predictedMotion,
                                geometry=geometry,
                                scoreCalibration=scoreCalibration,
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
                roundCount = (
                    stateObservation.attemptIndex + 1
                    if stateObservation is not None
                    and stateObservation.frameIndex == frame.frameIndex
                    else None
                )
                resultRecorder.record(
                    frame,
                    result,
                    stateScore=stateScore,
                    roundCount=roundCount,
                )
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
    scoreCalibration: ScoreCalibration,
) -> ProjectedObservation:
    projection = geometry.projectLocalBoxBoundary(
        observation.bbox,
        view.spec,
        frame.rgb.shape[1],
        frame.rgb.shape[0],
    )
    motion = scoreViewCenterMotion(view.spec.bfov.center, predictedMotion)
    appearanceProbability = (
        observation.appearanceProbability
        if observation.appearanceProbability is not None
        else calibrateBackendFusedScore(observation.fusedScore, scoreCalibration)
    )
    singleScore = composeSingleScore(
        appearanceProbability,
        motion.effectiveProbability,
        scoreCalibration,
    )
    scaleScore = _scaleScore(observation.bbox, view)
    normalizedRadius, edgeMargin = _projectionQuality(observation.bbox, view)
    return ProjectedObservation(
        viewId=view.spec.viewId,
        bfov=projection.bfov,
        bbox=projection.bbox,
        modelScore=observation.modelScore,
        appearanceScore=observation.appearanceScore,
        motionScore=motion.effectiveProbability,
        scaleScore=scaleScore,
        fusedScore=singleScore,
        localBox=observation.bbox,
        backendFusedScore=observation.fusedScore,
        appearanceProbability=appearanceProbability,
        rawMotionScore=motion.rawScore,
        motionProbability=motion.probability,
        motionReliability=motion.reliability,
        singleScore=singleScore,
        erpBoundary=projection.erpBoundary,
        envelopeInflation=projection.envelopeInflation,
        normalizedRadius=normalizedRadius,
        edgeMargin=edgeMargin,
    )


def _scaleScore(box: BBoxXYWH, view: LocalView) -> float:
    viewArea = max(float(view.spec.outputWidthPx * view.spec.outputHeightPx), 1.0)
    boxArea = max(float(box.widthPx * box.heightPx), 1e-6)
    return float(np.clip(1.0 - abs(math.log(boxArea / viewArea)) / 4.0, 0.0, 1.0))


def _projectionQuality(box: BBoxXYWH, view: LocalView) -> tuple[float, float]:
    width = float(view.spec.outputWidthPx)
    height = float(view.spec.outputHeightPx)
    centerX = box.xPx + box.widthPx / 2.0
    centerY = box.yPx + box.heightPx / 2.0
    halfDiagonal = max(math.hypot(width / 2.0, height / 2.0), 1e-9)
    normalizedRadius = math.hypot(centerX - width / 2.0, centerY - height / 2.0) / halfDiagonal
    edgeMargin = min(
        box.xPx,
        box.yPx,
        width - (box.xPx + box.widthPx),
        height - (box.yPx + box.heightPx),
    ) / max(width, height, 1.0)
    return max(0.0, normalizedRadius), max(0.0, edgeMargin)


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
