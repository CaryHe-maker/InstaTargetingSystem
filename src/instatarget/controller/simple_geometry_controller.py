"""A small ERP -> geometry -> tracker smoke-test controller."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from instatarget.core.errors import ProtocolError
from instatarget.core.protocols import SphericalGeometry, TrackerBackend
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    FramePacket,
    LocalObservation,
    LocalView,
    ProjectedObservation,
    TemplateCommand,
    TemplateCommandKind,
    TrackResult,
    TrackStatus,
    ViewSpec,
)
from instatarget.visualization import VisualizationRecorder


@dataclass(slots=True)
class SimpleGeometryTrackController:
    """Track one target by cropping each frame around the latest projected BFoV."""

    geometry: SphericalGeometry
    tracker: TrackerBackend
    visualization: VisualizationRecorder | None = None
    viewWidthPx: int = 256
    viewHeightPx: int = 256
    viewId: int = 0
    templateBoxFraction: float = 0.6
    _currentBfov: BFoV | None = field(init=False, default=None, repr=False)
    _currentResult: TrackResult | None = field(init=False, default=None, repr=False)
    _initialized: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.viewWidthPx <= 0 or self.viewHeightPx <= 0:
            raise ProtocolError("view dimensions must be positive")
        if not 0.0 < self.templateBoxFraction <= 1.0:
            raise ProtocolError("templateBoxFraction must be in (0, 1]")
        if self.viewId < 0:
            raise ProtocolError("viewId must be non-negative")

    def initialize(self, frame: FramePacket, initialBox: BBoxXYWH) -> TrackResult:
        if self._initialized:
            raise ProtocolError("controller is already initialized")
        initialBfov = self.geometry.bboxToBfov(initialBox, frame.rgb.shape[1], frame.rgb.shape[0])
        view = self._crop(frame, initialBfov)
        templateBox = _centerBox(
            view.spec.outputWidthPx,
            view.spec.outputHeightPx,
            self.templateBoxFraction,
        )
        self.tracker.initialize(view, templateBox)
        resultBox = self.geometry.bfovToBbox(initialBfov, frame.rgb.shape[1], frame.rgb.shape[0])
        result = TrackResult(
            sequenceId=frame.sequenceId,
            frameIndex=frame.frameIndex,
            bbox=resultBox,
            bfov=initialBfov,
            confidence=1.0,
            status=TrackStatus.TRACKING,
            valid=True,
            depthSummary=None,
        )
        self._currentBfov = initialBfov
        self._currentResult = result
        self._initialized = True
        self._recordInitialization(frame, view, templateBox, resultBox, initialBfov)
        return result

    def step(self, frame: FramePacket) -> TrackResult:
        if not self._initialized or self._currentBfov is None:
            raise ProtocolError("controller is not initialized")

        view = self._crop(frame, self._currentBfov)
        command = TemplateCommand(
            kind=TemplateCommandKind.KEEP,
            frameIndex=frame.frameIndex,
            viewId=None,
            localBox=None,
            expectedRevision=_templateRevision(self.tracker) + 1,
        )
        observations = self.tracker.infer([view], command)
        if not observations:
            raise ProtocolError("tracker returned no observations")
        observation = observations[0]
        projectedBfov = self.geometry.localBoxToBfov(observation.bbox, view.spec)
        projectedBox = self.geometry.bfovToBbox(
            projectedBfov,
            frame.rgb.shape[1],
            frame.rgb.shape[0],
        )
        confidence = float(observation.fusedScore)
        status = TrackStatus.TRACKING if confidence >= 0.5 else TrackStatus.UNCERTAIN
        result = TrackResult(
            sequenceId=frame.sequenceId,
            frameIndex=frame.frameIndex,
            bbox=projectedBox,
            bfov=projectedBfov,
            confidence=confidence,
            status=status,
            valid=confidence > 0.0,
            depthSummary=observation.depthSummary,
        )
        self._currentBfov = projectedBfov
        self._currentResult = result
        self._recordStep(frame, view, observation, projectedBfov, projectedBox)
        return result

    def run(self, frames: Iterable[FramePacket], initialBox: BBoxXYWH) -> list[TrackResult]:
        iterator = iter(frames)
        firstFrame = next(iterator, None)
        if firstFrame is None:
            raise ProtocolError("frame sequence is empty")
        results = [self.initialize(firstFrame, initialBox)]
        for frame in iterator:
            results.append(self.step(frame))
        return results

    def close(self) -> None:
        self.tracker.close()
        self._currentBfov = None
        self._currentResult = None
        self._initialized = False

    def _crop(self, frame: FramePacket, bfov: BFoV) -> LocalView:
        spec = ViewSpec(
            viewId=self.viewId,
            bfov=bfov,
            outputWidthPx=self.viewWidthPx,
            outputHeightPx=self.viewHeightPx,
        )
        return self.geometry.cropViews(frame, [spec])[0]

    def _recordInitialization(
        self,
        frame: FramePacket,
        view: LocalView,
        templateBox: BBoxXYWH,
        resultBox: BBoxXYWH,
        bfov: BFoV,
    ) -> None:
        if self.visualization is None:
            return
        self.visualization.recordLocalRgb(frame, [view])
        self.visualization.recordBackendBoxes(frame, [view], [_localObservation(view, templateBox)])
        self.visualization.recordGeometryBoxes(
            frame,
            [_projectedObservation(self.viewId, bfov, resultBox)],
        )

    def _recordStep(
        self,
        frame: FramePacket,
        view: LocalView,
        observation: LocalObservation,
        bfov: BFoV,
        resultBox: BBoxXYWH,
    ) -> None:
        if self.visualization is None:
            return
        self.visualization.recordLocalRgb(frame, [view])
        self.visualization.recordBackendBoxes(frame, [view], [observation])
        self.visualization.recordGeometryBoxes(
            frame,
            [_projectedObservation(self.viewId, bfov, resultBox, observation)],
        )


def _centerBox(widthPx: int, heightPx: int, fraction: float) -> BBoxXYWH:
    boxWidthPx = max(1.0, min(float(widthPx), round(widthPx * fraction)))
    boxHeightPx = max(1.0, min(float(heightPx), round(heightPx * fraction)))
    xPx = (widthPx - boxWidthPx) / 2.0
    yPx = (heightPx - boxHeightPx) / 2.0
    return BBoxXYWH(xPx=xPx, yPx=yPx, widthPx=boxWidthPx, heightPx=boxHeightPx)


def _templateRevision(tracker: TrackerBackend) -> int:
    revision = getattr(tracker, "templateRevision", 0)
    return int(revision)


def _localObservation(view: LocalView, box: BBoxXYWH) -> LocalObservation:
    return LocalObservation(
        viewId=view.spec.viewId,
        bbox=box,
        modelScore=1.0,
        appearanceScore=1.0,
        depthScore=0.0,
        fusedScore=1.0,
        depthSummary=None,
        latencyNs=0,
    )


def _projectedObservation(
    viewId: int,
    bfov: BFoV,
    box: BBoxXYWH,
    observation: LocalObservation | None = None,
) -> ProjectedObservation:
    fusedScore = observation.fusedScore if observation is not None else 1.0
    modelScore = observation.modelScore if observation is not None else 1.0
    appearanceScore = observation.appearanceScore if observation is not None else 1.0
    depthScore = observation.depthScore if observation is not None else 0.0
    return ProjectedObservation(
        viewId=viewId,
        bfov=bfov,
        bbox=box,
        modelScore=modelScore,
        appearanceScore=appearanceScore,
        motionScore=fusedScore,
        scaleScore=fusedScore,
        depthScore=depthScore,
        fusedScore=fusedScore,
        depthSummary=observation.depthSummary if observation is not None else None,
)


__all__ = ["SimpleGeometryTrackController"]
