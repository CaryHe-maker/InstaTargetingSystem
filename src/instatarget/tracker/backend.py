"""RGB-only tracker backend facade."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter_ns

from instatarget.core.errors import ProtocolError
from instatarget.core.protocols import TrackerBackend as TrackerBackendProtocol
from instatarget.core.types import BBoxXYWH, LocalObservation, LocalView, TemplateCommand
from instatarget.tracker.hit_backend import HiTBackend
from instatarget.tracker.observation import buildRgbObservation
from instatarget.tracker.template import TemplateCache


class TrackerBackendImpl(TrackerBackendProtocol):
    """Own one RGB HiT session, its template cache, and local observations."""

    def __init__(self, hitBackend: HiTBackend) -> None:
        self._hitBackend = hitBackend
        self._templates = TemplateCache()
        self._previousViews: dict[int, LocalView] = {}
        self._previousViewsFrameIndex: int | None = None
        self._initialized = False
        self._closed = False

    @property
    def templateRevision(self) -> int:
        return self._templates.revision

    def initialize(self, template: LocalView, templateBox: BBoxXYWH) -> None:
        if self._closed:
            raise ProtocolError("tracker backend is closed")
        if self._initialized:
            raise ProtocolError("tracker backend is already initialized")
        self._templates.initialize(self._hitBackend, template, templateBox)
        self._initialized = True
        self._previousViews = {template.spec.viewId: _copyView(template)}
        self._previousViewsFrameIndex = 0

    def infer(
        self,
        views: Sequence[LocalView],
        command: TemplateCommand,
    ) -> Sequence[LocalObservation]:
        if self._closed:
            raise ProtocolError("tracker backend is closed")
        if not self._initialized:
            raise ProtocolError("tracker backend has not been initialized")
        _validateViewSequence(views)
        self._templates.apply(self._hitBackend, command, self._previousViews)
        snapshot = self._templates.snapshot()
        inferenceStartedNs = perf_counter_ns()
        predictions = self._hitBackend.inferBatch(
            tuple(view.rgb for view in views),
            (snapshot.anchor.features,),
        )
        sharedInferenceNs = (
            (perf_counter_ns() - inferenceStartedNs) // len(views) if views else 0
        )
        observations = tuple(
            buildRgbObservation(view, prediction, sharedInferenceNs)
            for view, prediction in zip(views, predictions, strict=True)
        )
        currentViews = {view.spec.viewId: _copyView(view) for view in views}
        currentFrameIndex = int(command.frameIndex)
        if self._previousViewsFrameIndex == currentFrameIndex:
            self._previousViews.update(currentViews)
        else:
            self._previousViews = currentViews
        self._previousViewsFrameIndex = currentFrameIndex
        return observations

    def close(self) -> None:
        if self._closed:
            return
        self._hitBackend.close()
        self._templates.clear()
        self._previousViews.clear()
        self._previousViewsFrameIndex = None
        self._closed = True


def _validateViewSequence(views: Sequence[LocalView]) -> None:
    viewIds = [view.spec.viewId for view in views]
    if len(viewIds) != len(set(viewIds)):
        raise ProtocolError("tracker infer views must have unique viewIds")


def _copyView(view: LocalView) -> LocalView:
    rgb = view.rgb.copy()
    rgb.setflags(write=False)
    return LocalView(spec=view.spec, rgb=rgb)


TrackerBackend = TrackerBackendImpl

__all__ = ["TrackerBackend", "TrackerBackendImpl"]
