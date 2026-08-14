"""Single-HiT tracker backend for RGB-only and depth-edge-enhanced RGB."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter_ns

from instatarget.core.errors import ProtocolError
from instatarget.core.protocols import TrackerBackend as TrackerBackendProtocol
from instatarget.core.types import (
    BBoxXYWH,
    DepthPlane,
    LocalObservation,
    LocalView,
    TemplateCommand,
)
from instatarget.tracker.depth_preprocessor import DepthPreprocessor
from instatarget.tracker.hit_backend import HiTBackend
from instatarget.tracker.observation import buildRgbObservation
from instatarget.tracker.template import TemplateCache


class TrackerBackendImpl(TrackerBackendProtocol):
    """Own the HiT session, template slots, and local RGB observations."""

    def __init__(
        self,
        hitBackend: HiTBackend,
        depthProcessor: DepthPreprocessor | None = None,
        depthEnabled: bool | None = None,
    ) -> None:
        self._hitBackend = hitBackend
        self._templates = TemplateCache()
        self._previousViews: dict[int, LocalView] = {}
        self._depthProcessor = depthProcessor
        self._depthEnabled = (
            bool(depthEnabled)
            if depthEnabled is not None
            else self._depthProcessor is not None
        )
        self._initialized = False
        self._closed = False
        self._lastPreparedViews: tuple[LocalView, ...] = ()

    @property
    def templateRevision(self) -> int:
        return self._templates.revision

    @property
    def lastPreparedViews(self) -> tuple[LocalView, ...]:
        return self._lastPreparedViews

    def initialize(self, template: LocalView, templateBox: BBoxXYWH) -> None:
        if self._closed:
            raise ProtocolError("tracker backend is closed")
        if self._initialized:
            raise ProtocolError("tracker backend is already initialized")
        prepared = self.prepareViews((template,))[0]
        self._lastPreparedViews = (prepared,)
        self._templates.initialize(self._hitBackend, prepared, templateBox)
        self._initialized = True
        self._previousViews = {prepared.spec.viewId: _copyView(prepared)}

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
        observations: list[LocalObservation] = []
        preparedViews = self.prepareViews(views)
        self._lastPreparedViews = preparedViews
        for sourceView, view in zip(views, preparedViews, strict=True):
            startedNs = perf_counter_ns()
            prediction = self._hitBackend.infer(view.rgb, snapshot.features)
            depthSummary = (
                self._depthProcessor.summarizeLocal(sourceView, prediction.bbox)
                if self._depthEnabled
                and self._depthProcessor is not None
                and sourceView.depth is not None
                else None
            )
            rgbObservation = buildRgbObservation(view, prediction, 0)
            observations.append(
                LocalObservation(
                    viewId=view.spec.viewId,
                    bbox=rgbObservation.bbox,
                    modelScore=prediction.modelScore,
                    appearanceScore=prediction.appearanceScore,
                    depthScore=0.0,
                    fusedScore=prediction.appearanceScore,
                    depthSummary=depthSummary,
                    latencyNs=perf_counter_ns() - startedNs,
                )
            )
        self._previousViews = {view.spec.viewId: _copyView(view) for view in preparedViews}
        return observations

    def prepareViews(self, views: Sequence[LocalView]) -> tuple[LocalView, ...]:
        """Return the exact images that will be sent to HiT and visualized."""
        prepared: list[LocalView] = []
        for view in views:
            if not self._depthEnabled or self._depthProcessor is None or view.depth is None:
                prepared.append(view)
                continue
            rgb = self._depthProcessor.enhanceRgb(view.rgb, view.depth)
            rgb.setflags(write=False)
            prepared.append(LocalView(spec=view.spec, rgb=rgb, depth=view.depth))
        return tuple(prepared)

    def close(self) -> None:
        if self._closed:
            return
        self._hitBackend.close()
        self._templates.clear()
        self._previousViews.clear()
        self._lastPreparedViews = ()
        self._closed = True



def _validateViewSequence(views: Sequence[LocalView]) -> None:
    viewIds = [view.spec.viewId for view in views]
    if len(viewIds) != len(set(viewIds)):
        raise ProtocolError("tracker infer views must have unique viewIds")


def _copyView(view: LocalView) -> LocalView:
    rgb = view.rgb.copy()
    rgb.setflags(write=False)
    depth = view.depth
    if depth is not None:
        values = depth.values.copy()
        mask = depth.validMask.copy()
        values.setflags(write=False)
        mask.setflags(write=False)
        depth = DepthPlane(values=values, validMask=mask, unit=depth.unit)
    return LocalView(spec=view.spec, rgb=rgb, depth=depth)


TrackerBackend = TrackerBackendImpl

__all__ = ["TrackerBackend", "TrackerBackendImpl"]
