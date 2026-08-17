"""TrackerBackend facade supporting RGB-only and optional RGB-D inference."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter_ns

import numpy as np

from instatarget.core.errors import ModelError, ProtocolError
from instatarget.core.protocols import TrackerBackend as TrackerBackendProtocol
from instatarget.core.types import (
    BBoxXYWH,
    DepthPlane,
    LocalObservation,
    LocalView,
    TemplateCommand,
)
from instatarget.tracker.depth_encoder import DepthEncoder, DepthPrediction
from instatarget.tracker.depth_preprocessor import DepthPreprocessor
from instatarget.tracker.fusion_head import FusionHead
from instatarget.tracker.hit_backend import HiTBackend
from instatarget.tracker.observation import buildRgbObservation
from instatarget.tracker.template import TemplateCache


class TrackerBackendImpl(TrackerBackendProtocol):
    """Own the HiT session, template slots, and local RGB observations."""

    def __init__(
        self,
        hitBackend: HiTBackend,
        depthProcessor: DepthPreprocessor | None = None,
        depthEncoder: DepthEncoder | None = None,
        fusionHead: FusionHead | None = None,
        *,
        depthBackend: DepthEncoder | None = None,
        depthEnabled: bool | None = None,
        depthScoreWeight: float = 0.15,
    ) -> None:
        self._hitBackend = hitBackend
        self._templates = TemplateCache()
        self._previousViews: dict[int, LocalView] = {}
        self._previousViewsFrameIndex: int | None = None
        self._depthProcessor = depthProcessor
        self._depthEncoder = depthEncoder if depthEncoder is not None else depthBackend
        self._fusionHead = fusionHead or FusionHead(depthScoreWeight=depthScoreWeight)
        self._depthEnabled = (
            bool(depthEnabled)
            if depthEnabled is not None
            else self._depthProcessor is not None or self._depthEncoder is not None
        )
        self._depthTemplates: list[object] = []
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
        if self._depthEnabled and template.depth is not None and self._depthEncoder is not None:
            self._depthTemplates = [self._encodeDepthTemplate(template, templateBox)]
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
        pendingDepthFeature = self._prepareDepthCommand(command)
        self._templates.apply(self._hitBackend, command, self._previousViews)
        self._commitDepthCommand(command, pendingDepthFeature)
        snapshot = self._templates.snapshot()
        inferenceStartedNs = perf_counter_ns()
        predictions = self._hitBackend.inferBatch(
            tuple(view.rgb for view in views), snapshot.features
        )
        depthScores = self._inferDepthScores(views, tuple(self._depthTemplates))
        sharedInferenceNs = (
            (perf_counter_ns() - inferenceStartedNs) // len(views) if views else 0
        )
        observations: list[LocalObservation] = []
        for index, (view, prediction) in enumerate(zip(views, predictions, strict=True)):
            postprocessStartedNs = perf_counter_ns()
            if not self._depthEnabled or view.depth is None:
                observations.append(
                    buildRgbObservation(
                        view,
                        prediction,
                        sharedInferenceNs + perf_counter_ns() - postprocessStartedNs,
                    )
                )
                continue
            depthSummary = (
                self._depthProcessor.summarizeLocal(view, prediction.bbox)
                if self._depthProcessor is not None
                else None
            )
            depthScore = depthScores.get(index, 0.0)
            if depthSummary is None:
                depthScore = 0.0
            elif self._depthProcessor is not None and self._depthEncoder is None:
                depthScore = self._depthProcessor.score(depthSummary)
            fusedScore = self._fusionHead.fuse(
                prediction.appearanceScore,
                depthScore,
                contextScore=prediction.modelScore,
                depthAvailable=depthSummary is not None,
            )
            observations.append(
                LocalObservation(
                    viewId=view.spec.viewId,
                    bbox=buildRgbObservation(view, prediction, 0).bbox,
                    modelScore=prediction.modelScore,
                    appearanceScore=prediction.appearanceScore,
                    depthScore=depthScore,
                    fusedScore=fusedScore,
                    depthSummary=depthSummary,
                    latencyNs=(
                        sharedInferenceNs + perf_counter_ns() - postprocessStartedNs
                    ),
                )
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
        if self._depthEncoder is not None and hasattr(self._depthEncoder, "close"):
            self._depthEncoder.close()
        self._templates.clear()
        self._previousViews.clear()
        self._previousViewsFrameIndex = None
        self._depthTemplates.clear()
        self._closed = True

    def _encodeDepthTemplate(self, view: LocalView, bbox: BBoxXYWH | None = None) -> object:
        if self._depthProcessor is None or self._depthEncoder is None or view.depth is None:
            raise ProtocolError("depth template requires a depth processor, encoder and depth view")
        image = self._depthProcessor.colorize(view.depth)
        encoder = self._depthEncoder
        if hasattr(encoder, "encodeTemplate"):
            try:
                return encoder.encodeTemplate(image, bbox)
            except TypeError:
                return encoder.encodeTemplate(image)
        if hasattr(encoder, "encode"):
            return encoder.encode(image)
        raise ProtocolError("depth encoder must implement encode or encodeTemplate")

    def _inferDepthScores(
        self,
        views: Sequence[LocalView],
        templateFeatures: tuple[object, ...],
    ) -> dict[int, float]:
        if self._depthProcessor is None or self._depthEncoder is None:
            return {}
        indexedImages = tuple(
            (index, self._depthProcessor.colorize(view.depth))
            for index, view in enumerate(views)
            if view.depth is not None
        )
        if not indexedImages:
            return {}
        try:
            results = self._depthEncoder.inferBatch(
                tuple(image for _, image in indexedImages), templateFeatures
            )
        except (ProtocolError, ModelError):
            raise
        except Exception as error:
            raise ModelError(f"depth branch inference failed: {error}") from error
        scores: dict[int, float] = {}
        for (index, _), result in zip(indexedImages, results, strict=True):
            score = (
                result.depthScore
                if isinstance(result, DepthPrediction)
                else getattr(result, "depthScore", getattr(result, "appearanceScore", result))
            )
            try:
                numericScore = float(score)
            except (TypeError, ValueError) as error:
                raise ModelError("depth branch returned an invalid score") from error
            if not np.isfinite(numericScore) or not 0.0 <= numericScore <= 1.0:
                raise ModelError("depth branch score must be in [0, 1]")
            scores[index] = numericScore
        return scores

    def _prepareDepthCommand(self, command: TemplateCommand) -> object | None:
        if not self._depthEnabled or self._depthEncoder is None:
            return None
        if command.kind.name == "RESET_TO_ANCHOR":
            return None
        if command.kind.name not in {"UPDATE_RECENT", "UPDATE_STABLE"}:
            return None
        if not getattr(self._depthEncoder, "supportsOnlineTemplates", True):
            raise ProtocolError("depth HiT session does not support online template commands")
        if command.viewId is None or command.localBox is None:
            return None
        view = self._previousViews.get(command.viewId)
        if view is None or view.depth is None or self._depthProcessor is None:
            return None
        return self._encodeDepthTemplate(view, command.localBox)

    def _commitDepthCommand(self, command: TemplateCommand, feature: object | None) -> None:
        if not self._depthEnabled or self._depthEncoder is None:
            return
        if command.kind.name == "RESET_TO_ANCHOR":
            self._depthTemplates = self._depthTemplates[:1]
            return
        if feature is None or command.kind.name not in {"UPDATE_RECENT", "UPDATE_STABLE"}:
            return
        if command.kind.name == "UPDATE_RECENT":
            if len(self._depthTemplates) == 1:
                self._depthTemplates.append(feature)
            else:
                self._depthTemplates[1] = feature
        else:
            while len(self._depthTemplates) < 3:
                self._depthTemplates.append(self._depthTemplates[0])
            self._depthTemplates[2] = feature


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
