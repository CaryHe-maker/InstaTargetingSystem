"""Template update policy owned by the DTC control thread."""

from __future__ import annotations

from dataclasses import dataclass

from instatarget.controller.decision_gate import FrameAggregate
from instatarget.core.config import TrackingConfig
from instatarget.core.types import BBoxXYWH, TemplateCommandKind, TrackStatus


@dataclass(frozen=True, slots=True)
class TemplateDecision:
    kind: TemplateCommandKind
    viewId: int | None = None
    localBox: BBoxXYWH | None = None


class TemplatePolicy:
    """Protect anchor and dynamic slots from uncertain or recovery observations."""

    def __init__(self, trackingConfig: TrackingConfig) -> None:
        self._config = trackingConfig

    def decide(
        self,
        status: TrackStatus,
        stableFrames: int,
        aggregate: FrameAggregate | None,
    ) -> TemplateDecision:
        if status is not TrackStatus.TRACKING or aggregate is None or not aggregate.supported:
            return TemplateDecision(TemplateCommandKind.KEEP)
        if aggregate.localBox is None:
            return TemplateDecision(TemplateCommandKind.KEEP)
        stableThreshold = self._config.stableFramesBeforeUpdate
        recentThreshold = max(2, stableThreshold // 2)
        if stableFrames >= stableThreshold and stableFrames % stableThreshold == 0:
            return TemplateDecision(
                TemplateCommandKind.UPDATE_STABLE,
                aggregate.representativeViewId,
                aggregate.localBox,
            )
        if stableFrames == recentThreshold:
            return TemplateDecision(
                TemplateCommandKind.UPDATE_RECENT,
                aggregate.representativeViewId,
                aggregate.localBox,
            )
        return TemplateDecision(TemplateCommandKind.KEEP)


__all__ = ["TemplateDecision", "TemplatePolicy"]
