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
    """Keep the frame-zero anchor as the only runtime template."""

    def __init__(self, trackingConfig: TrackingConfig) -> None:
        del trackingConfig

    def decide(
        self,
        status: TrackStatus,
        stableFrames: int,
        aggregate: FrameAggregate | None,
    ) -> TemplateDecision:
        del status, stableFrames, aggregate
        return TemplateDecision(TemplateCommandKind.KEEP)


__all__ = ["TemplateDecision", "TemplatePolicy"]
