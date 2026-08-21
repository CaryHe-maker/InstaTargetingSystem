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

    def __init__(self, trackingConfig: TrackingConfig, experimentVariant: str = "") -> None:
        del trackingConfig
        self._variant = experimentVariant
        self._uncertainStreak = 0

    def decide(
        self,
        status: TrackStatus,
        stableFrames: int,
        aggregate: FrameAggregate | None,
    ) -> TemplateDecision:
        if self._variant not in {"template_strict", "template_relaxed"}:
            return TemplateDecision(TemplateCommandKind.KEEP)
        if status is TrackStatus.UNCERTAIN:
            self._uncertainStreak += 1
            if self._uncertainStreak >= 3:
                self._uncertainStreak = 0
                return TemplateDecision(TemplateCommandKind.RESET_TO_ANCHOR)
            return TemplateDecision(TemplateCommandKind.KEEP)
        self._uncertainStreak = 0
        if aggregate is None or aggregate.localBox is None:
            return TemplateDecision(TemplateCommandKind.KEEP)
        requiredStableFrames = 3 if self._variant == "template_strict" else 2
        scoreThreshold = 0.80 if self._variant == "template_strict" else 0.60
        if status is TrackStatus.TRACKING and stableFrames >= requiredStableFrames:
            if aggregate.decisionScore >= scoreThreshold:
                return TemplateDecision(
                    TemplateCommandKind.UPDATE_RECENT,
                    viewId=aggregate.representativeViewId,
                    localBox=aggregate.localBox,
                )
        return TemplateDecision(TemplateCommandKind.KEEP)


__all__ = ["TemplateDecision", "TemplatePolicy"]
