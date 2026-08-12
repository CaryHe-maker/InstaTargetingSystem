"""Final per-frame result visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from instatarget.core.types import FramePacket, TrackResult
from instatarget.visualization.image import drawBoxRgb
from instatarget.visualization.png import writeRgbPng


@dataclass(frozen=True, slots=True)
class ResultVisualizationRecorder:
    """Write exactly one ERP image per frame with the committed green box."""

    outputRoot: Path

    def record(
        self,
        frame: FramePacket,
        result: TrackResult,
        *,
        stateScore: float | None = None,
    ) -> Path:
        outputPath = self.outputRoot / f"frame_{int(frame.frameIndex):06d}.png"
        scoreLabel = "stateScore=N/A" if stateScore is None else f"stateScore={stateScore:.4f}"
        return writeRgbPng(
            outputPath,
            drawBoxRgb(
                frame.rgb,
                result.bbox,
                wrapHorizontal=True,
                label=scoreLabel,
            ),
        )


__all__ = ["ResultVisualizationRecorder"]
