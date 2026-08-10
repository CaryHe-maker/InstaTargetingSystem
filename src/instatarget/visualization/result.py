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

    def record(self, frame: FramePacket, result: TrackResult) -> Path:
        outputPath = self.outputRoot / f"frame_{int(frame.frameIndex):06d}.png"
        return writeRgbPng(
            outputPath,
            drawBoxRgb(frame.rgb, result.bbox, wrapHorizontal=True),
        )


__all__ = ["ResultVisualizationRecorder"]
