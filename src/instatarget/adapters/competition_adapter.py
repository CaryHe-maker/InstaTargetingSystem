"""Competition input and output adapter."""

from __future__ import annotations

from dataclasses import dataclass

from instatarget.core.errors import DecodeError, OutputError
from instatarget.core.types import BBoxXYWH, TrackResult
from instatarget.geometry.seam import splitSeamBox, wrapPixelX


@dataclass(slots=True)
class CompetitionAdapter:
    """Convert internal track results to a competition-friendly box stream."""

    frameWidthPx: int
    frameHeightPx: int
    strategy: str = "split"

    def adaptResult(self, result: TrackResult) -> tuple[BBoxXYWH, ...]:
        box = result.bbox
        if self.strategy == "keep":
            return (box,)
        normalized = BBoxXYWH(
            xPx=wrapPixelX(box.xPx, self.frameWidthPx),
            yPx=box.yPx,
            widthPx=box.widthPx,
            heightPx=box.heightPx,
        )
        if normalized.xPx + normalized.widthPx <= self.frameWidthPx:
            return (normalized,)
        if self.strategy == "shift":
            shift = max(0.0, self.frameWidthPx - normalized.widthPx)
            return (
                BBoxXYWH(
                    xPx=max(0.0, min(shift, normalized.xPx)),
                    yPx=normalized.yPx,
                    widthPx=min(normalized.widthPx, float(self.frameWidthPx)),
                    heightPx=normalized.heightPx,
                ),
            )
        if self.strategy == "split":
            return splitSeamBox(normalized, self.frameWidthPx)
        raise OutputError(f"unsupported competition strategy: {self.strategy}")

    def formatResult(self, result: TrackResult) -> tuple[str, ...]:
        return tuple(_formatBox(box) for box in self.adaptResult(result))

    def parseLine(self, line: str) -> BBoxXYWH:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise DecodeError(f"invalid competition result line: {line!r}")
        try:
            return BBoxXYWH(*(float(part) for part in parts))
        except ValueError as error:
            raise DecodeError(f"invalid competition result line: {line!r}") from error


def _formatBox(box: BBoxXYWH) -> str:
    return ",".join(f"{value:.6f}" for value in (box.xPx, box.yPx, box.widthPx, box.heightPx))


__all__ = ["CompetitionAdapter"]
