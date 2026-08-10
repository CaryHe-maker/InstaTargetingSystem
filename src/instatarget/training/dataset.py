"""Lazy training samples backed by the same format-neutral frame source."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from instatarget.core.types import BBoxXYWH, FramePacket
from instatarget.data.pseudo_track_builder import PseudoTrackBuilder
from instatarget.data.registry import DatasetSource, openDataset


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One model-ready sample; no torch dependency is required at this layer."""

    frame: FramePacket
    targetInstanceId: int
    targetBox: BBoxXYWH
    visible: bool

    @property
    def rgb(self) -> np.ndarray:
        return self.frame.rgb


class AirSim360TrainingDataset:
    """Iterate aligned RGB/depth/mask frames and derive pseudo boxes.

    The object deliberately yields NumPy-backed :class:`TrainingSample`s so a
    PyTorch/ONNX input adapter can be added later without coupling the data
    reader to a training framework.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        sequenceId: str | None = None,
        targetInstanceId: int | None = None,
        format: str = "auto",
    ) -> None:
        self.root = str(root)
        self.sequenceId = sequenceId
        self.format = format
        self._targetInstanceId = targetInstanceId
        self._source: DatasetSource | None = None
        self._frameCount = 0
        self._prepare()

    def _prepare(self) -> None:
        source = openDataset(self.root, format=self.format, sequenceId=self.sequenceId)
        self._source = source
        self._frameCount = max(int(source.frameCount), 0)
        if self._targetInstanceId is None:
            source.close()
            raise ValueError("targetInstanceId is required for deterministic training")
        source.close()

    @property
    def targetInstanceId(self) -> int:
        if self._targetInstanceId is None:
            raise ValueError("dataset contains no instance mask; targetInstanceId is required")
        return self._targetInstanceId

    def __len__(self) -> int:
        return self._frameCount

    def __iter__(self) -> Iterator[TrainingSample]:
        source = openDataset(self.root, format=self.format, sequenceId=self.sequenceId)
        builder = PseudoTrackBuilder()
        try:
            while (frame := source.read()) is not None:
                box, visible = builder.buildPseudoGroundTruth(frame, self.targetInstanceId)
                yield TrainingSample(frame, self.targetInstanceId, box, visible)
        finally:
            source.close()

    def __getitem__(self, index: int) -> TrainingSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source = openDataset(self.root, format=self.format, sequenceId=self.sequenceId)
        try:
            frame = None
            for _ in range(index + 1):
                frame = source.read()
            if frame is None:
                raise IndexError(index)
            box, visible = PseudoTrackBuilder().buildPseudoGroundTruth(frame, self.targetInstanceId)
            return TrainingSample(frame, self.targetInstanceId, box, visible)
        finally:
            source.close()


__all__ = ["AirSim360TrainingDataset", "TrainingSample"]
