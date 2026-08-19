"""Manifest-backed training pairs using the production spherical geometry."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from instatarget.core.config import TrainingDataConfig
from instatarget.core.errors import ConfigError, DecodeError
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    SequenceId,
    ViewSpec,
)
from instatarget.data.pseudo_track_builder import PseudoTrackBuilder
from instatarget.data.registry import DatasetSource, openDataset
from instatarget.geometry.projection_math import (
    cameraBasis,
    erpPixelToSphericalPoint,
    makeSphericalPoint,
)
from instatarget.geometry.spherical_geometry import SphericalGeometryImpl
from instatarget.training.augment import LocalViewAugmenter

_ALLOWED_SPLITS = frozenset({"train", "validation", "calibration", "holdout"})


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """Legacy ERP sample retained for AirSim mask-backed dataset inspection."""

    frame: FramePacket
    targetInstanceId: int
    targetBox: BBoxXYWH
    visible: bool

    @property
    def rgb(self) -> NDArray[np.uint8]:
        return self.frame.rgb


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    sequenceId: str
    videoPath: Path
    frameIndex: int
    timestamp: float
    targetInstanceId: int
    bbox: BBoxXYWH | None
    visible: bool
    occluded: bool
    truncated: bool
    width: int
    height: int
    labelSource: str
    labelQuality: float
    split: str
    difficultType: str = "normal"

    def __post_init__(self) -> None:
        if not self.sequenceId or self.frameIndex < 0 or self.timestamp < 0.0:
            raise ConfigError("manifest identity, frame index, and timestamp must be valid")
        if self.targetInstanceId < 0 or self.width <= 0 or self.height <= 0:
            raise ConfigError("manifest target ID and dimensions must be valid")
        if self.split not in _ALLOWED_SPLITS:
            raise ConfigError(f"unsupported manifest split: {self.split}")
        if not self.labelSource:
            raise ConfigError("manifest labelSource must be non-empty")
        if not 0.0 <= self.labelQuality <= 1.0:
            raise ConfigError("manifest labelQuality must be in [0, 1]")
        if self.visible and self.bbox is None:
            raise ConfigError("visible manifest records require a bbox")


@dataclass(frozen=True, slots=True)
class TrainingPair:
    templateRgb: NDArray[np.uint8]
    searchRgb: NDArray[np.uint8]
    targetBoxCxCyWh: NDArray[np.float32]
    present: bool
    labelQuality: float
    sequenceId: str
    templateFrameIndex: int
    searchFrameIndex: int
    difficultType: str
    labelSource: str
    searchFovDeg: float


class AirSim360TrainingDataset:
    """Inspect aligned RGB/mask frames without coupling to torch."""

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
        source = openDataset(self.root, format=self.format, sequenceId=self.sequenceId)
        try:
            self._frameCount = max(int(source.frameCount), 0)
            if targetInstanceId is None:
                raise ValueError("targetInstanceId is required for deterministic training")
        finally:
            source.close()

    @property
    def targetInstanceId(self) -> int:
        if self._targetInstanceId is None:
            raise ValueError("targetInstanceId is required")
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
        source: DatasetSource = openDataset(
            self.root, format=self.format, sequenceId=self.sequenceId
        )
        try:
            frame = None
            for _ in range(index + 1):
                frame = source.read()
            if frame is None:
                raise IndexError(index)
            box, visible = PseudoTrackBuilder().buildPseudoGroundTruth(
                frame, self.targetInstanceId
            )
            return TrainingSample(frame, self.targetInstanceId, box, visible)
        finally:
            source.close()


class VideoFrameDecoder:
    """Small per-worker LRU of VideoCapture instances with indexed seeks."""

    def __init__(self, cacheSize: int = 4) -> None:
        if cacheSize <= 0:
            raise ValueError("cacheSize must be positive")
        self._cacheSize = cacheSize
        self._captures: OrderedDict[Path, Any] = OrderedDict()

    def read(self, record: ManifestRecord) -> FramePacket:
        import cv2

        path = record.videoPath
        capture = self._captures.pop(path, None)
        if capture is None:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise DecodeError(f"cannot open training video: {path}")
        self._captures[path] = capture
        while len(self._captures) > self._cacheSize:
            _, old = self._captures.popitem(last=False)
            old.release()
        if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != record.frameIndex:
            capture.set(cv2.CAP_PROP_POS_FRAMES, record.frameIndex)
        ok, bgr = capture.read()
        if not ok or bgr is None:
            raise DecodeError(f"cannot decode {path} frame {record.frameIndex}")
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if rgb.shape[:2] != (record.height, record.width):
            raise DecodeError(
                "manifest dimensions do not match decoded frame: "
                f"expected={(record.height, record.width)}, actual={rgb.shape[:2]}"
            )
        return FramePacket(
            sequenceId=SequenceId(record.sequenceId),
            frameIndex=FrameIndex(record.frameIndex),
            timestampNs=int(round(record.timestamp * 1_000_000_000.0)),
            rgb=rgb,
        )

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def __del__(self) -> None:
        self.close()


class ManifestPairDataset:
    """Deterministic template/search pairs generated from a sequence-level manifest."""

    def __init__(
        self,
        config: TrainingDataConfig,
        split: str,
        *,
        seed: int,
        augmenter: LocalViewAugmenter | None = None,
    ) -> None:
        self.config = config
        self.split = split
        self.seed = seed
        self.epoch = 0
        self.augmenter = augmenter
        allRecords = loadManifest(config.manifest)
        _validateSequenceSplits(allRecords)
        self.records = tuple(
            item
            for item in allRecords
            if item.split == split and item.labelQuality >= config.minimumLabelQuality
        )
        if not self.records:
            raise ConfigError(f"manifest has no usable records for split={split}")
        grouped: dict[tuple[str, int], list[ManifestRecord]] = defaultdict(list)
        for item in self.records:
            grouped[(item.sequenceId, item.targetInstanceId)].append(item)
        self._groups = {
            key: tuple(sorted(items, key=lambda item: item.frameIndex))
            for key, items in grouped.items()
        }
        self._templateCandidates = {
            key: tuple(
                item
                for item in items
                if item.visible
                and not item.occluded
                and not item.truncated
                and item.bbox is not None
            )
            for key, items in self._groups.items()
        }
        missing = [key for key, items in self._templateCandidates.items() if not items]
        if missing:
            raise ConfigError(f"manifest targets have no stable template frame: {missing[:5]}")
        self._geometry = SphericalGeometryImpl()
        self._decoder: VideoFrameDecoder | None = None

    def __len__(self) -> int:
        return len(self.records)

    def setEpoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __getitem__(self, index: int) -> TrainingPair:
        searchRecord = self.records[index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        key = (searchRecord.sequenceId, searchRecord.targetInstanceId)
        templates = self._templateCandidates[key]
        eligible = tuple(
            item
            for item in templates
            if self.config.minFrameGap
            <= abs(searchRecord.frameIndex - item.frameIndex)
            <= self.config.maxFrameGap
        )
        if not eligible:
            eligible = templates
        templateRecord = eligible[int(rng.integers(0, len(eligible)))]
        decoder = self._getDecoder()
        templateFrame = decoder.read(templateRecord)
        searchFrame = decoder.read(searchRecord)
        assert templateRecord.bbox is not None

        templateBfov = self._geometry.bboxToBfov(
            templateRecord.bbox, templateRecord.width, templateRecord.height
        )
        templateSpec = _contextSpec(
            0,
            templateBfov,
            self.config.templateSizePx,
            contextFactor=2.0,
            maxFovDeg=self.config.maxFovDeg,
        )
        templateRgb = self._geometry.cropViews(templateFrame, (templateSpec,))[0].rgb

        forcedNegative = bool(rng.random() < self.config.negativeSampleRatio)
        isPositive = searchRecord.visible and searchRecord.bbox is not None and not forcedNegative
        targetBfov = (
            self._geometry.bboxToBfov(
                searchRecord.bbox, searchRecord.width, searchRecord.height
            )
            if searchRecord.bbox is not None
            else templateBfov
        )
        fovDeg = float(rng.uniform(self.config.minFovDeg, self.config.maxFovDeg))
        minimumFov = max(
            targetBfov.horizontalFovRad,
            targetBfov.verticalFovRad,
        ) * 1.5 * 180.0 / pi
        fovDeg = min(self.config.maxFovDeg, max(fovDeg, minimumFov))
        fovRad = fovDeg * pi / 180.0
        if isPositive:
            yawJitter = float(rng.uniform(-0.15, 0.15) * fovRad)
            pitchJitter = float(rng.uniform(-0.15, 0.15) * fovRad)
        else:
            direction = -1.0 if rng.random() < 0.5 else 1.0
            yawJitter = direction * min(pi * 0.9, fovRad * 1.2)
            pitchJitter = 0.0
        searchCenter = makeSphericalPoint(
            targetBfov.center.yawRad + yawJitter,
            targetBfov.center.pitchRad + pitchJitter,
        )
        searchSpec = ViewSpec(
            viewId=1,
            bfov=BFoV(
                center=searchCenter,
                horizontalFovRad=fovRad,
                verticalFovRad=fovRad,
            ),
            outputWidthPx=self.config.searchSizePx,
            outputHeightPx=self.config.searchSizePx,
        )
        searchRgb = self._geometry.cropViews(searchFrame, (searchSpec,))[0].rgb
        if self.augmenter is not None:
            templateRgb = self.augmenter(templateRgb, rng)
            searchRgb = self.augmenter(searchRgb, rng)
        localBox = (
            _erpBoxToNormalizedLocal(
                searchRecord.bbox,
                searchSpec,
                searchRecord.width,
                searchRecord.height,
            )
            if isPositive and searchRecord.bbox is not None
            else None
        )
        present = localBox is not None
        box = localBox if localBox is not None else np.zeros(4, dtype=np.float32)
        return TrainingPair(
            templateRgb=templateRgb,
            searchRgb=searchRgb,
            targetBoxCxCyWh=box,
            present=present,
            labelQuality=searchRecord.labelQuality,
            sequenceId=searchRecord.sequenceId,
            templateFrameIndex=templateRecord.frameIndex,
            searchFrameIndex=searchRecord.frameIndex,
            difficultType=("off_view" if forcedNegative else searchRecord.difficultType),
            labelSource=searchRecord.labelSource,
            searchFovDeg=fovDeg,
        )

    def _getDecoder(self) -> VideoFrameDecoder:
        if self._decoder is None:
            self._decoder = VideoFrameDecoder(self.config.decoderCacheSize)
        return self._decoder


def loadManifest(path: str | Path) -> tuple[ManifestRecord, ...]:
    manifestPath = Path(path).expanduser().resolve()
    if not manifestPath.is_file():
        raise ConfigError(f"training manifest does not exist: {manifestPath}")
    if manifestPath.suffix.lower() == ".csv":
        with manifestPath.open("r", encoding="utf-8-sig", newline="") as stream:
            rows: Sequence[Mapping[str, Any]] = tuple(csv.DictReader(stream))
    else:
        with manifestPath.open("r", encoding="utf-8") as stream:
            if manifestPath.suffix.lower() == ".jsonl":
                rows = tuple(json.loads(line) for line in stream if line.strip())
            else:
                loaded = json.load(stream)
                rows = loaded["records"] if isinstance(loaded, dict) else loaded
    if not isinstance(rows, Sequence) or not rows:
        raise ConfigError("training manifest must contain at least one record")
    return tuple(_parseRecord(row, manifestPath.parent) for row in rows)


def _parseRecord(raw: Mapping[str, Any], root: Path) -> ManifestRecord:
    required = {
        "sequenceId", "videoPath", "frameIndex", "timestamp", "targetInstanceId",
        "bbox", "visible", "occluded", "truncated", "width", "height",
        "labelSource", "labelQuality", "split",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ConfigError(f"manifest record fields missing: {missing}")
    videoPath = Path(str(raw["videoPath"])).expanduser()
    if not videoPath.is_absolute():
        videoPath = (root / videoPath).resolve()
    bboxRaw = raw["bbox"]
    if isinstance(bboxRaw, str):
        bboxRaw = json.loads(bboxRaw) if bboxRaw.strip() else None
    bbox = None
    if bboxRaw is not None:
        if isinstance(bboxRaw, Mapping):
            values = [bboxRaw[name] for name in ("x", "y", "width", "height")]
        elif isinstance(bboxRaw, Sequence) and not isinstance(bboxRaw, (str, bytes)):
            values = list(bboxRaw)
        else:
            raise ConfigError("manifest bbox must be null, [x,y,w,h], or a mapping")
        if len(values) != 4:
            raise ConfigError("manifest bbox must contain four values")
        bbox = BBoxXYWH(*(float(value) for value in values))
    return ManifestRecord(
        sequenceId=str(raw["sequenceId"]),
        videoPath=videoPath,
        frameIndex=int(raw["frameIndex"]),
        timestamp=float(raw["timestamp"]),
        targetInstanceId=int(raw["targetInstanceId"]),
        bbox=bbox,
        visible=_asBool(raw["visible"]),
        occluded=_asBool(raw["occluded"]),
        truncated=_asBool(raw["truncated"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        labelSource=str(raw["labelSource"]),
        labelQuality=float(raw["labelQuality"]),
        split=str(raw["split"]),
        difficultType=str(raw.get("difficultType", "normal")),
    )


def _validateSequenceSplits(records: Sequence[ManifestRecord]) -> None:
    sequenceSplits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sequenceSplits[record.sequenceId].add(record.split)
    leaking = {key: value for key, value in sequenceSplits.items() if len(value) > 1}
    if leaking:
        raise ConfigError(f"sequence-level split leakage detected: {leaking}")


def _contextSpec(
    viewId: int,
    target: BFoV,
    outputSize: int,
    *,
    contextFactor: float,
    maxFovDeg: float,
) -> ViewSpec:
    maxFov = maxFovDeg * pi / 180.0
    return ViewSpec(
        viewId=viewId,
        bfov=BFoV(
            center=target.center,
            horizontalFovRad=min(
                maxFov,
                max(5.0 * pi / 180.0, target.horizontalFovRad * contextFactor),
            ),
            verticalFovRad=min(
                maxFov,
                max(5.0 * pi / 180.0, target.verticalFovRad * contextFactor),
            ),
        ),
        outputWidthPx=outputSize,
        outputHeightPx=outputSize,
    )


def _erpBoxToNormalizedLocal(
    bbox: BBoxXYWH,
    spec: ViewSpec,
    frameWidth: int,
    frameHeight: int,
) -> NDArray[np.float32] | None:
    edge = np.linspace(0.0, 1.0, 33, dtype=np.float64)
    xs = np.concatenate(
        (
            bbox.xPx + edge * bbox.widthPx,
            np.full_like(edge, bbox.xPx + bbox.widthPx),
            bbox.xPx + (1.0 - edge) * bbox.widthPx,
            np.full_like(edge, bbox.xPx),
        )
    )
    ys = np.concatenate(
        (
            np.full_like(edge, bbox.yPx),
            bbox.yPx + edge * bbox.heightPx,
            np.full_like(edge, bbox.yPx + bbox.heightPx),
            bbox.yPx + (1.0 - edge) * bbox.heightPx,
        )
    )
    vectors = np.asarray(
        [
            (point.x, point.y, point.z)
            for point in (
                erpPixelToSphericalPoint(
                    float(x % frameWidth),
                    float(np.clip(y, 0.0, frameHeight)),
                    frameWidth,
                    frameHeight,
                )
                for x, y in zip(xs, ys, strict=True)
            )
        ],
        dtype=np.float64,
    )
    forward, right, up = cameraBasis(spec.bfov)
    depth = vectors @ forward
    visible = depth > 1e-8
    if not np.any(visible):
        return None
    horizontal = (vectors[visible] @ right) / depth[visible]
    vertical = (vectors[visible] @ up) / depth[visible]
    localX = (horizontal / np.tan(spec.bfov.horizontalFovRad / 2.0) + 1.0) / 2.0
    localY = (1.0 - vertical / np.tan(spec.bfov.verticalFovRad / 2.0)) / 2.0
    x0, x1 = float(np.clip(np.min(localX), 0.0, 1.0)), float(np.clip(np.max(localX), 0.0, 1.0))
    y0, y1 = float(np.clip(np.min(localY), 0.0, 1.0)), float(np.clip(np.max(localY), 0.0, 1.0))
    if x1 - x0 <= 1e-6 or y1 - y0 <= 1e-6:
        return None
    return np.asarray(
        ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0),
        dtype=np.float32,
    )


def _asBool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false", "1", "0"}:
        return value.lower() in {"true", "1"}
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ConfigError(f"manifest boolean value is invalid: {value!r}")


__all__ = [
    "AirSim360TrainingDataset",
    "ManifestPairDataset",
    "ManifestRecord",
    "TrainingPair",
    "TrainingSample",
    "VideoFrameDecoder",
    "loadManifest",
]
