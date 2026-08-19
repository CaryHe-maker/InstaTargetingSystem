"""Build indexed training manifests from the labeled 360 tracking dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import pi
from pathlib import Path

from instatarget.core.errors import ConfigError
from instatarget.core.types import BFoV
from instatarget.geometry.projection_math import makeSphericalPoint
from instatarget.geometry.spherical_geometry import SphericalGeometryImpl


@dataclass(frozen=True, slots=True)
class SequenceFiles:
    group: str
    sequenceId: str
    videoPath: Path
    groundtruthPath: Path


def discoverSequences(root: str | Path) -> tuple[SequenceFiles, ...]:
    dataRoot = Path(root).expanduser().resolve()
    sequences: list[SequenceFiles] = []
    for group in ("train_sim", "train_real"):
        groupRoot = dataRoot / group
        sequenceList = groupRoot / "seqlist.txt"
        if not sequenceList.is_file():
            raise ConfigError(f"training sequence list does not exist: {sequenceList}")
        names = tuple(
            line.strip()
            for line in sequenceList.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
        if len(names) != len(set(names)):
            raise ConfigError(f"duplicate sequence names in {sequenceList}")
        for name in names:
            directory = groupRoot / name
            video = directory / "video.mp4"
            groundtruth = directory / "groundtruth.txt"
            if not video.is_file() or not groundtruth.is_file():
                raise ConfigError(f"sequence is incomplete: {directory}")
            sequences.append(
                SequenceFiles(
                    group=group,
                    sequenceId=f"{group}/{name}",
                    videoPath=video,
                    groundtruthPath=groundtruth,
                )
            )
    return tuple(sequences)


def assignSequenceSplits(
    sequences: Sequence[SequenceFiles],
    *,
    seed: int,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, str]:
    values = ratios or {
        "train": 0.70,
        "validation": 0.15,
        "calibration": 0.05,
        "holdout": 0.10,
    }
    if set(values) != {"train", "validation", "calibration", "holdout"}:
        raise ConfigError("split ratios must define train/validation/calibration/holdout")
    if any(value < 0.0 for value in values.values()) or abs(sum(values.values()) - 1.0) > 1e-9:
        raise ConfigError("split ratios must be non-negative and sum to one")
    result: dict[str, str] = {}
    for group in sorted({item.group for item in sequences}):
        members = sorted(
            (item for item in sequences if item.group == group),
            key=lambda item: _stableKey(seed, item.sequenceId),
        )
        counts = _allocateCounts(len(members), values)
        offset = 0
        for split in ("train", "validation", "calibration", "holdout"):
            for item in members[offset : offset + counts[split]]:
                result[item.sequenceId] = split
            offset += counts[split]
    return result


def buildManifest(
    root: str | Path,
    output: str | Path,
    *,
    seed: int,
    excludedSequenceIds: Iterable[str] = (),
) -> dict[str, int]:
    import cv2

    sequences = discoverSequences(root)
    excluded = frozenset(excludedSequenceIds)
    unknown = excluded - {item.sequenceId for item in sequences}
    if unknown:
        raise ConfigError(f"excluded sequence IDs were not found: {sorted(unknown)}")
    included = tuple(item for item in sequences if item.sequenceId not in excluded)
    splits = assignSequenceSplits(included, seed=seed)
    geometry = SphericalGeometryImpl()
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    counts = {name: 0 for name in ("train", "validation", "calibration", "holdout")}
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for sequence in included:
            capture = cv2.VideoCapture(str(sequence.videoPath))
            try:
                if not capture.isOpened():
                    raise ConfigError(f"cannot open training video: {sequence.videoPath}")
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frameCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
            finally:
                capture.release()
            if width <= 0 or height <= 0 or frameCount <= 0:
                raise ConfigError(f"invalid video metadata: {sequence.videoPath}")
            if not fps > 0.0:
                fps = 30.0
            groundtruth = _readGroundtruth(sequence.groundtruthPath)
            if len(groundtruth) != frameCount:
                raise ConfigError(
                    "groundtruth/video length mismatch: "
                    f"sequence={sequence.sequenceId}, gt={len(groundtruth)}, video={frameCount}"
                )
            split = splits[sequence.sequenceId]
            for frameIndex, (yawDeg, pitchDeg, horizontalDeg, verticalDeg) in enumerate(
                groundtruth
            ):
                targetAbsent = horizontalDeg == 0.0 or verticalDeg == 0.0
                labelValid = (
                    -90.0 <= pitchDeg <= 90.0
                    and 0.0 < horizontalDeg < 180.0
                    and 0.0 < verticalDeg < 180.0
                )
                labelUsable = labelValid or targetAbsent
                visible = labelValid
                bbox = None
                if labelValid:
                    bfov = BFoV(
                        center=makeSphericalPoint(yawDeg * pi / 180.0, pitchDeg * pi / 180.0),
                        horizontalFovRad=horizontalDeg * pi / 180.0,
                        verticalFovRad=verticalDeg * pi / 180.0,
                    )
                    erpBox = geometry.bfovToBbox(bfov, width, height)
                    bbox = [erpBox.xPx, erpBox.yPx, erpBox.widthPx, erpBox.heightPx]
                record = {
                    "sequenceId": sequence.sequenceId,
                    "videoPath": str(sequence.videoPath),
                    "frameIndex": frameIndex,
                    "timestamp": frameIndex / fps,
                    "targetInstanceId": 0,
                    "bbox": bbox,
                    "visible": visible,
                    "occluded": targetAbsent,
                    "truncated": False,
                    "width": width,
                    "height": height,
                    "labelSource": (
                        "official_bfov" if labelUsable else "official_bfov_invalid"
                    ),
                    "labelQuality": 1.0 if labelUsable else 0.0,
                    "split": split,
                    "difficultType": (
                        "normal"
                        if labelValid
                        else (
                            "target_absent"
                            if targetAbsent
                            else "invalid_bfov"
                        )
                    ),
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                counts[split] += 1
    temporary.replace(destination)
    return counts


def _readGroundtruth(path: Path) -> tuple[tuple[float, float, float, float], ...]:
    result = []
    for lineNumber, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.replace("\t", ",").split(",")
        if len(fields) < 4:
            fields = line.split()
        if len(fields) < 4:
            raise ConfigError(f"invalid groundtruth line {path}:{lineNumber}")
        try:
            values = tuple(float(value.strip()) for value in fields[:4])
        except ValueError as error:
            raise ConfigError(f"invalid groundtruth line {path}:{lineNumber}") from error
        if values[2] < 0.0 or values[3] < 0.0:
            raise ConfigError(f"unsupported BFoV at {path}:{lineNumber}: {values}")
        result.append(values)
    if not result:
        raise ConfigError(f"groundtruth is empty: {path}")
    return tuple(result)


def _allocateCounts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    raw = {name: total * value for name, value in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    priority = sorted(raw, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in priority[:remainder]:
        counts[name] += 1
    return counts


def _stableKey(seed: int, sequenceId: str) -> bytes:
    return hashlib.sha256(f"{seed}:{sequenceId}".encode()).digest()


__all__ = [
    "SequenceFiles",
    "assignSequenceSplits",
    "buildManifest",
    "discoverSequences",
]
