"""Official competition runner and BFoV submission format adapter."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TextIO

import numpy as np

from instatarget.app.driver import (
    buildRuntime,
    closeBackend,
    finalizeSink,
    openSink,
    runTracking,
)
from instatarget.core.config import AppConfig, loadConfig
from instatarget.core.errors import ConfigError, DecodeError, OutputError, ProtocolError
from instatarget.core.types import BFoV, FrameIndex, FramePacket, SequenceId, TrackResult
from instatarget.geometry import makeSphericalPoint

DEFAULT_DATASET_DIR = "data"
DEFAULT_RESULT_DIR = "result"
DEFAULT_CONFIG_PATH = "configs/RGBonly.yaml"


class OpenCvVideoSource:
    """Small OpenCV-backed frame source used by the offline evaluator."""

    def __init__(self, sequenceId: str) -> None:
        self.sequenceId = sequenceId
        self._capture = None
        self._firstFrame: np.ndarray | None = None
        self._cursor = 0
        self._fps = 30.0
        self.frameCount = 0

    @property
    def frameWidthPx(self) -> int:
        if self._firstFrame is None:
            raise DecodeError("video source is not open")
        return int(self._firstFrame.shape[1])

    @property
    def frameHeightPx(self) -> int:
        if self._firstFrame is None:
            raise DecodeError("video source is not open")
        return int(self._firstFrame.shape[0])

    def open(self, path: str) -> None:
        try:
            import cv2
        except ImportError as error:
            raise DecodeError("competition video input requires opencv-python-headless") from error
        self.close()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            raise DecodeError(f"cannot open video source: {path}")
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise DecodeError(f"video source has no readable frames: {path}")
        self._capture = capture
        self._firstFrame = _bgrToRgb(frame)
        self._cursor = 0
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        self._fps = fps if np.isfinite(fps) and fps > 0.0 else 30.0
        frameCount = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameCount = (
            max(0, int(frameCount)) if np.isfinite(frameCount) and frameCount > 0.0 else 0
        )

    def read(self) -> FramePacket | None:
        if self._capture is None or self._firstFrame is None:
            raise ProtocolError("video source is not open")
        if self._cursor == 0:
            rgb = self._firstFrame
        else:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                return None
            rgb = _bgrToRgb(frame)
        frame = FramePacket(
            sequenceId=SequenceId(self.sequenceId),
            frameIndex=FrameIndex(self._cursor),
            timestampNs=int(self._cursor * 1_000_000_000 / max(self._fps, 1e-6)),
            rgb=rgb,
        )
        self._cursor += 1
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._firstFrame = None
        self._cursor = 0
        self.frameCount = 0


class BfovResultSink:
    """Atomically write one official BFoV line for every tracked frame."""

    def __init__(self, initialBfov: BFoV | None = None) -> None:
        self._initialBfov = initialBfov
        self._stream: TextIO | None = None
        self._destination: Path | None = None
        self._partial: Path | None = None
        self._count = 0

    def open(self, destination: str) -> None:
        if self._stream is not None:
            raise ProtocolError("competition result sink is already open")
        self._destination = Path(destination).expanduser().resolve()
        self._destination.parent.mkdir(parents=True, exist_ok=True)
        self._partial = self._destination.with_name(self._destination.name + ".partial")
        self._stream = self._partial.open("w", encoding="utf-8", newline="\n")
        self._count = 0

    def write(self, result: TrackResult) -> None:
        if self._stream is None:
            raise ProtocolError("competition result sink is not open")
        expectedFrame = self._count
        if int(result.frameIndex) != expectedFrame:
            raise OutputError(
                f"competition result frame order mismatch: expected={expectedFrame}, "
                f"actual={result.frameIndex}"
            )
        if self._count == 0 and self._initialBfov is not None:
            self._stream.write(formatBfov(self._initialBfov))
        else:
            self._stream.write(formatCompetitionResult(result))
        self._stream.write("\n")
        self._count += 1

    def finalize(self, expectedFrameCount: int) -> None:
        if self._stream is None or self._destination is None or self._partial is None:
            raise ProtocolError("competition result sink is not open")
        if expectedFrameCount != self._count:
            actualCount = self._count
            self.close()
            raise OutputError(
                f"competition result count mismatch: expected={expectedFrameCount}, "
                f"actual={actualCount}"
            )
        self._stream.flush()
        self._stream.close()
        os.replace(self._partial, self._destination)
        self._stream = None
        self._destination = None
        self._partial = None

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = None
        self._destination = None
        self._partial = None
        self._count = 0


def formatCompetitionResult(result: TrackResult) -> str:
    """Format a TrackResult as clon,clat,fov_h,fov_v in degrees."""
    if not result.valid:
        return "0.000,0.000,0.000,0.000"
    return formatBfov(result.bfov)


def formatBfov(bfov: BFoV) -> str:
    """Format a valid BFoV using the official degree convention."""
    return ",".join(
        f"{value:.3f}"
        for value in (
            np.degrees(bfov.center.yawRad),
            np.degrees(bfov.center.pitchRad),
            np.degrees(bfov.horizontalFovRad),
            np.degrees(bfov.verticalFovRad),
        )
    )


def loadInitialBfov(path: str | Path) -> BFoV:
    values = [part.strip() for part in Path(path).read_text(encoding="utf-8").split(",")]
    if len(values) < 4:
        raise DecodeError(f"init.txt must contain clon,clat,fov_h,fov_v: {path}")
    try:
        clonDeg, clatDeg, horizontalDeg, verticalDeg = (float(value) for value in values[:4])
    except ValueError as error:
        raise DecodeError(f"invalid BFoV in {path}") from error
    if not -180.0 <= clonDeg < 180.0 or not -90.0 <= clatDeg <= 90.0:
        raise DecodeError(f"BFoV center is outside the official range in {path}")
    if not 0.0 < horizontalDeg < 180.0 or not 0.0 < verticalDeg < 180.0:
        raise DecodeError(f"BFoV field of view is outside the official range in {path}")
    return BFoV(
        center=makeSphericalPoint(np.radians(clonDeg), np.radians(clatDeg)),
        horizontalFovRad=float(np.radians(horizontalDeg)),
        verticalFovRad=float(np.radians(verticalDeg)),
    )


def listSequences(datasetDir: str | Path) -> list[str]:
    root = Path(datasetDir)
    if not root.is_dir():
        raise DecodeError(f"dataset directory does not exist: {root}")
    seqlist = root / "seqlist.txt"
    if seqlist.is_file():
        names = [line.strip() for line in seqlist.read_text(encoding="utf-8").splitlines()]
        names = [name for name in names if name and not name.startswith("#")]
        missing = [name for name in names if not (root / name).is_dir()]
        if missing:
            raise DecodeError(f"seqlist.txt references missing sequence directories: {missing}")
        return names
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and findVideo(path) is not None and (path / "init.txt").is_file()
    )


def findVideo(sequenceDir: str | Path) -> Path | None:
    videos = sorted(Path(sequenceDir).glob("*.mp4"))
    return videos[0] if videos else None


def trackOneSequence(
    sequenceDir: str | Path,
    configPath: str | Path = DEFAULT_CONFIG_PATH,
    resultPath: str | Path | None = None,
) -> int:
    sequencePath = Path(sequenceDir)
    video = findVideo(sequencePath)
    if video is None:
        raise DecodeError(f"sequence has no .mp4 video: {sequencePath}")
    initialBfov = loadInitialBfov(sequencePath / "init.txt")
    config = loadConfig(configPath)
    requireRgbOnlyConfig(config)
    runtime = buildRuntime(config)
    source = OpenCvVideoSource(sequencePath.name)
    sink = BfovResultSink(initialBfov)
    resultCount = 0
    try:
        source.open(str(video))
        initialBox = runtime.geometry.bfovToBbox(
            initialBfov, source.frameWidthPx, source.frameHeightPx
        )
        if resultPath is None:
            raise OutputError("resultPath is required for competition tracking")
        openSink(sink, str(resultPath))
        resultCount = runTracking(
            source=source,
            initialBox=initialBox,
            geometry=runtime.geometry,
            controller=runtime.controller,
            backend=runtime.backend,
            sink=sink,
            depthProcessor=runtime.depthProcessor,
            recorder=runtime.recorder,
        )
        finalizeSink(sink, resultCount)
        return resultCount
    finally:
        try:
            sink.close()
        finally:
            closeBackend(runtime.backend)
            source.close()


def runCompetition(
    datasetDir: str | Path | None = None,
    resultDir: str | Path | None = None,
    configPath: str | Path | None = None,
) -> int:
    dataset = Path(datasetDir or os.environ.get("DATASET_DIR", DEFAULT_DATASET_DIR))
    output = Path(resultDir or os.environ.get("RESULT_DIR", DEFAULT_RESULT_DIR))
    config = Path(configPath or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH))
    sequences = listSequences(dataset)
    if not sequences:
        raise DecodeError(f"no valid sequences found in {dataset}")
    output.mkdir(parents=True, exist_ok=True)
    totalFrames = 0
    start = time.monotonic()
    for index, sequence in enumerate(sequences, 1):
        frames = trackOneSequence(
            sequenceDir=dataset / sequence,
            configPath=config,
            resultPath=output / f"{sequence}.txt",
        )
        totalFrames += frames
        print(f"[competition] [{index}/{len(sequences)}] {sequence}: {frames} frames")
    elapsed = max(time.monotonic() - start, 1e-9)
    print(f"[competition] completed {totalFrames} frames ({totalFrames / elapsed:.1f} FPS)")
    return 0


def requireRgbOnlyConfig(config: AppConfig) -> None:
    """Reject depth-enabled settings on the official RGB-only path."""
    if config.depth.enabled:
        raise ConfigError("competition submission requires depth.enabled=false")
    if config.backendFusion.depthScoreWeight != 0.0:
        raise ConfigError("competition submission requires depthScoreWeight=0")


def _bgrToRgb(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame[:, :, ::-1])


__all__ = [
    "BfovResultSink",
    "OpenCvVideoSource",
    "formatBfov",
    "formatCompetitionResult",
    "findVideo",
    "listSequences",
    "loadInitialBfov",
    "requireRgbOnlyConfig",
    "runCompetition",
    "trackOneSequence",
]
