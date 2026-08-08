import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from instatarget.core.config import VisualizationConfig
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    LocalObservation,
    LocalView,
    ProjectedObservation,
    SequenceId,
    SphericalPoint,
    ViewSpec,
)
from instatarget.visualization import FLUORESCENT_GREEN_RGB, VisualizationRecorder


class VisualizationRecorderTest(unittest.TestCase):
    def testDisabledRecorderDoesNotCreateOutputRoot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputRoot = Path(directory) / "visualization"
            recorder = _recorder(outputRoot, enabled=False)

            artifacts = recorder.recordLocalRgb(_frame(), [_view()])

            self.assertEqual(artifacts, ())
            self.assertFalse(outputRoot.exists())

    def testWritesLocalAndExistingDepthRgbWithoutChangingPixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = _recorder(Path(directory))
            frame = _frame()
            view = _view()
            depthRgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

            localPaths = recorder.recordLocalRgb(frame, [view])
            depthPaths = recorder.recordDepthRgb(frame, {view.spec.viewId: depthRgb})

            np.testing.assert_array_equal(_readRgbPng(localPaths[0]), view.rgb)
            np.testing.assert_array_equal(_readRgbPng(depthPaths[0]), depthRgb)
            self.assertIn("frame_000007", depthPaths[0].as_posix())
            self.assertIn("depth_rgb", depthPaths[0].as_posix())

    def testWritesBackendBoxOverLocalRgbInFluorescentGreen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = _recorder(Path(directory), stages={"backend_box"})
            frame = _frame()
            view = _view()
            originalRgb = view.rgb.copy()
            observation = _localObservation(BBoxXYWH(1.0, 1.0, 4.0, 3.0))

            paths = recorder.recordBackendBoxes(frame, [view], [observation])

            annotated = _readRgbPng(paths[0])
            np.testing.assert_array_equal(annotated[1, 1], FLUORESCENT_GREEN_RGB)
            np.testing.assert_array_equal(annotated[3, 4], FLUORESCENT_GREEN_RGB)
            np.testing.assert_array_equal(view.rgb, originalRgb)

    def testWritesWrappedGeometryBoxOverOriginalErpRgb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = _recorder(Path(directory), stages={"geometry_box"})
            frame = _frame()
            observation = _projectedObservation(BBoxXYWH(10.0, 2.0, 4.0, 3.0))

            paths = recorder.recordGeometryBoxes(frame, [observation])

            annotated = _readRgbPng(paths[0])
            np.testing.assert_array_equal(annotated[2, 10], FLUORESCENT_GREEN_RGB)
            np.testing.assert_array_equal(annotated[2, 0], FLUORESCENT_GREEN_RGB)
            np.testing.assert_array_equal(annotated[4, 1], FLUORESCENT_GREEN_RGB)
            np.testing.assert_array_equal(frame.rgb, np.zeros_like(frame.rgb))


def _recorder(
    outputRoot: Path,
    *,
    enabled: bool = True,
    stages: set[str] | None = None,
) -> VisualizationRecorder:
    selectedStages = stages or {"local_rgb", "depth_rgb", "backend_box", "geometry_box"}
    return VisualizationRecorder(
        VisualizationConfig(
            enabled=enabled,
            outputRoot=outputRoot,
            stages=frozenset(selectedStages),
        )
    )


def _frame() -> FramePacket:
    return FramePacket(
        sequenceId=SequenceId("sequence/a"),
        frameIndex=FrameIndex(7),
        timestampNs=1,
        rgb=np.zeros((6, 12, 3), dtype=np.uint8),
    )


def _view() -> LocalView:
    spec = ViewSpec(viewId=3, bfov=_bfov(), outputWidthPx=8, outputHeightPx=6)
    rgb = np.full((6, 8, 3), 40, dtype=np.uint8)
    return LocalView(spec=spec, rgb=rgb)


def _bfov() -> BFoV:
    center = SphericalPoint(x=1.0, y=0.0, z=0.0, yawRad=0.0, pitchRad=0.0)
    return BFoV(center=center, horizontalFovRad=1.0, verticalFovRad=1.0)


def _localObservation(bbox: BBoxXYWH) -> LocalObservation:
    return LocalObservation(
        viewId=3,
        bbox=bbox,
        modelScore=0.9,
        appearanceScore=0.8,
        depthScore=0.7,
        fusedScore=0.85,
        depthSummary=None,
        latencyNs=1,
    )


def _projectedObservation(bbox: BBoxXYWH) -> ProjectedObservation:
    return ProjectedObservation(
        viewId=3,
        bfov=_bfov(),
        bbox=bbox,
        modelScore=0.9,
        appearanceScore=0.8,
        motionScore=0.7,
        scaleScore=0.6,
        depthScore=0.5,
        fusedScore=0.85,
        depthSummary=None,
    )


def _readRgbPng(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    offset = 8
    compressed = bytearray()
    widthPx = 0
    heightPx = 0
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            widthPx, heightPx = struct.unpack(">II", data[:8])
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
    rows = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(
        heightPx, 1 + widthPx * 3
    )
    np.testing.assert_array_equal(rows[:, 0], 0)
    return rows[:, 1:].reshape(heightPx, widthPx, 3)


if __name__ == "__main__":
    unittest.main()
