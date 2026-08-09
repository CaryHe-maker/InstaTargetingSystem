import tempfile
import unittest
from pathlib import Path

import numpy as np

from instatarget.app.driver import SmokeHiTSession
from instatarget.controller import SimpleGeometryTrackController
from instatarget.core.config import VisualizationConfig
from instatarget.core.types import BBoxXYWH
from instatarget.data import DirectoryFrameSource
from instatarget.geometry import SphericalGeometryImpl
from instatarget.tracker import HiTBackend, TrackerBackendImpl
from instatarget.visualization import VisualizationRecorder
from instatarget.visualization.png import writeRgbPng


class SmokeControllerTest(unittest.TestCase):
    def testDirectoryFrameSourceReadsImagesInOrder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = np.full((6, 8, 3), 10, dtype=np.uint8)
            b = np.full((6, 8, 3), 40, dtype=np.uint8)
            writeRgbPng(root / "b.png", b)
            writeRgbPng(root / "a.png", a)

            source = DirectoryFrameSource()
            source.open(str(root))
            first = source.read()
            second = source.read()
            third = source.read()

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertIsNone(third)
            np.testing.assert_array_equal(first.rgb, a)
            np.testing.assert_array_equal(second.rgb, b)
            self.assertEqual(str(first.sequenceId), root.name)

    def testSmokeControllerWritesVisualizationArtifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputRoot = root / "input"
            outputRoot = root / "output"
            inputRoot.mkdir()
            frame0 = np.full((24, 32, 3), 18, dtype=np.uint8)
            frame1 = np.full((24, 32, 3), 90, dtype=np.uint8)
            writeRgbPng(inputRoot / "000.png", frame0)
            writeRgbPng(inputRoot / "001.png", frame1)

            source = DirectoryFrameSource()
            source.open(str(inputRoot))
            frames = []
            while True:
                frame = source.read()
                if frame is None:
                    break
                frames.append(frame)

            visualization = VisualizationRecorder(
                VisualizationConfig(
                    enabled=True,
                    outputRoot=outputRoot,
                    stages=frozenset({"local_rgb", "backend_box", "geometry_box"}),
                )
            )
            controller = SimpleGeometryTrackController(
                geometry=SphericalGeometryImpl(),
                tracker=TrackerBackendImpl(HiTBackend(SmokeHiTSession())),
                visualization=visualization,
                viewWidthPx=16,
                viewHeightPx=16,
            )

            results = controller.run(frames, BBoxXYWH(xPx=8.0, yPx=6.0, widthPx=8.0, heightPx=8.0))

            self.assertEqual(len(results), 2)
            self.assertGreater(results[1].confidence, 0.0)
            self.assertEqual(len(list(outputRoot.rglob("*.png"))), 6)
            self.assertTrue((outputRoot / inputRoot.name / "frame_000000" / "local_rgb").exists())
            self.assertTrue((outputRoot / inputRoot.name / "frame_000001" / "geometry_box").exists())


if __name__ == "__main__":
    unittest.main()
