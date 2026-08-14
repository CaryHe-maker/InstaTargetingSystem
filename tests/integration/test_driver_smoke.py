import tempfile
import unittest
from pathlib import Path

import numpy as np

from instatarget.app.driver import buildRuntime, finalizeSink, openSink, runTracking
from instatarget.core.config import loadConfig
from instatarget.core.types import BBoxXYWH
from instatarget.data.frame_source import FrameSource
from instatarget.tracker.hit_backend import HiTPrediction
from instatarget.visualization.png import writeRgbPng

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class DriverSmokeTest(unittest.TestCase):
    def testTrackingDriverProducesTwoResults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sequence"
            root.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                rgb = np.full((16, 24, 3), 40 + index * 10, dtype=np.uint8)
                writeRgbPng(root / f"frame_{index:04d}.png", rgb)

            config = loadConfig(REPOSITORY_ROOT / "configs" / "RGBonly.yaml")
            runtime = buildRuntime(config, hitSessionFactory=_TestHiTSession)
            source = FrameSource(sequenceId="sequence")
            source.open(str(root))
            output = Path(directory) / "result.txt"
            openSink(runtime.sink, str(output))

            resultCount = runTracking(
                source=source,
                initialBox=BBoxXYWH(4.0, 4.0, 8.0, 8.0),
                geometry=runtime.geometry,
                controller=runtime.controller,
                backend=runtime.backend,
                sink=runtime.sink,
                depthProcessor=runtime.depthProcessor,
                recorder=runtime.recorder,
            )
            finalizeSink(runtime.sink, resultCount)

            self.assertEqual(resultCount, 2)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)


class _TestHiTSession:
    supportsOnlineTemplates = True

    def __init__(self, _config) -> None:
        pass

    def encodeTemplate(self, _rgb, bbox: BBoxXYWH) -> object:
        return bbox

    def infer(self, rgb, templateFeatures) -> HiTPrediction:
        box = templateFeatures[-1]
        width = min(float(rgb.shape[1]), box.widthPx)
        height = min(float(rgb.shape[0]), box.heightPx)
        return HiTPrediction(BBoxXYWH(box.xPx, box.yPx, width, height), 0.99, 0.99)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
