import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from instatarget.app.driver import buildRuntime, finalizeSink, openSink, runTracking
from instatarget.core.config import loadConfig
from instatarget.core.types import BBoxXYWH
from instatarget.data.frame_source import FrameSource
from instatarget.eval.profiler import RuntimeProfiler
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
            runtime = buildRuntime(
                replace(
                    config,
                    scoring=replace(config.scoring, calibrationArtifact=None),
                ),
                hitSessionFactory=_TestHiTSession,
                allowUncalibratedScoring=True,
            )
            source = FrameSource(sequenceId="sequence")
            source.open(str(root))
            output = Path(directory) / "result.txt"
            openSink(runtime.sink, str(output))
            timer = _BoundaryTimer()
            resultRecorder = _CheckingResultRecorder(timer)
            profiler = RuntimeProfiler()

            resultCount = runTracking(
                source=_CheckingSource(source, timer),
                initialBox=BBoxXYWH(4.0, 4.0, 8.0, 8.0),
                geometry=runtime.geometry,
                controller=runtime.controller,
                backend=runtime.backend,
                sink=_CheckingSink(runtime.sink, timer),
                recorder=_CheckingRecorder(runtime.recorder, timer),
                resultRecorder=resultRecorder,
                processingTimer=timer,
                profiler=profiler,
                scoreCalibration=runtime.scoreCalibration,
            )
            finalizeSink(runtime.sink, resultCount)

            self.assertEqual(resultCount, 2)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(timer.starts, timer.stops)
            self.assertFalse(timer.active)
            self.assertEqual(resultRecorder.roundCounts[0], 0)
            self.assertIsNotNone(resultRecorder.roundCounts[1])
            self.assertGreaterEqual(resultRecorder.roundCounts[1], 1)
            self.assertEqual([row["frameIndex"] for row in profiler.frameRows], [0, 1])
            self.assertIn("total", profiler.summarizeFrames())


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


class _BoundaryTimer:
    def __init__(self) -> None:
        self.active = False
        self.starts = 0
        self.stops = 0

    def startProcessing(self) -> None:
        if self.active:
            raise AssertionError("processing interval started twice")
        self.active = True
        self.starts += 1

    def stopProcessing(self) -> None:
        if not self.active:
            raise AssertionError("processing interval stopped while inactive")
        self.active = False
        self.stops += 1


class _CheckingSource:
    def __init__(self, source, timer: _BoundaryTimer) -> None:
        self._source = source
        self._timer = timer

    def read(self):
        if not self._timer.active:
            raise AssertionError("frame read must be timed")
        return self._source.read()


class _CheckingSink:
    def __init__(self, sink, timer: _BoundaryTimer) -> None:
        self._sink = sink
        self._timer = timer

    def write(self, result) -> None:
        if self._timer.active:
            raise AssertionError("result sink must not be timed")
        self._sink.write(result)


class _CheckingRecorder:
    def __init__(self, recorder, timer: _BoundaryTimer) -> None:
        self._recorder = recorder
        self._timer = timer

    def _outsideProcessing(self) -> None:
        if self._timer.active:
            raise AssertionError("intermediate visualization must not be timed")

    def recordLocalRgb(self, *args, **kwargs):
        self._outsideProcessing()
        return self._call("recordLocalRgb", *args, **kwargs)

    def recordBackendBoxes(self, *args, **kwargs):
        self._outsideProcessing()
        return self._call("recordBackendBoxes", *args, **kwargs)

    def recordGeometryBoxes(self, *args, **kwargs):
        self._outsideProcessing()
        return self._call("recordGeometryBoxes", *args, **kwargs)

    def _call(self, methodName, *args, **kwargs):
        if self._recorder is None:
            return ()
        return getattr(self._recorder, methodName)(*args, **kwargs)


class _CheckingResultRecorder:
    def __init__(self, timer: _BoundaryTimer) -> None:
        self._timer = timer
        self.roundCounts: list[int | None] = []

    def record(self, *_args, **kwargs) -> None:
        if self._timer.active:
            raise AssertionError("result visualization must not be timed")
        self.roundCounts.append(kwargs.get("roundCount"))


if __name__ == "__main__":
    unittest.main()
