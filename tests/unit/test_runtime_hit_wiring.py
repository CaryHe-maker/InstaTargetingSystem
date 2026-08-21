import unittest
from pathlib import Path
from unittest.mock import patch

from instatarget.app.driver import _PrefetchReader, buildRuntime, closeBackend
from instatarget.core.config import loadConfig
from instatarget.core.types import BBoxXYWH
from instatarget.eval.profiler import RuntimeProfiler
from instatarget.geometry import SphericalGeometryImpl
from instatarget.tracker.hit_backend import HiTPrediction
from instatarget.tracker.pytorch_hit_session import _devicesMatch, _resolveHitRoot
from instatarget.app.driver import _recordGeometryProfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class RuntimeHiTWiringTest(unittest.TestCase):
    def testBundledHiTRuntimeIsInsideSourcePackage(self) -> None:
        root = _resolveHitRoot()

        self.assertEqual(root, REPOSITORY_ROOT / "src" / "instatarget" / "vendor" / "hit")
        self.assertTrue((root / "configs" / "HiT_Small.yaml").is_file())
        self.assertTrue((root / "lib" / "models" / "HiT").is_dir())

    def testRgbOnlyCreatesOneHiTSession(self) -> None:
        factory = _SessionFactory()
        config = loadConfig(REPOSITORY_ROOT / "configs" / "RGBonly.yaml")

        runtime = buildRuntime(
            config,
            hitSessionFactory=factory,
            geometryFactory=SphericalGeometryImpl,
        )
        closeBackend(runtime.backend)

        self.assertEqual(factory.count, 1)

    def testProductionRuntimeSelectsGpuGeometryByDefault(self) -> None:
        factory = _SessionFactory()
        config = loadConfig(REPOSITORY_ROOT / "configs" / "RGBonly.yaml")
        geometry = SphericalGeometryImpl(config.geometry.boundarySamplesPerEdge)

        with patch("instatarget.app.driver.GpuGeometryImpl", return_value=geometry) as gpuGeometry:
            runtime = buildRuntime(config, hitSessionFactory=factory)
        closeBackend(runtime.backend)

        gpuGeometry.assert_called_once_with(
            boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge
        )
        self.assertIs(runtime.geometry, geometry)

    def testGeometryProfileIsRecordedIntoPerFrameProfilerStats(self) -> None:
        profiler = RuntimeProfiler(enabled=True)

        class _FakeGeometry:
            lastProfile = {
                "frameToDevice": 100,
                "gpuCrop": 250,
                "gpuGeometryTotal": 400,
                "imageRoundTrips": 0,
            }

        profiler.startFrame(1)
        _recordGeometryProfile(profiler, _FakeGeometry())
        profiler.finishFrame()

        self.assertIn("frameToDevice", profiler.stats)
        self.assertIn("gpuCrop", profiler.stats)
        self.assertIn("gpuGeometryTotal", profiler.stats)
        self.assertEqual(profiler.frameRows[0]["geometryBatches"][0]["imageRoundTrips"], 0)

    def testCudaDeviceAliasMatchesCurrentIndexedDevice(self) -> None:
        import torch

        class _Cuda:
            @staticmethod
            def current_device() -> int:
                return 0

        class _Torch:
            cuda = _Cuda()

        self.assertTrue(
            _devicesMatch(torch.device("cuda:0"), torch.device("cuda"), _Torch())
        )
        self.assertFalse(
            _devicesMatch(torch.device("cuda:1"), torch.device("cuda"), _Torch())
        )

    def testPipelinePrefetchPreservesOrderAndReportsActualWork(self) -> None:
        import numpy as np

        from instatarget.core.types import FrameIndex, FramePacket, SequenceId

        frames = [
            FramePacket(SequenceId("prefetch"), FrameIndex(index), index, np.zeros((2, 2, 3), dtype=np.uint8))
            for index in range(3)
        ]

        class _Source:
            def read(self):
                return frames.pop(0) if frames else None

        reader = _PrefetchReader(_Source())
        reader.start()
        try:
            returned = [reader.read(), reader.read(), reader.read(), reader.read()]
        finally:
            reader.close()

        self.assertEqual([int(frame.frameIndex) for frame in returned[:-1]], [0, 1, 2])
        self.assertIsNone(returned[-1])
        self.assertTrue(reader.lastProfile["pipelinePrefetchEnabled"])
        self.assertGreaterEqual(reader.lastProfile["pipelineDecodeNs"], 0)
        self.assertGreaterEqual(reader.lastProfile["pipelineQueueWaitNs"], 0)

    def testProductionBackendDoesNotRetainDeviceViews(self) -> None:
        import numpy as np
        import torch

        from instatarget.core.types import BFoV, LocalView, ViewSpec
        from instatarget.geometry import makeSphericalPoint
        from instatarget.tracker.backend import TrackerBackendImpl
        from instatarget.tracker.hit_backend import HiTBackend

        backend = TrackerBackendImpl(HiTBackend(_FakeHiTSession()))
        view = LocalView(
            ViewSpec(0, BFoV(makeSphericalPoint(0.0, 0.0), 1.0, 1.0), 256, 256),
            np.zeros((256, 256, 3), dtype=np.uint8),
            deviceRgb=torch.zeros((3, 256, 256), dtype=torch.float32),
        )
        backend.initialize(view, BBoxXYWH(80.0, 80.0, 40.0, 40.0))
        backend._rememberViews((view,), 1)

        self.assertEqual(tuple(backend._previousViews), (0,))
        retained = backend._previousViews[0]
        self.assertIsNone(retained.deviceRgb)
        self.assertIsNot(retained.rgb, view.rgb)
        self.assertFalse(retained.rgb.flags.writeable)
        backend.close()


class _SessionFactory:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, _config):
        self.count += 1
        return _FakeHiTSession()


class _FakeHiTSession:
    supportsOnlineTemplates = True

    def encodeTemplate(self, _rgb, bbox: BBoxXYWH) -> object:
        return bbox

    def infer(self, _rgb, templateFeatures) -> HiTPrediction:
        return HiTPrediction(templateFeatures[-1], 0.9, 0.9)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
