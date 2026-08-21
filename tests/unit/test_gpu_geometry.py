import unittest

import numpy as np

from instatarget.core.types import BFoV, FrameIndex, FramePacket, SequenceId, ViewSpec
from instatarget.geometry import SphericalGeometryImpl, makeSphericalPoint
from instatarget.geometry.gpu_geometry import GpuGeometryImpl


class GpuGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for GPU Geometry regression")

    def testPerspectiveCropsMatchCpuAndStayOnDevice(self) -> None:
        height, width = 96, 192
        x = np.arange(width, dtype=np.uint16)[None, :]
        y = np.arange(height, dtype=np.uint16)[:, None]
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = (x * 3 + y * 5) % 256
        rgb[..., 1] = (x * 7 + y * 2) % 256
        rgb[..., 2] = (x * 11 + y * 13) % 256
        frame = FramePacket(SequenceId("gpu-regression"), FrameIndex(0), 0, rgb)
        specs = (
            ViewSpec(0, BFoV(makeSphericalPoint(0.0, 0.0), 1.2, 1.0), 256, 256),
            ViewSpec(1, BFoV(makeSphericalPoint(3.12, 0.1), 1.4, 1.1), 256, 256),
            ViewSpec(2, BFoV(makeSphericalPoint(-1.0, 1.25), 1.0, 0.6), 256, 256),
        )
        cpu = SphericalGeometryImpl().cropViews(frame, specs)
        gpuGeometry = GpuGeometryImpl()
        gpu = gpuGeometry.cropViews(frame, specs)
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]

        for cpuView, gpuView in zip(cpu, gpu, strict=True):
            expected = (cpuView.rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 - mean) / std
            actual = gpuView.deviceRgb.detach().cpu().numpy()
            self.assertEqual(str(gpuView.deviceRgb.device), "cuda:0")
            self.assertEqual(str(gpuView.deviceRgb.dtype), "torch.float32")
            reconstructed = np.rint((actual * std + mean) * 255.0).clip(0, 255)
            pixelError = np.abs(reconstructed - cpuView.rgb.transpose(2, 0, 1))
            self.assertLessEqual(float(pixelError.max()), 1.0)
            self.assertEqual(float(np.percentile(pixelError, 99)), 0.0)

        self.assertEqual(gpuGeometry.lastProfile["imageRoundTrips"], 0)
        self.assertEqual(gpuGeometry.lastProfile["gridCpuTransfers"], 0)
        self.assertEqual(gpuGeometry.lastProfile["frameTensorDevice"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
