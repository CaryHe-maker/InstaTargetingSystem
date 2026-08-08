import unittest

import numpy as np

from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    DepthPlane,
    FrameIndex,
    LocalView,
    TemplateCommand,
    TemplateCommandKind,
    ViewSpec,
)
from instatarget.geometry import makeSphericalPoint
from instatarget.tracker import (
    DepthEncoder,
    DepthPreprocessor,
    FusionHead,
    HiTBackend,
    HiTPrediction,
    TrackerBackendImpl,
)


class _RgbSession:
    supportsOnlineTemplates = True

    def encodeTemplate(self, rgb: np.ndarray, bbox: BBoxXYWH) -> object:
        return float(rgb.mean())

    def infer(self, rgb: np.ndarray, templateFeatures: tuple[object, ...]) -> HiTPrediction:
        return HiTPrediction(BBoxXYWH(1.0, 1.0, 3.0, 3.0), 0.8, 0.75)

    def close(self) -> None:
        return None


def _view(viewId: int, withDepth: bool = True) -> LocalView:
    spec = ViewSpec(
        viewId,
        BFoV(makeSphericalPoint(0.0, 0.0), 0.5, 0.5),
        6,
        6,
    )
    rgb = np.full((6, 6, 3), 100, dtype=np.uint8)
    if not withDepth:
        return LocalView(spec, rgb)
    values = np.full((6, 6), 8.0, dtype=np.float32)
    values[1:4, 1:4] = 3.0
    return LocalView(spec, rgb, DepthPlane(values, np.ones((6, 6), dtype=np.bool_), "m"))


def _keep(frameIndex: int, revision: int) -> TemplateCommand:
    return TemplateCommand(
        TemplateCommandKind.KEEP,
        FrameIndex(frameIndex),
        None,
        None,
        revision,
    )


class TrackerDepthTest(unittest.TestCase):
    def testPreprocessorMasksMissingValuesAndEnhancesContours(self) -> None:
        values = np.full((7, 7), 9.0, dtype=np.float32)
        values[2:5, 2:5] = 3.0
        mask = np.ones((7, 7), dtype=np.bool_)
        mask[0, 0] = False
        processor = DepthPreprocessor()

        result = processor.preprocess(DepthPlane(values, mask, "m"))

        self.assertEqual(result.depthRgb.dtype, np.uint8)
        self.assertEqual(result.depthRgb.shape, (7, 7, 3))
        self.assertTrue((result.depthRgb[0, 0] == 0).all())
        self.assertGreater(int(result.depthRgb[3, 3, 0]), int(result.depthRgb[1, 1, 0]))
        self.assertGreater(float(result.edge.max()), 0.0)

    def testSummariesEnforceMinimumValidRatio(self) -> None:
        processor = DepthPreprocessor(minValidRatio=0.5)
        values = np.ones((4, 4), dtype=np.float32)
        sparseMask = np.zeros((4, 4), dtype=np.bool_)
        sparseMask[0, 0] = True
        plane = DepthPlane(values, sparseMask, "m")

        self.assertIsNone(processor.summarizePlane(plane, BBoxXYWH(0, 0, 4, 4)))

    def testRgbDepthBackendProducesDepthAndFusedScores(self) -> None:
        backend = TrackerBackendImpl(
            HiTBackend(_RgbSession()),
            depthProcessor=DepthPreprocessor(),
            depthEncoder=DepthEncoder(),
            fusionHead=FusionHead(depthScoreWeight=0.2),
        )
        backend.initialize(_view(0), BBoxXYWH(1.0, 1.0, 3.0, 3.0))

        observation = backend.infer([_view(1)], _keep(1, 1))[0]

        self.assertIsNotNone(observation.depthSummary)
        self.assertGreater(observation.depthScore, 0.0)
        self.assertNotEqual(observation.fusedScore, observation.appearanceScore)

    def testRgbDepthBackendDegradesPerViewWhenDepthIsUnavailable(self) -> None:
        backend = TrackerBackendImpl(
            HiTBackend(_RgbSession()),
            depthProcessor=DepthPreprocessor(),
            depthEncoder=DepthEncoder(),
        )
        backend.initialize(_view(0), BBoxXYWH(1.0, 1.0, 3.0, 3.0))

        observation = backend.infer([_view(1, withDepth=False)], _keep(1, 1))[0]

        self.assertEqual(observation.depthScore, 0.0)
        self.assertEqual(observation.fusedScore, observation.appearanceScore)
        self.assertIsNone(observation.depthSummary)


if __name__ == "__main__":
    unittest.main()
