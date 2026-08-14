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
    DepthPreprocessor,
    HiTBackend,
    HiTPrediction,
    TrackerBackendImpl,
)


class _RgbSession:
    supportsOnlineTemplates = True

    def __init__(self) -> None:
        self.encoded: list[np.ndarray] = []
        self.inferred: list[np.ndarray] = []

    def encodeTemplate(self, rgb: np.ndarray, bbox: BBoxXYWH) -> object:
        self.encoded.append(rgb.copy())
        return float(rgb.mean())

    def infer(self, rgb: np.ndarray, templateFeatures: tuple[object, ...]) -> HiTPrediction:
        self.inferred.append(rgb.copy())
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

        self.assertEqual(result.edgeMask.dtype, np.bool_)
        self.assertEqual(result.edgeMask.shape, (7, 7))
        self.assertFalse(result.edgeMask[0, 0])
        self.assertTrue(result.edgeMask.any())
        self.assertGreater(float(result.edge.max()), 0.0)

        rgb = np.full((7, 7, 3), 100, dtype=np.uint8)
        enhanced = processor.enhanceRgb(rgb, DepthPlane(values, mask, "m"))
        self.assertTrue(np.array_equal(enhanced[~result.edgeMask], rgb[~result.edgeMask]))
        self.assertTrue(np.any(enhanced[result.edgeMask] != rgb[result.edgeMask]))

    def testSummariesEnforceMinimumValidRatio(self) -> None:
        processor = DepthPreprocessor(minValidRatio=0.5)
        values = np.ones((4, 4), dtype=np.float32)
        sparseMask = np.zeros((4, 4), dtype=np.bool_)
        sparseMask[0, 0] = True
        plane = DepthPlane(values, sparseMask, "m")

        self.assertIsNone(processor.summarizePlane(plane, BBoxXYWH(0, 0, 4, 4)))

    def testRgbDepthBackendSendsEnhancedRgbToOneHit(self) -> None:
        session = _RgbSession()
        template = _view(0)
        search = _view(1)
        processor = DepthPreprocessor()
        backend = TrackerBackendImpl(
            HiTBackend(session),
            depthProcessor=processor,
        )
        backend.initialize(template, BBoxXYWH(1.0, 1.0, 3.0, 3.0))

        observation = backend.infer([search], _keep(1, 1))[0]

        self.assertIsNotNone(observation.depthSummary)
        self.assertEqual(observation.depthScore, 0.0)
        self.assertEqual(observation.fusedScore, observation.appearanceScore)
        preparedSearch = backend.lastPreparedViews[0]
        expectedTemplate = processor.enhanceRgb(template.rgb, template.depth)
        expectedSearch = processor.enhanceRgb(search.rgb, search.depth)
        np.testing.assert_array_equal(session.encoded[0], expectedTemplate)
        np.testing.assert_array_equal(session.inferred[0], expectedSearch)
        np.testing.assert_array_equal(preparedSearch.rgb, expectedSearch)

    def testRgbDepthBackendDegradesPerViewWhenDepthIsUnavailable(self) -> None:
        backend = TrackerBackendImpl(
            HiTBackend(_RgbSession()),
            depthProcessor=DepthPreprocessor(),
        )
        backend.initialize(_view(0), BBoxXYWH(1.0, 1.0, 3.0, 3.0))

        observation = backend.infer([_view(1, withDepth=False)], _keep(1, 1))[0]

        self.assertEqual(observation.depthScore, 0.0)
        self.assertEqual(observation.fusedScore, observation.appearanceScore)
        self.assertIsNone(observation.depthSummary)


if __name__ == "__main__":
    unittest.main()
