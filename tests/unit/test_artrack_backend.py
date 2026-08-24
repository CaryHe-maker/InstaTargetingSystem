import unittest

import numpy as np

from instatarget.core.types import BFoV, BBoxXYWH, LocalView, SphericalPoint, TemplateCommand, TemplateCommandKind, ViewSpec
from instatarget.tracker import ARTrackBackend, ARTrackPrediction, TrackerBackendImpl
from instatarget.tracker.artrack_model import ARTrackTemplate


class _FakeARTrackSession:
    supportsOnlineTemplates = True

    def encodeTemplate(self, rgb, bbox):
        return ARTrackTemplate(rgb, bbox)

    def infer(self, rgb, templateFeatures):
        return ARTrackPrediction(BBoxXYWH(8.0, 9.0, 20.0, 18.0), 0.8, 0.8, 0.8)

    def close(self):
        return None


class ARTrackBackendTest(unittest.TestCase):
    def test_local_prediction_preserves_tracker_contract(self):
        point = SphericalPoint(1.0, 0.0, 0.0, 0.0, 0.0)
        spec = ViewSpec(0, BFoV(point, 1.0, 1.0), 64, 64)
        view = LocalView(spec, np.zeros((64, 64, 3), dtype=np.uint8))
        backend = TrackerBackendImpl(ARTrackBackend(_FakeARTrackSession()))
        backend.initialize(view, BBoxXYWH(20.0, 20.0, 16.0, 16.0))
        observations = backend.infer(
            (view,), TemplateCommand(TemplateCommandKind.KEEP, 1, None, None, 1)
        )
        self.assertEqual(observations[0].bbox.widthPx, 20.0)
        self.assertEqual(observations[0].predictedIoU, 0.8)
        backend.close()


if __name__ == "__main__":
    unittest.main()
