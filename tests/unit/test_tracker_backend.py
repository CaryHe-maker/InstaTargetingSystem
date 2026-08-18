import unittest

import numpy as np

from instatarget.core.errors import ModelError, ProtocolError
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    LocalView,
    TemplateCommand,
    TemplateCommandKind,
    ViewSpec,
)
from instatarget.geometry import makeSphericalPoint
from instatarget.tracker import HiTBackend, HiTPrediction, TrackerBackendImpl


class FakeHiTSession:
    def __init__(self, supportsOnlineTemplates: bool = True) -> None:
        self.supportsOnlineTemplates = supportsOnlineTemplates
        self.encoded: list[tuple[int, BBoxXYWH]] = []
        self.inferred: list[tuple[int, tuple[object, ...]]] = []
        self.batchInferred: list[tuple[int, ...]] = []
        self.closeCount = 0
        self.returnInvalidBox = False

    def encodeTemplate(self, rgb: np.ndarray, bbox: BBoxXYWH) -> object:
        token = f"template-{len(self.encoded)}"
        self.encoded.append((int(rgb[0, 0, 0]), bbox))
        return token

    def infer(self, rgb: np.ndarray, templateFeatures: tuple[object, ...]) -> HiTPrediction:
        self.inferred.append((int(rgb[0, 0, 0]), tuple(templateFeatures)))
        bbox = (
            BBoxXYWH(xPx=10.0, yPx=10.0, widthPx=1.0, heightPx=1.0)
            if self.returnInvalidBox
            else BBoxXYWH(xPx=-1.0, yPx=1.0, widthPx=3.0, heightPx=2.0)
        )
        return HiTPrediction(
            bbox=bbox,
            modelScore=0.8,
            appearanceScore=0.7,
        )

    def inferBatch(
        self,
        rgbs: tuple[np.ndarray, ...],
        templateFeatures: tuple[object, ...],
    ) -> tuple[HiTPrediction, ...]:
        self.batchInferred.append(tuple(int(rgb[0, 0, 0]) for rgb in rgbs))
        return tuple(self.infer(rgb, templateFeatures) for rgb in rgbs)

    def close(self) -> None:
        self.closeCount += 1


def _view(viewId: int, marker: int) -> LocalView:
    spec = ViewSpec(
        viewId=viewId,
        bfov=BFoV(
            center=makeSphericalPoint(0.0, 0.0),
            horizontalFovRad=0.5,
            verticalFovRad=0.5,
        ),
        outputWidthPx=4,
        outputHeightPx=4,
    )
    rgb = np.full((4, 4, 3), marker, dtype=np.uint8)
    return LocalView(spec=spec, rgb=rgb)


def _command(
    kind: TemplateCommandKind,
    frameIndex: int,
    revision: int,
    viewId: int | None = None,
) -> TemplateCommand:
    localBox = (
        BBoxXYWH(xPx=1.0, yPx=1.0, widthPx=2.0, heightPx=2.0)
        if viewId is not None
        else None
    )
    return TemplateCommand(
        kind=kind,
        frameIndex=FrameIndex(frameIndex),
        viewId=viewId,
        localBox=localBox,
        expectedRevision=revision,
    )


class TrackerBackendTest(unittest.TestCase):
    def testRgbOnlyLifecycleOrderAndStableObservation(self) -> None:
        session = FakeHiTSession()
        backend = TrackerBackendImpl(HiTBackend(session))
        template = _view(0, 10)
        search = _view(3, 20)

        with self.assertRaises(ProtocolError):
            backend.infer([search], _command(TemplateCommandKind.KEEP, 1, 1))

        backend.initialize(template, BBoxXYWH(xPx=1.0, yPx=1.0, widthPx=2.0, heightPx=2.0))
        with self.assertRaises(ProtocolError):
            backend.initialize(template, BBoxXYWH(xPx=1.0, yPx=1.0, widthPx=2.0, heightPx=2.0))

        observations = backend.infer([search], _command(TemplateCommandKind.KEEP, 1, 1))
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].viewId, 3)
        self.assertEqual(
            observations[0].bbox,
            BBoxXYWH(xPx=0.0, yPx=1.0, widthPx=2.0, heightPx=2.0),
        )
        self.assertEqual(observations[0].fusedScore, observations[0].appearanceScore)
        self.assertEqual(session.inferred[0][1], ("template-0",))

    def testSameRoundViewsUseOneBatchCall(self) -> None:
        session = FakeHiTSession()
        backend = TrackerBackendImpl(HiTBackend(session))
        backend.initialize(
            _view(0, 10),
            BBoxXYWH(xPx=0.0, yPx=0.0, widthPx=2.0, heightPx=2.0),
        )

        observations = backend.infer(
            [_view(1, 20), _view(2, 30), _view(3, 40), _view(4, 50)],
            _command(TemplateCommandKind.KEEP, 1, 1),
        )

        self.assertEqual(session.batchInferred, [(20, 30, 40, 50)])
        self.assertEqual([item.viewId for item in observations], [1, 2, 3, 4])

    def testTemplateCommandsUsePreviousViewsAndRevisionIsAtomic(self) -> None:
        session = FakeHiTSession()
        backend = TrackerBackendImpl(HiTBackend(session))
        backend.initialize(_view(0, 10), BBoxXYWH(xPx=0.0, yPx=0.0, widthPx=2.0, heightPx=2.0))
        backend.infer([_view(1, 20)], _command(TemplateCommandKind.KEEP, 1, 1))

        backend.infer(
            [_view(2, 30)],
            _command(TemplateCommandKind.UPDATE_RECENT, 2, 2, viewId=1),
        )
        self.assertEqual(backend.templateRevision, 2)
        self.assertEqual(session.encoded[-1][0], 20)
        self.assertEqual(session.inferred[-1][1], ("template-0",))

        with self.assertRaises(ProtocolError):
            backend.infer([], _command(TemplateCommandKind.KEEP, 3, 2))
        self.assertEqual(backend.templateRevision, 2)

        backend.infer([], _command(TemplateCommandKind.RESET_TO_ANCHOR, 3, 3))
        self.assertEqual(backend.templateRevision, 3)

    def testTemplateUpdateCanUseFirstRoundViewAfterSecondRound(self) -> None:
        session = FakeHiTSession()
        backend = TrackerBackendImpl(HiTBackend(session))
        backend.initialize(
            _view(0, 10),
            BBoxXYWH(xPx=0.0, yPx=0.0, widthPx=2.0, heightPx=2.0),
        )
        backend.infer(
            [_view(1, 20)],
            _command(TemplateCommandKind.KEEP, 1, 1),
        )
        backend.infer(
            [_view(4, 40)],
            _command(TemplateCommandKind.KEEP, 1, 2),
        )

        backend.infer(
            [],
            _command(TemplateCommandKind.UPDATE_RECENT, 2, 3, viewId=1),
        )

        self.assertEqual(backend.templateRevision, 3)
        self.assertEqual(session.encoded[-1][0], 20)

    def testUnsupportedOnlineTemplatesRejectUpdates(self) -> None:
        session = FakeHiTSession(supportsOnlineTemplates=False)
        backend = TrackerBackendImpl(HiTBackend(session))
        backend.initialize(_view(0, 10), BBoxXYWH(xPx=0.0, yPx=0.0, widthPx=2.0, heightPx=2.0))
        with self.assertRaises(ProtocolError):
            backend.infer([], _command(TemplateCommandKind.UPDATE_STABLE, 1, 1, viewId=0))
        self.assertEqual(backend.templateRevision, 0)

    def testInvalidPredictionBoxAndCloseAreHandled(self) -> None:
        session = FakeHiTSession()
        backend = TrackerBackendImpl(HiTBackend(session))
        backend.initialize(_view(0, 10), BBoxXYWH(xPx=0.0, yPx=0.0, widthPx=2.0, heightPx=2.0))
        session.returnInvalidBox = True
        with self.assertRaises(ModelError):
            backend.infer([_view(0, 20)], _command(TemplateCommandKind.KEEP, 1, 1))
        backend.close()
        backend.close()
        self.assertEqual(session.closeCount, 1)
        with self.assertRaises(ProtocolError):
            backend.infer([], _command(TemplateCommandKind.KEEP, 2, 2))


if __name__ == "__main__":
    unittest.main()
