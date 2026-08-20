import unittest

import numpy as np

from instatarget.app.speculative_scheduler import (
    bindTaskViews,
    mergeRound2AndSpeculativeRound1,
    partitionMergedOutputs,
    validateRoutedBatch,
)
from instatarget.controller.speculative_pipeline import (
    RollbackReason,
    SpeculativePipeline,
    evaluateSpeculation,
)
from instatarget.core.config import SpeculativePipelineConfig
from instatarget.core.errors import ProtocolError
from instatarget.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    InferenceRole,
    LocalObservation,
    LocalView,
    RoutedInferenceTask,
    RoutedLocalObservation,
    SequenceId,
    TaskKey,
    TrackResult,
    TrackStatus,
    ViewSpec,
)
from instatarget.geometry import makeSphericalPoint


def _config(**changes) -> SpeculativePipelineConfig:
    values = dict(
        enabled=True,
        batchMergeEnabled=True,
        maxRollbackRate=0.20,
        centerGapRatio=0.50,
        logScaleGap=0.25,
        minimumDirectionConfidence=0.80,
        maxSpeculativeAgeFrames=1,
    )
    values.update(changes)
    return SpeculativePipelineConfig(**values)


def _view(viewId: int = 0, yawRad: float = 0.0) -> ViewSpec:
    return ViewSpec(
        viewId=viewId,
        bfov=BFoV(makeSphericalPoint(yawRad, 0.0), 1.2, 1.2),
        outputWidthPx=4,
        outputHeightPx=4,
    )


def _task(frameIndex: int, role: InferenceRole, viewId: int, generation: int = 1):
    view = _view(viewId)
    key = TaskKey(
        SequenceId("sequence"),
        FrameIndex(frameIndex),
        1 if role is InferenceRole.ROUND2_SHAPE else 0,
        viewId,
        generation,
        role,
    )
    return RoutedInferenceTask(key, LocalView(view, np.zeros((4, 4, 3), dtype=np.uint8)))


def _output(task: RoutedInferenceTask) -> RoutedLocalObservation:
    observation = LocalObservation(
        viewId=task.key.viewId,
        bbox=BBoxXYWH(1.0, 1.0, 2.0, 2.0),
        modelScore=0.9,
        appearanceScore=0.9,
        fusedScore=0.9,
        latencyNs=1,
    )
    return RoutedLocalObservation(task.key, observation)


def _committed(
    status: TrackStatus = TrackStatus.TRACKING,
    *,
    yawRad: float = 0.0,
    horizontalSizeRad: float = 0.4,
    verticalSizeRad: float = 0.3,
) -> TrackResult:
    bfov = BFoV(
        makeSphericalPoint(yawRad, 0.0),
        horizontalSizeRad,
        verticalSizeRad,
    )
    return TrackResult(
        SequenceId("sequence"),
        FrameIndex(1),
        BBoxXYWH(1.0, 1.0, 2.0, 2.0),
        bfov,
        0.9,
        status,
        True,
    )


def _routedStateOutput(state) -> tuple[RoutedLocalObservation, ...]:
    task = RoutedInferenceTask(
        state.taskKeys[0],
        LocalView(_view(), np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    return (_output(task),)


def _createPending(pipeline: SpeculativePipeline):
    return pipeline.create(
        sequenceId=SequenceId("sequence"),
        frameIndex=FrameIndex(2),
        directionCenter=makeSphericalPoint(0.0, 0.0),
        horizontalSizeRad=0.4,
        verticalSizeRad=0.3,
        motionUncertaintyRad=0.05,
        directionConfidence=0.9,
        sourceStateRevision=1,
        views=(_view(),),
    )


class SpeculativePipelineTest(unittest.TestCase):
    def testDefaultConfigurationDisablesCreation(self) -> None:
        pipeline = SpeculativePipeline(SpeculativePipelineConfig())
        with self.assertRaises(ProtocolError):
            pipeline.create(
                sequenceId=SequenceId("sequence"),
                frameIndex=FrameIndex(2),
                directionCenter=makeSphericalPoint(0.0, 0.0),
                horizontalSizeRad=0.4,
                verticalSizeRad=0.3,
                motionUncertaintyRad=0.05,
                directionConfidence=0.9,
                sourceStateRevision=1,
                views=(_view(),),
            )

    def testAcceptsMatchingGenerationRevisionAndCoverage(self) -> None:
        pipeline = SpeculativePipeline(_config())
        state = _createPending(pipeline)
        decision = pipeline.evaluate(
            committedResult=_committed(),
            formalStateRevision=1,
            routedObservations=_routedStateOutput(state),
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(pipeline.summary().acceptanceRate, 1.0)

    def testGenerationMismatchAndLostInvalidateWithoutFormalWrite(self) -> None:
        pipeline = SpeculativePipeline(_config())
        state = _createPending(pipeline)
        decision = evaluateSpeculation(
            config=_config(),
            state=state,
            committedResult=_committed(),
            formalStateRevision=1,
            currentGeneration=2,
            routedObservations=_routedStateOutput(state),
        )
        self.assertEqual(decision.rollbackReason, RollbackReason.GENERATION_MISMATCH)
        self.assertFalse(decision.accepted)

        lostPipeline = SpeculativePipeline(_config())
        lostState = _createPending(lostPipeline)
        lostDecision = lostPipeline.evaluate(
            committedResult=_committed(TrackStatus.LOST),
            formalStateRevision=1,
            routedObservations=_routedStateOutput(lostState),
        )
        self.assertEqual(lostDecision.rollbackReason, RollbackReason.EXPLICIT_LOST)

    def testRevisionCenterScaleCoverageAndCloseRollbackPaths(self) -> None:
        cases = (
            (2, _committed(), (_view(),), RollbackReason.REVISION_MISMATCH),
            (1, _committed(yawRad=0.7), (_view(),), RollbackReason.CENTER_GAP),
            (
                1,
                _committed(horizontalSizeRad=0.8, verticalSizeRad=0.6),
                (_view(),),
                RollbackReason.SCALE_GAP,
            ),
            (1, _committed(), (_view(yawRad=2.0),), RollbackReason.COVERAGE),
        )
        for formalRevision, committed, views, expectedReason in cases:
            with self.subTest(reason=expectedReason):
                pipeline = SpeculativePipeline(_config())
                state = pipeline.create(
                    sequenceId=SequenceId("sequence"),
                    frameIndex=FrameIndex(2),
                    directionCenter=makeSphericalPoint(0.0, 0.0),
                    horizontalSizeRad=0.4,
                    verticalSizeRad=0.3,
                    motionUncertaintyRad=0.05,
                    directionConfidence=0.9,
                    sourceStateRevision=1,
                    views=views,
                )
                decision = pipeline.evaluate(
                    committedResult=committed,
                    formalStateRevision=formalRevision,
                    routedObservations=_routedStateOutput(state),
                )
                self.assertEqual(decision.rollbackReason, expectedReason)

        pipeline = SpeculativePipeline(_config())
        _createPending(pipeline)
        pipeline.closeSequence(SequenceId("sequence"))
        self.assertIsNone(pipeline.pending)
        with self.assertRaises(ProtocolError):
            pipeline.create(
                sequenceId=SequenceId("sequence"),
                frameIndex=FrameIndex(2),
                directionCenter=makeSphericalPoint(0.0, 0.0),
                horizontalSizeRad=0.4,
                verticalSizeRad=0.3,
                motionUncertaintyRad=0.05,
                directionConfidence=0.9,
                sourceStateRevision=1,
                views=(_view(),),
            )

    def testEmptyAndNonfiniteOutputsRollbackExplicitly(self) -> None:
        emptyPipeline = SpeculativePipeline(_config())
        _createPending(emptyPipeline)
        emptyDecision = emptyPipeline.evaluate(
            committedResult=_committed(),
            formalStateRevision=1,
            routedObservations=(),
        )
        self.assertEqual(emptyDecision.rollbackReason, RollbackReason.EMPTY_OUTPUT)

        nonfinitePipeline = SpeculativePipeline(_config())
        state = _createPending(nonfinitePipeline)
        routed = _routedStateOutput(state)
        object.__setattr__(routed[0].observation, "presenceProbability", float("nan"))
        decision = nonfinitePipeline.evaluate(
            committedResult=_committed(),
            formalStateRevision=1,
            routedObservations=routed,
        )
        self.assertEqual(decision.rollbackReason, RollbackReason.NONFINITE_OUTPUT)

    def testBatchMergeAndRoutingUseTaskKeys(self) -> None:
        round2 = tuple(_task(1, InferenceRole.ROUND2_SHAPE, viewId) for viewId in range(4, 8))
        speculative = tuple(
            _task(2, InferenceRole.SPECULATIVE_ROUND1_DIRECTION, viewId)
            for viewId in range(4)
        )
        merged = mergeRound2AndSpeculativeRound1(_config(), round2, speculative)
        self.assertEqual(
            [item.key.role for item in merged],
            [InferenceRole.ROUND2_SHAPE] * 4
            + [InferenceRole.SPECULATIVE_ROUND1_DIRECTION] * 4,
        )
        validateRoutedBatch(
            tuple(item.key for item in merged),
            tuple(item.key for item in reversed(merged)),
        )
        formal, routedSpeculative = partitionMergedOutputs(
            merged,
            tuple(_output(item) for item in reversed(merged)),
        )
        self.assertEqual(tuple(item.key for item in formal), tuple(item.key for item in round2))
        self.assertEqual(
            tuple(item.key for item in routedSpeculative),
            tuple(item.key for item in speculative),
        )
        self.assertEqual(
            bindTaskViews((speculative[0].key,), (speculative[0].view,)),
            (speculative[0],),
        )
        with self.assertRaises(ProtocolError):
            mergeRound2AndSpeculativeRound1(
                _config(),
                round2,
                tuple(
                    _task(3, InferenceRole.SPECULATIVE_ROUND1_DIRECTION, viewId)
                    for viewId in range(4)
                ),
            )
        with self.assertRaises(ProtocolError):
            mergeRound2AndSpeculativeRound1(_config(), round2[:3], speculative)
        with self.assertRaises(ProtocolError):
            validateRoutedBatch(
                (round2[0].key, round2[0].key),
                (round2[0].key, round2[0].key),
            )


if __name__ == "__main__":
    unittest.main()
