"""Pure scheduling helpers for optional cross-frame inference batches."""

from __future__ import annotations

from collections.abc import Sequence

from instatarget.core.config import SpeculativePipelineConfig
from instatarget.core.errors import ProtocolError
from instatarget.core.types import (
    InferenceRole,
    LocalView,
    RoutedInferenceTask,
    RoutedLocalObservation,
    SearchPlan,
    TaskKey,
)

_NORMAL_ROUND_VIEW_COUNT = 4


def buildPlanTasks(
    plan: SearchPlan,
    views: Sequence[LocalView],
    *,
    generation: int,
) -> tuple[RoutedInferenceTask, ...]:
    """Bind a formal SearchPlan to TaskKeys without inferring identity from batch position."""
    viewTuple = tuple(views)
    expectedViewIds = tuple(spec.viewId for spec in plan.views)
    actualViewIds = tuple(view.spec.viewId for view in viewTuple)
    if actualViewIds != expectedViewIds:
        raise ProtocolError("local views must preserve SearchPlan view order")
    if plan.attemptIndex == 0:
        role = InferenceRole.ROUND1_DIRECTION
    elif plan.attemptIndex == 1:
        role = InferenceRole.ROUND2_SHAPE
    else:
        raise ProtocolError("only round1 and round2 plans can become routed tasks")
    keys = tuple(
        TaskKey(
            sequenceId=plan.sequenceId,
            frameIndex=plan.frameIndex,
            attemptIndex=plan.attemptIndex,
            viewId=view.spec.viewId,
            generation=generation,
            role=role,
        )
        for view in viewTuple
    )
    return bindTaskViews(keys, viewTuple)


def bindTaskViews(
    keys: Sequence[TaskKey],
    views: Sequence[LocalView],
) -> tuple[RoutedInferenceTask, ...]:
    """Bind precomputed keys to crops after exact identity and order validation."""
    keyTuple = tuple(keys)
    viewTuple = tuple(views)
    if len(keyTuple) != len(viewTuple):
        raise ProtocolError("TaskKeys and LocalViews must have equal length")
    return tuple(
        RoutedInferenceTask(key, view)
        for key, view in zip(keyTuple, viewTuple, strict=True)
    )


def mergeRound2AndSpeculativeRound1(
    config: SpeculativePipelineConfig,
    round2: Sequence[RoutedInferenceTask],
    speculativeRound1: Sequence[RoutedInferenceTask],
) -> tuple[RoutedInferenceTask, ...]:
    """Build the documented ``[R2(t), R1(t+1)]`` batch without changing task identity."""
    if not (config.enabled and config.batchMergeEnabled):
        raise ProtocolError("speculative batch merge is disabled")
    if not round2 or not speculativeRound1:
        raise ProtocolError("batch merge requires both R2 and speculative R1 tasks")
    if (
        len(round2) != _NORMAL_ROUND_VIEW_COUNT
        or len(speculativeRound1) != _NORMAL_ROUND_VIEW_COUNT
    ):
        raise ProtocolError("speculative batch merge requires exactly four R2 and four R1 tasks")
    merged = tuple(round2) + tuple(speculativeRound1)
    if not merged:
        raise ProtocolError("cannot merge an empty inference batch")
    if any(task.key.role is not InferenceRole.ROUND2_SHAPE for task in round2):
        raise ProtocolError("round2 batch contains a non-round2 task")
    if any(task.key.attemptIndex != 1 for task in round2):
        raise ProtocolError("round2 tasks must use attemptIndex 1")
    if any(
        task.key.role is not InferenceRole.SPECULATIVE_ROUND1_DIRECTION
        for task in speculativeRound1
    ):
        raise ProtocolError("speculative batch contains a non-speculative task")
    if any(task.key.attemptIndex != 0 for task in speculativeRound1):
        raise ProtocolError("speculative R1 tasks must use attemptIndex 0")
    if round2 and speculativeRound1:
        round2Frame = int(round2[0].key.frameIndex)
        speculativeFrame = int(speculativeRound1[0].key.frameIndex)
        if speculativeFrame != round2Frame + 1:
            raise ProtocolError("speculative R1 must target the frame immediately after R2")
        if any(int(task.key.frameIndex) != round2Frame for task in round2):
            raise ProtocolError("all round2 tasks must target the same frame")
        if any(int(task.key.frameIndex) != speculativeFrame for task in speculativeRound1):
            raise ProtocolError("all speculative R1 tasks must target the same frame")
    keys = tuple(task.key for task in merged)
    if len(keys) != len(set(keys)):
        raise ProtocolError("merged inference batch contains duplicate TaskKeys")
    if len({key.sequenceId for key in keys}) != 1:
        raise ProtocolError("merged inference batch cannot cross sequence boundaries")
    if len({key.generation for key in keys}) != 1:
        raise ProtocolError("merged inference batch must use one generation")
    return merged


def validateRoutedBatch(
    requested: Sequence[TaskKey],
    returned: Sequence[TaskKey],
) -> None:
    """Validate identity before any output can reach a formal transaction."""
    requestedTuple = tuple(requested)
    returnedTuple = tuple(returned)
    if len(set(requestedTuple)) != len(requestedTuple):
        raise ProtocolError("routed batch request contains duplicate TaskKeys")
    if len(returnedTuple) != len(requestedTuple):
        raise ProtocolError("routed batch returned an unexpected number of outputs")
    if len(set(returnedTuple)) != len(returnedTuple):
        raise ProtocolError("routed batch returned duplicate TaskKeys")
    if set(returnedTuple) != set(requestedTuple):
        raise ProtocolError("routed batch returned unknown or missing TaskKeys")


def partitionMergedOutputs(
    requested: Sequence[RoutedInferenceTask],
    returned: Sequence[RoutedLocalObservation],
) -> tuple[tuple[RoutedLocalObservation, ...], tuple[RoutedLocalObservation, ...]]:
    """Restore request order, then split formal R2 and speculative R1 outputs."""
    requestedTuple = tuple(requested)
    expectedKeys = tuple(task.key for task in requestedTuple)
    returnedTuple = tuple(returned)
    validateRoutedBatch(expectedKeys, tuple(item.key for item in returnedTuple))
    byKey = {item.key: item for item in returnedTuple}
    ordered = tuple(byKey[key] for key in expectedKeys)
    formal = tuple(
        item for item in ordered if item.key.role is InferenceRole.ROUND2_SHAPE
    )
    speculative = tuple(
        item
        for item in ordered
        if item.key.role is InferenceRole.SPECULATIVE_ROUND1_DIRECTION
    )
    if len(formal) != _NORMAL_ROUND_VIEW_COUNT or len(speculative) != _NORMAL_ROUND_VIEW_COUNT:
        raise ProtocolError("merged output roles do not restore a four-plus-four batch")
    return formal, speculative


__all__ = [
    "bindTaskViews",
    "buildPlanTasks",
    "mergeRound2AndSpeculativeRound1",
    "partitionMergedOutputs",
    "validateRoutedBatch",
]
