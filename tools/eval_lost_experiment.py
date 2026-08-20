"""Evaluate isolated LOST-state policies without changing production modules."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, replace
from math import asin, pi, sqrt
from pathlib import Path
from time import perf_counter_ns
from types import MethodType
from typing import Any

import numpy as np
from eval_manifest_controller import (
    CandidateRecorder,
    ManifestVideoSource,
    _experimentMetadata,
    _summarize,
)

from instatarget.app.driver import _projectObservation, buildRuntime, closeBackend
from instatarget.controller.recovery_planner import (
    PlannedView,
    RecoveryPlanner,
    _motionCenter,
    _offsetDirection,
    _trackingSize,
)
from instatarget.controller.state_model import (
    AttemptKind,
    AttemptRecord,
    RecoveryMemory,
    TrackMode,
    TransitionDecision,
    TransitionReason,
)
from instatarget.core.config import loadConfig
from instatarget.core.errors import ProtocolError
from instatarget.core.protocols import FrameCommitted, MoreViewsRequired
from instatarget.core.types import FramePacket, TrackStatus
from instatarget.eval.otb_metrics import circularBBoxIoU, trackingLossRate
from instatarget.eval.profiler import RuntimeProfiler
from instatarget.training.dataset import ManifestRecord, loadManifest

POLICIES = ("immediate_q80", "rollback_q90", "hysteresis_q90")
VIEW_STRATEGIES = ("cube6_type1", "dual_cube12", "cube6_adaptive_type1")


@dataclass(frozen=True, slots=True)
class ControllerCheckpoint:
    state: dict[str, Any]
    backendRevision: int


class ExperimentalScoreGroup:
    """Ten-score history with an experiment-selected LOST order statistic."""

    def __init__(self, lostRank: int) -> None:
        if lostRank not in (8, 9):
            raise ValueError("lostRank must be 8 or 9")
        self.values: deque[float] = deque(maxlen=10)
        self._lostRank = lostRank

    def append(self, score: float) -> None:
        value = float(score)
        if not 0.0 <= value <= 1.0:
            raise ProtocolError("StateScore must be in [0, 1]")
        self.values.append(value)

    def thresholds(self) -> tuple[float, float] | None:
        if len(self.values) < 3:
            return None
        if len(self.values) < 10:
            maximum = max(self.values)
            minimum = min(self.values)
            uncertain = 0.5 * maximum + 0.5 * minimum
            minimumWeight = 0.8 if self._lostRank == 8 else 0.9
            lost = (1.0 - minimumWeight) * maximum + minimumWeight * minimum
            return uncertain, lost
        ordered = sorted(self.values, reverse=True)
        return ordered[4], ordered[self._lostRank - 1]


class ExperimentalStateMachine:
    """State policy used only by this evaluation tool."""

    def __init__(self, trackingConfig: Any, policy: str) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown LOST policy: {policy}")
        self._config = trackingConfig
        self._policy = policy
        self._scoreGroup = ExperimentalScoreGroup(8 if policy == "immediate_q80" else 9)
        self._lowRun = 0
        self._lowVotes: deque[bool] = deque(maxlen=3)
        self._recoveryHighRun = 0
        self._suppressedTransitions = 0
        self._rollbackRequested = False

    @property
    def scoreGroup(self) -> ExperimentalScoreGroup:
        return self._scoreGroup

    def initialize(self) -> Any:
        self._scoreGroup = ExperimentalScoreGroup(
            8 if self._policy == "immediate_q80" else 9
        )
        self._lowRun = 0
        self._lowVotes.clear()
        self._recoveryHighRun = 0
        self._suppressedTransitions = 0
        self._rollbackRequested = False
        return None

    def suppressLostEntry(self, frames: int) -> None:
        self._suppressedTransitions = max(0, int(frames))
        self._lowRun = 0
        self._lowVotes.clear()
        self._rollbackRequested = False

    def consumeRollbackRequest(self) -> bool:
        requested = self._rollbackRequested
        self._rollbackRequested = False
        return requested

    def transition(
        self,
        mode: TrackMode,
        stateScore: float,
        *,
        measurementAccepted: bool,
    ) -> TransitionDecision:
        score = float(stateScore)
        if not 0.0 <= score <= 1.0:
            raise ProtocolError("StateScore must be in [0, 1]")
        nextMode, reason = self._select(mode, score, measurementAccepted)
        reset = mode is TrackMode.LOST and measurementAccepted
        return TransitionDecision(
            "COMMIT",
            nextMode,
            reason,
            measurementAccepted,
            resetMotionHistory=reset,
            resetRecoveryEpoch=reset,
        )

    def _select(
        self,
        mode: TrackMode,
        score: float,
        measurementAccepted: bool,
    ) -> tuple[TrackMode, TransitionReason]:
        if mode is TrackMode.INIT or len(self._scoreGroup.values) < 2:
            return TrackMode.TRACKING, TransitionReason.RELIABLE_MEASUREMENT
        if len(self._scoreGroup.values) == 2:
            if score > self._scoreGroup.values[-1]:
                return TrackMode.TRACKING, TransitionReason.RELIABLE_MEASUREMENT
            return TrackMode.UNCERTAIN, TransitionReason.WEAK_MEASUREMENT

        thresholds = self._scoreGroup.thresholds()
        assert thresholds is not None
        uncertainThreshold, lostThreshold = thresholds
        lostEntrySuppressed = self._suppressedTransitions > 0
        if lostEntrySuppressed:
            self._suppressedTransitions -= 1

        if self._policy == "hysteresis_q90" and mode is TrackMode.LOST:
            strong = score >= uncertainThreshold and measurementAccepted
            self._recoveryHighRun = self._recoveryHighRun + 1 if strong else 0
            if self._recoveryHighRun >= 2:
                self._recoveryHighRun = 0
                self._lowVotes.clear()
                return TrackMode.TRACKING, TransitionReason.REACQUIRED
            return TrackMode.LOST, TransitionReason.RECOVERY_PROGRESS

        if score >= uncertainThreshold:
            self._lowRun = 0
            self._lowVotes.clear()
            return TrackMode.TRACKING, TransitionReason.RELIABLE_MEASUREMENT
        if score >= lostThreshold:
            self._lowRun = 0
            self._lowVotes.append(False)
            return TrackMode.UNCERTAIN, TransitionReason.WEAK_MEASUREMENT

        if lostEntrySuppressed:
            self._lowRun = 0
            self._lowVotes.clear()
            return TrackMode.UNCERTAIN, TransitionReason.HARD_MISS

        if self._policy == "immediate_q80":
            return TrackMode.LOST, TransitionReason.HARD_MISS
        if self._policy == "rollback_q90":
            self._lowRun += 1
            if self._lowRun >= 2:
                self._lowRun = 0
                self._rollbackRequested = True
                return TrackMode.LOST, TransitionReason.PATIENCE_EXHAUSTED
            return TrackMode.UNCERTAIN, TransitionReason.HARD_MISS

        self._lowVotes.append(True)
        if sum(self._lowVotes) >= 2:
            self._lowVotes.clear()
            return TrackMode.LOST, TransitionReason.PATIENCE_EXHAUSTED
        return TrackMode.UNCERTAIN, TransitionReason.HARD_MISS

    def recordScore(self, stateScore: float) -> None:
        self._scoreGroup.append(stateScore)


class ExperimentalRecoveryPlanner(RecoveryPlanner):
    """LOST-only view alternatives; other states delegate to production planning."""

    def __init__(
        self,
        geometryConfig: Any,
        trackingConfig: Any,
        recoveryConfig: Any,
        strategy: str,
    ) -> None:
        super().__init__(geometryConfig, trackingConfig, recoveryConfig)
        if strategy not in VIEW_STRATEGIES:
            raise ValueError(f"unknown LOST view strategy: {strategy}")
        self.strategy = strategy
        self.refinementSize: tuple[float, float] | None = None

    @property
    def usesSecondRound(self) -> bool:
        return self.strategy != "dual_cube12"

    def buildViews(self, *args: Any, **kwargs: Any) -> tuple[PlannedView, ...]:
        status = args[7] if len(args) > 7 else kwargs["status"]
        if status is not TrackStatus.LOST:
            return super().buildViews(*args, **kwargs)

        fallbackBfov = args[5]
        predictedMotion = args[6]
        searchSeed = kwargs.get("searchSeedCenter")
        attemptIndex = int(kwargs.get("attemptIndex", 0))
        viewIdStart = int(kwargs.get("viewIdStart", 0))
        viewBudget = kwargs.get("viewBudget")
        budget = self._tracking.maxViewsPerFrameTotal if viewBudget is None else int(viewBudget)
        predictedCenter = (
            _motionCenter(predictedMotion) if predictedMotion is not None else fallbackBfov.center
        )

        if self.strategy == "dual_cube12":
            if attemptIndex != 0 or budget < 12:
                raise ProtocolError("dual cubemap LOST strategy requires one 12-view attempt")
            complement = _offsetDirection(
                predictedCenter,
                pi / 4.0,
                asin(1.0 / sqrt(3.0)),
            )
            return self._cubeMap(
                predictedCenter, viewIdStart, 0, rolePrefix="cubemap_primary"
            ) + self._cubeMap(
                complement, viewIdStart + 6, 0, rolePrefix="cubemap_complement"
            )

        if attemptIndex == 0:
            if budget < 6:
                raise ProtocolError("LOST cubemap first round requires six views")
            self.refinementSize = None
            return self._cubeMap(predictedCenter, viewIdStart, 0, rolePrefix="cubemap")
        if attemptIndex != 1 or budget < 4:
            raise ProtocolError("LOST Type1 refinement requires four second-round views")
        center = searchSeed or predictedCenter
        size = self.refinementSize or _trackingSize(predictedMotion, fallbackBfov)
        return self._fourCorners(
            center,
            viewIdStart,
            1,
            dynamicSize=size,
            forceMaxFov=self.strategy == "cube6_type1",
        )


class ExperimentCandidateRecorder(CandidateRecorder):
    def __init__(self, truth: dict[int, ManifestRecord]) -> None:
        super().__init__(truth)
        self.executedViewCounts: Counter[int] = Counter()
        self.executedForwardCounts: Counter[int] = Counter()

    def recordLocalRgb(self, frame: FramePacket, views: Any) -> None:
        self.executedViewCounts[int(frame.frameIndex)] += len(views)
        self.executedForwardCounts[int(frame.frameIndex)] += 1
        super().recordLocalRgb(frame, views)

    def resetFinalFrame(self, frameIndex: int) -> None:
        self.rows = [row for row in self.rows if int(row["frameIndex"]) != frameIndex]
        self.viewCounts[frameIndex] = 0
        self.forwardCounts[frameIndex] = 0
        for mapping in (self._local, self._views, self._rounds):
            for key in [key for key in mapping if int(key[0]) == frameIndex]:
                del mapping[key]


class SyntheticTimer:
    def __init__(self, intervalsNs: list[int]) -> None:
        self.intervalsNs = intervalsNs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"E:\NewDownload\train"))
    parser.add_argument("--split", choices=("validation",), required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--view-strategy", choices=VIEW_STRATEGIES, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--spherical-samples-yaw", type=int, default=128)
    parser.add_argument("--spherical-samples-pitch", type=int, default=64)
    parser.add_argument("--progress-interval-seconds", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(
        profile=False,
        cudnn_benchmark=False,
        channels_last=False,
        reuse_buffers=False,
        pinned_nonblocking=False,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.progress_interval_seconds < 0.0:
        raise ValueError("progress interval must be non-negative")
    os.environ["INSTARGET_PROFILE"] = "0"
    datasetRoot = args.dataset_root.expanduser().resolve()
    manifestPath = args.manifest.expanduser().resolve()
    if not manifestPath.is_relative_to(datasetRoot):
        raise RuntimeError(f"manifest must be inside canonical root: {datasetRoot}")
    records = tuple(
        sorted(
            (
                record
                for record in loadManifest(manifestPath)
                if record.split == args.split and record.sequenceId == args.sequence
            ),
            key=lambda item: item.frameIndex,
        )
    )
    if not records or records[0].frameIndex != 0 or records[0].bbox is None:
        raise RuntimeError("selected sequence requires a visible frame-0 initialization")
    selected = records[: args.max_frames] if args.max_frames is not None else records
    if len(selected) < 2:
        raise RuntimeError("LOST experiment requires at least two frames")

    appConfig = loadConfig(args.config)
    appConfig = replace(
        appConfig,
        model=replace(appConfig.model, weights=args.weights.resolve(), precision="fp32"),
        visualization=replace(appConfig.visualization, enabled=False),
    )
    runtime = buildRuntime(appConfig)
    controller = runtime.controller
    machine = ExperimentalStateMachine(appConfig.tracking, args.policy)
    planner = ExperimentalRecoveryPlanner(
        appConfig.geometry,
        appConfig.tracking,
        appConfig.recovery,
        args.view_strategy,
    )
    controller._stateMachine = machine
    controller._planner = planner
    controller._consume = MethodType(_experimentalConsume, controller)
    controller._shouldEscalate = MethodType(_experimentalShouldEscalate, controller)

    truth = {record.frameIndex: record for record in selected}
    source = ManifestVideoSource(records, args.max_frames)
    recorder = ExperimentCandidateRecorder(truth)
    profiler = RuntimeProfiler(enabled=False)
    source.open("")
    results: dict[int, Any] = {}
    evaluatedModes: dict[int, TrackMode] = {}
    timings: Counter[int] = Counter()
    checkpoints: dict[int, ControllerCheckpoint] = {}
    frameCache: dict[int, FramePacket] = {}
    rollbackEvents = 0
    replayedExecutions = 0
    rollbackTargets: list[int] = []
    initializedNs = perf_counter_ns()
    progressIntervalNs = int(args.progress_interval_seconds * 1_000_000_000.0)
    nextProgressNs = initializedNs + progressIntervalNs
    try:
        frame0 = source.read()
        if frame0 is None:
            raise RuntimeError("sequence is empty")
        initPlan = controller.buildInitialization(frame0, selected[0].bbox)
        templateView = runtime.geometry.cropViews(frame0, [initPlan.templateView])[0]
        runtime.backend.initialize(templateView, initPlan.templateBox)
        results[0] = controller.commitInitialization(initPlan)
        recorder.recordLocalRgb(frame0, [templateView])
        timings[0] += perf_counter_ns() - initializedNs

        while True:
            frame = source.read()
            if frame is None:
                break
            frameIndex = int(frame.frameIndex)
            frameCache[frameIndex] = frame
            checkpoints[frameIndex] = _checkpoint(controller, runtime.backend)
            result, elapsed, evaluatedMode = _processFrame(
                frame,
                controller,
                runtime,
                recorder,
            )
            results[frameIndex] = result
            evaluatedModes[frameIndex] = evaluatedMode
            timings[frameIndex] += elapsed

            if args.policy == "rollback_q90" and machine.consumeRollbackRequest():
                target = frameIndex - 1
                if target < 1 or target not in checkpoints or target not in frameCache:
                    raise RuntimeError(f"rollback target is unavailable: {target}")
                rollbackEvents += 1
                rollbackTargets.append(target)
                _restore(controller, runtime.backend, checkpoints[target])
                machine = controller._stateMachine
                machine.suppressLostEntry(2)
                controller._mode = TrackMode.LOST
                controller._entryReason = TransitionReason.PATIENCE_EXHAUSTED
                controller._modeAgeFrames = 0
                controller._recoveryFrames = 0
                controller._recovery = RecoveryMemory()
                controller._recovery.reset(target)
                for replayIndex in (target, frameIndex):
                    recorder.resetFinalFrame(replayIndex)
                    checkpoints[replayIndex] = _checkpoint(controller, runtime.backend)
                    replayResult, replayElapsed, replayMode = _processFrame(
                        frameCache[replayIndex],
                        controller,
                        runtime,
                        recorder,
                    )
                    results[replayIndex] = replayResult
                    evaluatedModes[replayIndex] = replayMode
                    timings[replayIndex] += replayElapsed
                    replayedExecutions += 1
                machine.consumeRollbackRequest()
            for oldIndex in list(frameCache):
                if oldIndex < frameIndex - 2:
                    frameCache.pop(oldIndex, None)
                    checkpoints.pop(oldIndex, None)
            nowNs = perf_counter_ns()
            if progressIntervalNs > 0 and nowNs >= nextProgressNs:
                _printProgress(
                    args,
                    selected,
                    truth,
                    results,
                    evaluatedModes,
                    rollbackEvents=rollbackEvents,
                    replayedExecutions=replayedExecutions,
                    startedNs=initializedNs,
                    nowNs=nowNs,
                )
                nextProgressNs = nowNs + progressIntervalNs
    finally:
        closeBackend(runtime.backend)
        source.close()

    orderedResults = [results[index] for index in range(len(selected))]
    timer = SyntheticTimer(
        [timings[index] for index in range(len(selected))] + [0]
    )
    report = _summarize(
        args,
        selected,
        orderedResults,
        timer,
        recorder,
        profiler,
        candidateMinScore=appConfig.tracking.candidateMinScore,
    )
    report["experiment"] = {
        **_experimentMetadata(args, appConfig),
        "kind": "lost_state_isolated",
        "policy": args.policy,
        "viewStrategy": args.view_strategy,
    }
    _addLostMetrics(
        report,
        orderedResults,
        recorder,
        evaluatedModes=evaluatedModes,
        rollbackEvents=rollbackEvents,
        replayedExecutions=replayedExecutions,
        rollbackTargets=rollbackTargets,
    )
    _writeArtifacts(args.output, report, recorder, selected, timings)
    if not args.quiet:
        print(json.dumps(report["summary"], indent=2))
    return 0


def _printProgress(
    args: argparse.Namespace,
    selected: tuple[ManifestRecord, ...],
    truth: dict[int, ManifestRecord],
    results: dict[int, Any],
    evaluatedModes: dict[int, TrackMode],
    *,
    rollbackEvents: int,
    replayedExecutions: int,
    startedNs: int,
    nowNs: int,
) -> None:
    visibleIous: list[float] = []
    lostVisibleIous: list[float] = []
    for frameIndex, result in sorted(results.items()):
        if frameIndex == 0:
            continue
        record = truth[frameIndex]
        if not record.visible or record.bbox is None:
            continue
        iou = circularBBoxIoU(result.bbox, record.bbox, record.width)
        visibleIous.append(iou)
        if evaluatedModes.get(frameIndex) is TrackMode.LOST:
            lostVisibleIous.append(iou)

    processed = len(results)
    total = len(selected)
    elapsedSeconds = max(0.0, (nowNs - startedNs) / 1_000_000_000.0)
    remainingSeconds = (
        elapsedSeconds * (total - processed) / processed if processed else 0.0
    )
    logicalFrames = max(1, processed - 1)
    meanIou = float(np.mean(visibleIous)) if visibleIous else 0.0
    lossRate = trackingLossRate(visibleIous) if visibleIous else 0.0
    lostMeanIou = float(np.mean(lostVisibleIous)) if lostVisibleIous else 0.0
    print(
        "[progress] "
        f"{args.policy}/{args.view_strategy} {args.sequence} "
        f"frames={processed}/{total} ({processed / total:.1%}) "
        f"mean_iou={meanIou:.6f} loss_rate={lossRate:.6f} "
        f"lost_frames={sum(mode is TrackMode.LOST for mode in evaluatedModes.values())} "
        f"lost_iou={lostMeanIou:.6f} rollbacks={rollbackEvents} "
        f"replay_rate={replayedExecutions / logicalFrames:.4f} "
        f"elapsed={_formatDuration(elapsedSeconds)} eta={_formatDuration(remainingSeconds)}",
        flush=True,
    )


def _formatDuration(seconds: float) -> str:
    totalSeconds = max(0, int(round(seconds)))
    hours, remainder = divmod(totalSeconds, 3600)
    minutes, secondsPart = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secondsPart:02d}"
    return f"{minutes:02d}:{secondsPart:02d}"


def _processFrame(
    frame: FramePacket,
    controller: Any,
    runtime: Any,
    recorder: ExperimentCandidateRecorder,
) -> tuple[Any, int, TrackMode]:
    started = perf_counter_ns()
    plan = controller.beginFrame(frame)
    while True:
        views = tuple(runtime.geometry.cropViews(frame, plan.views))
        raw = tuple(runtime.backend.infer(views, plan.templateCommand))
        from instatarget.controller import calibrateLocalAppearanceProbabilities

        observations = calibrateLocalAppearanceProbabilities(raw, runtime.scoreCalibration)
        projected = tuple(
            _projectObservation(
                frame=frame,
                view=view,
                observation=observation,
                predictedMotion=plan.predictedMotion,
                geometry=runtime.geometry,
                scoreCalibration=runtime.scoreCalibration,
            )
            for view, observation in zip(views, observations, strict=True)
        )
        recorder.recordLocalRgb(frame, views)
        recorder.recordBackendBoxes(frame, views, observations)
        recorder.recordGeometryBoxes(frame, projected)
        step = controller.consume(plan, projected)
        if isinstance(step, MoreViewsRequired):
            plan = step.plan
            continue
        stateObservation = controller.lastStateObservation
        if stateObservation is None or int(stateObservation.frameIndex) != int(frame.frameIndex):
            raise RuntimeError("controller did not publish a final state observation")
        return step.result, perf_counter_ns() - started, stateObservation.evaluatedMode


def _checkpoint(controller: Any, backend: Any) -> ControllerCheckpoint:
    state = {
        name: value
        for name, value in controller.__dict__.items()
        if name not in {"_consume", "_shouldEscalate"}
    }
    return ControllerCheckpoint(deepcopy(state), int(backend.templateRevision))


def _restore(controller: Any, backend: Any, checkpoint: ControllerCheckpoint) -> None:
    experimentalConsume = controller.__dict__["_consume"]
    experimentalEscalation = controller.__dict__["_shouldEscalate"]
    controller.__dict__.clear()
    controller.__dict__.update(deepcopy(checkpoint.state))
    controller._consume = experimentalConsume
    controller._shouldEscalate = experimentalEscalation
    backend._templates._revision = checkpoint.backendRevision
    backend._previousViews.clear()
    backend._previousViewsFrameIndex = None


def _experimentalShouldEscalate(
    self: Any,
    evaluation: Any,
    transaction: Any,
    allowEscalation: bool,
) -> bool:
    if transaction.startingMode is TrackMode.LOST:
        return bool(
            allowEscalation
            and self._planner.usesSecondRound
            and transaction.attemptIndex == 0
            and transaction.remainingViews >= 4
        )
    return bool(
        evaluation.escalationRecommended
        and allowEscalation
        and self._trackingConfig.sameFrameEscalationEnabled
        and transaction.attemptIndex + 1 < self._trackingConfig.maxAttemptsPerFrame
        and transaction.remainingViews > 0
    )


def _experimentalConsume(
    self: Any,
    plan: Any,
    observations: Any,
    *,
    allowEscalation: bool,
) -> MoreViewsRequired | FrameCommitted:
    self._requireInitialized()
    planned = self._planned
    transaction = self._transaction
    if planned is None or transaction is None or planned.plan != plan:
        raise ProtocolError("search response does not match the pending attempt")
    if (
        plan.transactionId != transaction.transactionId
        or plan.attemptIndex != transaction.attemptIndex
    ):
        raise ProtocolError("search response transaction identity mismatch")
    expectedBackendRevision = self._backendRevision + 1
    if plan.templateCommand.expectedRevision != expectedBackendRevision:
        raise ProtocolError("backend template revision mismatch")
    self._backendRevision = plan.templateCommand.expectedRevision
    priorObservations = (
        tuple(item for attempt in transaction.attempts for item in attempt.observations)
        if plan.attemptIndex > 0
        else ()
    )
    evaluation = self._evaluator.evaluate(
        state=planned.state,
        plan=plan,
        observations=observations,
        priorObservations=priorObservations,
        prediction=planned.prediction,
        predictedBfov=planned.predictedBfov,
        referenceBoxAreaPx=(
            min(float(planned.frame.rgb.shape[1]), self._currentBox.widthPx)
            * self._currentBox.heightPx
        ),
        geometry=self._geometry,
        frameWidthPx=planned.frame.rgb.shape[1],
        frameHeightPx=planned.frame.rgb.shape[0],
    )
    thresholds = self._stateMachine.scoreGroup.thresholds()
    if thresholds is not None:
        evaluation = replace(
            evaluation,
            uncertainThreshold=thresholds[0],
            lostThreshold=thresholds[1],
        )
    transaction.attempts.append(
        AttemptRecord(
            kind=AttemptKind.PRIMARY if plan.attemptIndex == 0 else AttemptKind.ESCALATION,
            attemptIndex=plan.attemptIndex,
            plan=plan,
            observations=tuple(observations),
            evaluation=evaluation,
        )
    )
    transaction.completedAttempts += 1
    if self._shouldEscalate(evaluation, transaction, allowEscalation):
        if (
            planned.state.mode is TrackMode.LOST
            and self._planner.strategy == "cube6_adaptive_type1"
            and evaluation.measuredBfov is not None
        ):
            self._planner.refinementSize = (
                evaluation.measuredBfov.horizontalFovRad,
                evaluation.measuredBfov.verticalFovRad,
            )
        transaction.attemptIndex += 1
        self._planned = None
        nextPlan = self._buildAttempt(
            frame=planned.frame,
            state=planned.state,
            prediction=planned.prediction,
            predictedBfov=planned.predictedBfov,
            attemptIndex=transaction.attemptIndex,
            searchSeed=evaluation.searchSeedCenter,
            viewIdStart=max((view.viewId for view in plan.views), default=-1) + 1,
        )
        return MoreViewsRequired(nextPlan)
    decision = self._stateMachine.transition(
        planned.state.mode,
        evaluation.stateScore,
        measurementAccepted=evaluation.measurementAccepted,
    )
    result = self._commit(planned, evaluation, decision)
    self._stateMachine.recordScore(evaluation.stateScore)
    self._lastStateObservation = evaluation
    self._lastTransition = decision
    return FrameCommitted(result)


def _addLostMetrics(
    report: dict[str, Any],
    results: list[Any],
    recorder: ExperimentCandidateRecorder,
    *,
    evaluatedModes: dict[int, TrackMode],
    rollbackEvents: int,
    replayedExecutions: int,
    rollbackTargets: list[int],
) -> None:
    metrics = {int(row["frameIndex"]): row for row in report["frameMetrics"]}
    lostSearchVisibleIous = [
        float(metrics[int(result.frameIndex)]["circularErpIoU"])
        for result in results[1:]
        if evaluatedModes[int(result.frameIndex)] is TrackMode.LOST
        and metrics.get(int(result.frameIndex), {}).get("visible")
    ]
    lostSearchFrameCount = sum(
        evaluatedModes[int(result.frameIndex)] is TrackMode.LOST for result in results[1:]
    )
    transitions = Counter()
    for previous, current in zip(results, results[1:]):
        transitions[f"{previous.status.name}->{current.status.name}"] += 1
    nonInitial = max(1, len(results) - 1)
    summary = report["summary"]
    summary["lostSearchFrameCount"] = lostSearchFrameCount
    summary["lostSearchFrameRate"] = lostSearchFrameCount / nonInitial
    summary["lostSearchVisibleFrameCount"] = len(lostSearchVisibleIous)
    summary["lostSearchMeanIoU"] = (
        float(np.mean(lostSearchVisibleIous)) if lostSearchVisibleIous else 0.0
    )
    summary["lostSearchZeroIoURate"] = (
        float(np.mean(np.asarray(lostSearchVisibleIous) <= 1e-12))
        if lostSearchVisibleIous
        else 0.0
    )
    summary["lostSearchSuccessRateAt0.5"] = (
        float(np.mean(np.asarray(lostSearchVisibleIous) > 0.5))
        if lostSearchVisibleIous
        else 0.0
    )
    summary["finalLostStatusFrameCount"] = sum(
        result.status is TrackStatus.LOST for result in results[1:]
    )
    summary["rollbackEventCount"] = rollbackEvents
    summary["replayedFrameExecutions"] = replayedExecutions
    summary["rollbackFrameExecutionRate"] = replayedExecutions / nonInitial
    summary["rollbackTargetFrames"] = rollbackTargets
    summary["executedAverageViewsPerFrame"] = float(
        np.mean([recorder.executedViewCounts[index] for index in range(1, len(results))])
    )
    summary["executedAverageForwardsPerFrame"] = float(
        np.mean([recorder.executedForwardCounts[index] for index in range(1, len(results))])
    )
    summary["statusTransitions"] = dict(sorted(transitions.items()))


def _writeArtifacts(
    output: Path,
    report: dict[str, Any],
    recorder: ExperimentCandidateRecorder,
    records: tuple[ManifestRecord, ...],
    timings: Counter[int],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    candidates = output.with_name(f"{output.stem}.candidates.jsonl")
    candidates.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in recorder.rows),
        encoding="utf-8",
    )
    timingPath = output.with_name(f"{output.stem}.timings.jsonl")
    timingPath.write_text(
        "".join(
            json.dumps(
                {
                    "sequenceId": record.sequenceId,
                    "frameIndex": int(record.frameIndex),
                    "totalProcessingMs": timings[int(record.frameIndex)] / 1_000_000.0,
                    "viewCount": recorder.viewCounts[int(record.frameIndex)],
                    "forwardCount": recorder.forwardCounts[int(record.frameIndex)],
                    "executedViewCount": recorder.executedViewCounts[int(record.frameIndex)],
                    "executedForwardCount": recorder.executedForwardCounts[int(record.frameIndex)],
                },
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
