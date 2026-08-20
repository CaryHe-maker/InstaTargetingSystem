from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from instatarget.controller.state_model import TrackMode
from instatarget.core.config import loadConfig
from instatarget.core.types import BBoxXYWH, BFoV, TrackStatus
from instatarget.geometry.projection_math import makeSphericalPoint

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from eval_lost_experiment import (  # noqa: E402
    ExperimentalRecoveryPlanner,
    ExperimentalScoreGroup,
    ExperimentalStateMachine,
)


def testExperimentalScoreGroupUsesRequestedOrderStatistic() -> None:
    q80 = ExperimentalScoreGroup(8)
    q90 = ExperimentalScoreGroup(9)
    for score in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        q80.append(score)
        q90.append(score)
    assert q80.thresholds() == (0.6, 0.3)
    assert q90.thresholds() == (0.6, 0.2)


def testRollbackPolicyRequiresTwoLowScoresAndSuppressesTwoReplayFrames() -> None:
    machine = ExperimentalStateMachine(SimpleNamespace(candidateMinScore=0.5), "rollback_q90")
    machine.initialize()
    for score in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
        machine.recordScore(score)
    first = machine.transition(TrackMode.TRACKING, 0.01, measurementAccepted=False)
    machine.recordScore(0.01)
    second = machine.transition(TrackMode.UNCERTAIN, 0.0, measurementAccepted=False)
    assert first.nextMode is TrackMode.UNCERTAIN
    assert second.nextMode is TrackMode.LOST
    assert machine.consumeRollbackRequest()

    machine.suppressLostEntry(2)
    strong = machine.transition(TrackMode.LOST, 1.0, measurementAccepted=True)
    machine.recordScore(1.0)
    low = machine.transition(TrackMode.TRACKING, 0.0, measurementAccepted=False)
    assert strong.nextMode is TrackMode.TRACKING
    assert low.nextMode is TrackMode.UNCERTAIN
    assert not machine.consumeRollbackRequest()


def testHysteresisPolicyRequiresTwoStrongRecoveryFrames() -> None:
    machine = ExperimentalStateMachine(
        SimpleNamespace(candidateMinScore=0.5),
        "hysteresis_q90",
    )
    machine.initialize()
    for score in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
        machine.recordScore(score)
    first = machine.transition(TrackMode.LOST, 0.95, measurementAccepted=True)
    machine.recordScore(0.95)
    second = machine.transition(TrackMode.LOST, 0.96, measurementAccepted=True)
    assert first.nextMode is TrackMode.LOST
    assert second.nextMode is TrackMode.TRACKING


def testLostViewStrategiesRespectSixPlusFourAndTwelveViewBudgets() -> None:
    root = Path(__file__).resolve().parents[2]
    config = loadConfig(root / "configs" / "RGBonly.yaml")
    center = makeSphericalPoint(0.0, 0.0)
    bfov = BFoV(center, 0.4, 0.3)
    box = BBoxXYWH(100.0, 100.0, 80.0, 60.0)

    sequential = ExperimentalRecoveryPlanner(
        config.geometry,
        config.tracking,
        config.recovery,
        "cube6_type1",
    )
    first = sequential.buildViews(
        1,
        1440,
        720,
        box,
        box,
        bfov,
        None,
        TrackStatus.LOST,
        attemptIndex=0,
        viewBudget=12,
    )
    second = sequential.buildViews(
        1,
        1440,
        720,
        box,
        box,
        bfov,
        None,
        TrackStatus.LOST,
        attemptIndex=1,
        viewIdStart=6,
        viewBudget=6,
        searchSeedCenter=center,
    )
    dual = ExperimentalRecoveryPlanner(
        config.geometry,
        config.tracking,
        config.recovery,
        "dual_cube12",
    ).buildViews(
        1,
        1440,
        720,
        box,
        box,
        bfov,
        None,
        TrackStatus.LOST,
        attemptIndex=0,
        viewBudget=12,
    )
    assert len(first) == 6
    assert len(second) == 4
    assert len(dual) == 12
    assert len({item.spec.viewId for item in dual}) == 12
