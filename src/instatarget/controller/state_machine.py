"""Score-group state reduction for the four-state controller."""

from __future__ import annotations

from dataclasses import dataclass

from instatarget.controller.state_model import (
    ScoreGroup,
    TrackMode,
    TransitionDecision,
    TransitionReason,
)
from instatarget.core.config import TrackingConfig
from instatarget.core.errors import ProtocolError
from instatarget.core.types import TrackStatus


@dataclass(frozen=True, slots=True)
class StateUpdate:
    status: TrackStatus
    uncertainFrames: int
    recoveryFrames: int
    accepted: bool
    recovered: bool
    reason: TransitionReason = TransitionReason.RELIABLE_MEASUREMENT


class TrackStateMachine:
    """State selection and a committed ten-score rolling history."""

    def __init__(self, trackingConfig: TrackingConfig) -> None:
        self._config = trackingConfig
        self._status: TrackStatus | None = None
        self._scoreGroup = ScoreGroup()
        self._uncertainFrames = 0
        self._recoveryFrames = 0
        self._lostFailureCount = 0
        self._lostCounterFreezeFrames = 0

    @property
    def status(self) -> TrackStatus | None:
        return self._status

    @property
    def uncertainFrames(self) -> int:
        return self._uncertainFrames

    @property
    def recoveryFrames(self) -> int:
        """Compatibility counter; LOST has no public RECOVERING state."""
        return self._recoveryFrames

    @property
    def scoreGroup(self) -> ScoreGroup:
        return self._scoreGroup

    def initialize(self) -> StateUpdate:
        if self._status is not None:
            raise ProtocolError("track state machine is already initialized")
        self._status = TrackStatus.TRACKING
        self._scoreGroup = ScoreGroup()
        self._uncertainFrames = 0
        self._recoveryFrames = 0
        self._lostFailureCount = 0
        self._lostCounterFreezeFrames = 0
        return StateUpdate(TrackStatus.TRACKING, 0, 0, True, False, TransitionReason.INITIALIZED)

    def transition(
        self,
        mode: TrackMode,
        stateScore: float,
        *,
        measurementAccepted: bool,
    ) -> TransitionDecision:
        """Choose the next mode before appending the current score."""
        if not 0.0 <= float(stateScore) <= 1.0:
            raise ProtocolError("StateScore must be in [0, 1]")
        if mode is TrackMode.INIT:
            nextMode = TrackMode.TRACKING
            reason = TransitionReason.INITIALIZED
        elif len(self._scoreGroup.values) < 2:
            nextMode = TrackMode.TRACKING
            reason = TransitionReason.RELIABLE_MEASUREMENT
        elif len(self._scoreGroup.values) == 2:
            nextMode = (
                TrackMode.TRACKING
                if stateScore > self._scoreGroup.values[-1]
                else TrackMode.UNCERTAIN
            )
            reason = TransitionReason.RELIABLE_MEASUREMENT
        else:
            thresholds = self._scoreGroup.thresholds()
            assert thresholds is not None
            uncertainThreshold, lostThreshold = thresholds
            if stateScore <= 0.0 and uncertainThreshold == 0.0 and lostThreshold == 0.0:
                # A missing candidate is encoded as zero.  Treat the degenerate all-zero
                # warm-up window as LOST instead of allowing ``score >= UT`` to loop forever.
                nextMode = TrackMode.LOST
            elif stateScore >= uncertainThreshold:
                nextMode = TrackMode.TRACKING
            elif stateScore >= lostThreshold:
                nextMode = TrackMode.UNCERTAIN
            else:
                nextMode = TrackMode.LOST
            reason = (
                TransitionReason.RELIABLE_MEASUREMENT
                if nextMode is TrackMode.TRACKING
                else TransitionReason.WEAK_MEASUREMENT
                if nextMode is TrackMode.UNCERTAIN
                else TransitionReason.HARD_MISS
            )
        action = "COMMIT"
        if mode is TrackMode.LOST:
            self._lostFailureCount = 0
            self._consumeLostCounterFreeze()
        elif nextMode is TrackMode.LOST:
            if self._lostCounterFreezeFrames > 0:
                self._consumeLostCounterFreeze()
                nextMode = TrackMode.UNCERTAIN
                reason = TransitionReason.WEAK_MEASUREMENT
            elif self._lostFailureCount == 0:
                self._lostFailureCount = 1
                nextMode = TrackMode.UNCERTAIN
                reason = TransitionReason.WEAK_MEASUREMENT
                action = "DEFER_LOST"
            else:
                self._lostFailureCount = 0
                nextMode = TrackMode.UNCERTAIN
                action = "ROLLBACK_LOST"
        else:
            self._lostFailureCount = 0
            self._consumeLostCounterFreeze()

        reset = mode is TrackMode.LOST and measurementAccepted
        return TransitionDecision(
            action,
            nextMode,
            reason,
            measurementAccepted,
            resetMotionHistory=reset,
            resetRecoveryEpoch=reset,
        )

    def beginLostReplay(self, freezeFrames: int = 2) -> None:
        """Reset failure patience while replaying the two invalidated frames."""
        if freezeFrames < 0:
            raise ValueError("freezeFrames must be non-negative")
        self._lostFailureCount = 0
        self._lostCounterFreezeFrames = freezeFrames

    def _consumeLostCounterFreeze(self) -> None:
        if self._lostCounterFreezeFrames > 0:
            self._lostCounterFreezeFrames -= 1

    def recordScore(self, stateScore: float) -> None:
        self._scoreGroup.append(stateScore)

    def update(self, score: float | None, supported: bool, hasCandidate: bool) -> StateUpdate:
        """Compatibility adapter for old scalar callers."""
        if self._status is None:
            raise ProtocolError("track state machine has not been initialized")
        value = 0.0 if score is None else float(score)
        accepted = bool(hasCandidate and supported and value >= self._config.candidateMinScore)
        mode = {
            TrackStatus.TRACKING: TrackMode.TRACKING,
            TrackStatus.UNCERTAIN: TrackMode.UNCERTAIN,
            TrackStatus.LOST: TrackMode.LOST,
        }[self._status]
        decision = self.transition(mode, value, measurementAccepted=accepted)
        self._status = {
            TrackMode.TRACKING: TrackStatus.TRACKING,
            TrackMode.UNCERTAIN: TrackStatus.UNCERTAIN,
            TrackMode.LOST: TrackStatus.LOST,
        }[decision.nextMode]
        self.recordScore(value)
        self._uncertainFrames = (
            self._uncertainFrames + 1 if self._status is TrackStatus.UNCERTAIN else 0
        )
        self._recoveryFrames = (
            self._recoveryFrames + 1 if self._status is TrackStatus.LOST else 0
        )
        return StateUpdate(
            self._status,
            self._uncertainFrames,
            self._recoveryFrames,
            accepted,
            mode is TrackMode.LOST and accepted,
            decision.reason,
        )


__all__ = ["StateUpdate", "TrackStateMachine"]
