"""DTC state transitions with hysteresis and bounded recovery counters."""

from __future__ import annotations

from dataclasses import dataclass

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


class TrackStateMachine:
    """Own only state labels and counters; candidate scoring stays in DecisionGate."""

    def __init__(self, trackingConfig: TrackingConfig) -> None:
        self._config = trackingConfig
        self._status: TrackStatus | None = None
        self._uncertainFrames = 0
        self._recoveryFrames = 0

    @property
    def status(self) -> TrackStatus | None:
        return self._status

    @property
    def uncertainFrames(self) -> int:
        return self._uncertainFrames

    @property
    def recoveryFrames(self) -> int:
        return self._recoveryFrames

    def initialize(self) -> StateUpdate:
        if self._status is not None:
            raise ProtocolError("track state machine is already initialized")
        self._status = TrackStatus.TRACKING
        self._uncertainFrames = 0
        self._recoveryFrames = 0
        return StateUpdate(TrackStatus.TRACKING, 0, 0, True, False)

    def update(self, score: float | None, supported: bool, hasCandidate: bool) -> StateUpdate:
        if self._status is None:
            raise ProtocolError("track state machine has not been initialized")
        candidateScore = 0.0 if score is None else float(score)
        normalAccepted = (
            hasCandidate
            and supported
            and candidateScore >= self._config.acceptThreshold
        )
        recovered = (
            hasCandidate
            and supported
            and candidateScore >= self._config.recoverAcceptThreshold
            and self._status in {TrackStatus.UNCERTAIN, TrackStatus.RECOVERING, TrackStatus.LOST}
        )

        if self._status is TrackStatus.TRACKING:
            if normalAccepted:
                self._resetStable()
            else:
                self._uncertainFrames += 1
                self._recoveryFrames = 0
                self._status = TrackStatus.UNCERTAIN
        elif self._status is TrackStatus.UNCERTAIN:
            if normalAccepted:
                self._resetStable()
            else:
                self._uncertainFrames += 1
                if self._uncertainFrames >= self._config.uncertainPatience:
                    self._status = TrackStatus.RECOVERING
                    self._recoveryFrames = 0
        elif self._status is TrackStatus.RECOVERING:
            if recovered:
                self._resetStable()
            else:
                self._recoveryFrames += 1
                if self._recoveryFrames >= self._config.maxRecoveryFrames:
                    self._status = TrackStatus.LOST
        else:
            if recovered:
                self._resetStable()
            else:
                self._recoveryFrames += 1

        return StateUpdate(
            status=self._status,
            uncertainFrames=self._uncertainFrames,
            recoveryFrames=self._recoveryFrames,
            accepted=normalAccepted or recovered,
            recovered=recovered,
        )

    def _resetStable(self) -> None:
        self._status = TrackStatus.TRACKING
        self._uncertainFrames = 0
        self._recoveryFrames = 0


__all__ = ["StateUpdate", "TrackStateMachine"]
