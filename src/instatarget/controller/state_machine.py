"""Pure V2 state reduction with a backwards-compatible scalar adapter."""

from __future__ import annotations

from dataclasses import dataclass

from instatarget.controller.state_model import (
    EvidenceLevel,
    MeasurementEvidence,
    StateObservation,
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
    """State reducer; durable state is owned by the controller, not this object."""

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
        return StateUpdate(TrackStatus.TRACKING, 0, 0, True, False, TransitionReason.INITIALIZED)

    def transition(
        self,
        mode: TrackMode,
        observation: StateObservation,
        uncertainFrames: int,
        recoveryConfirmFrames: int,
    ) -> TransitionDecision:
        """Reduce one final frame observation without mutating controller memory."""
        if mode is TrackMode.INIT:
            return TransitionDecision(
                "COMMIT", TrackMode.TRACKING, TransitionReason.INITIALIZED, True
            )
        evidence = observation.evidence
        reliable = {
            MeasurementEvidence.RELIABLE_FUSED,
            MeasurementEvidence.RELIABLE_SINGLE,
        }
        if mode is TrackMode.TRACKING:
            if evidence in reliable:
                return TransitionDecision(
                    "COMMIT", TrackMode.TRACKING, TransitionReason.RELIABLE_MEASUREMENT, True
                )
            return TransitionDecision(
                "COMMIT",
                TrackMode.UNCERTAIN,
                (
                    TransitionReason.HARD_MISS
                    if evidence is MeasurementEvidence.MISSING
                    else TransitionReason.WEAK_MEASUREMENT
                ),
                False,
            )
        if mode is TrackMode.UNCERTAIN:
            if evidence in reliable:
                return TransitionDecision(
                    "COMMIT", TrackMode.TRACKING, TransitionReason.RELIABLE_MEASUREMENT, True
                )
            if evidence is MeasurementEvidence.WEAK and (
                uncertainFrames + 1 < self._config.uncertainPatience
            ):
                return TransitionDecision(
                    "COMMIT",
                    TrackMode.UNCERTAIN,
                    TransitionReason.WEAK_MEASUREMENT,
                    False,
                )
            return TransitionDecision(
                "COMMIT", TrackMode.LOST, TransitionReason.PATIENCE_EXHAUSTED, False
            )
        if mode is TrackMode.RECOVERING:
            if evidence is MeasurementEvidence.RELIABLE_FUSED:
                return TransitionDecision(
                    "COMMIT",
                    TrackMode.TRACKING,
                    TransitionReason.REACQUIRED,
                    True,
                    resetMotionHistory=True,
                    resetRecoveryEpoch=True,
                )
            if evidence is MeasurementEvidence.RELIABLE_SINGLE:
                if recoveryConfirmFrames + 1 >= self._config.recoverConfirmFrames:
                    return TransitionDecision(
                        "COMMIT",
                        TrackMode.TRACKING,
                        TransitionReason.REACQUIRED,
                        True,
                        resetMotionHistory=True,
                        resetRecoveryEpoch=True,
                    )
                return TransitionDecision(
                    "COMMIT", TrackMode.RECOVERING, TransitionReason.RECOVERY_PROGRESS, False
                )
            return TransitionDecision(
                "COMMIT",
                TrackMode.LOST,
                TransitionReason.HARD_MISS,
                False,
            )
        if mode is TrackMode.LOST:
            if evidence is MeasurementEvidence.RELIABLE_FUSED:
                return TransitionDecision(
                    "COMMIT",
                    TrackMode.TRACKING,
                    TransitionReason.REACQUIRED,
                    True,
                    resetMotionHistory=True,
                    resetRecoveryEpoch=True,
                )
            if evidence is MeasurementEvidence.RELIABLE_SINGLE:
                return TransitionDecision(
                    "COMMIT", TrackMode.RECOVERING, TransitionReason.RECOVERY_PROGRESS, False
                )
            return TransitionDecision("COMMIT", TrackMode.LOST, TransitionReason.HARD_MISS, False)
        return TransitionDecision(
            "COMMIT", TrackMode.TERMINATED, TransitionReason.END_OF_STREAM, False
        )

    def update(self, score: float | None, supported: bool, hasCandidate: bool) -> StateUpdate:
        """Compatibility adapter used by legacy unit callers.

        New controller code calls ``transition`` with a full ``StateObservation``.  This method
        intentionally preserves the old scalar API while applying the V2 evidence bands.
        """
        if self._status is None:
            raise ProtocolError("track state machine has not been initialized")
        value = 0.0 if score is None else float(score)
        if not hasCandidate:
            evidence = EvidenceLevel.REJECTED
        elif value >= self._config.recoverAcceptThreshold and supported:
            evidence = EvidenceLevel.REACQUIRED
        elif value >= self._config.acceptThreshold and supported:
            evidence = EvidenceLevel.CONFIRMED
        elif value >= self._config.uncertainThreshold:
            evidence = EvidenceLevel.WEAK
        else:
            evidence = EvidenceLevel.REJECTED
        if self._status is TrackStatus.TRACKING:
            if evidence is EvidenceLevel.CONFIRMED:
                self._resetStable()
            else:
                self._uncertainFrames = 1
                self._recoveryFrames = 0
                self._status = TrackStatus.UNCERTAIN
        elif self._status is TrackStatus.UNCERTAIN:
            if evidence is EvidenceLevel.CONFIRMED:
                self._resetStable()
            else:
                self._uncertainFrames += 1
                if self._uncertainFrames >= self._config.uncertainPatience:
                    self._status = TrackStatus.RECOVERING
                    self._recoveryFrames = 0
        elif self._status is TrackStatus.RECOVERING:
            if evidence is EvidenceLevel.REACQUIRED:
                self._resetStable()
                return StateUpdate(self._status, 0, 0, True, True, TransitionReason.REACQUIRED)
            self._recoveryFrames += 1
            if self._recoveryFrames >= self._config.maxRecoveryFrames:
                self._status = TrackStatus.LOST
        elif self._status is TrackStatus.LOST:
            if evidence is EvidenceLevel.REACQUIRED:
                self._resetStable()
                return StateUpdate(self._status, 0, 0, True, True, TransitionReason.REACQUIRED)
        accepted = evidence in {EvidenceLevel.CONFIRMED, EvidenceLevel.REACQUIRED}
        recovered = evidence is EvidenceLevel.REACQUIRED
        return StateUpdate(
            self._status,
            self._uncertainFrames,
            self._recoveryFrames,
            accepted,
            recovered,
            TransitionReason.RELIABLE_MEASUREMENT if accepted else TransitionReason.HARD_MISS,
        )

    def _resetStable(self) -> None:
        self._status = TrackStatus.TRACKING
        self._uncertainFrames = 0
        self._recoveryFrames = 0


__all__ = ["StateUpdate", "TrackStateMachine"]
