from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from agent.session.config import SessionConfig, SnapMode


class RandomSource(Protocol):
    def randrange(self, stop: int) -> int: ...


class StopReason(str, Enum):
    MAX_MATCHES = "max_matches"
    MAX_RUNTIME = "max_runtime"
    NO_TICKET = "no_ticket"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    USER_STOPPED = "user_stopped"


class RecoveryAction(str, Enum):
    RETRY = "retry"
    ANDROID_BACK = "android_back"
    WAIT = "wait"
    RESTART = "restart"
    STOP = "stop"


@dataclass(slots=True)
class SessionState:
    config: SessionConfig
    started_at: float
    completed_matches: int = 0
    current_turn: int = 0
    snapped_this_match: bool = False
    snap_decision_made: bool = False
    retry_count: int = 0
    back_count: int = 0
    restart_count: int = 0
    unknown_since: float | None = None
    last_known_state: str = "task_started"
    stop_reason: StopReason | None = None

    def should_stop(self, now: float) -> bool:
        if self.stop_reason is not None:
            return True
        if (
            self.config.max_matches > 0
            and self.completed_matches >= self.config.max_matches
        ):
            self.stop_reason = StopReason.MAX_MATCHES
            return True
        if (
            self.config.max_minutes > 0
            and now - self.started_at >= self.config.max_minutes * 60
        ):
            self.stop_reason = StopReason.MAX_RUNTIME
            return True
        return False

    def request_stop(self, reason: StopReason) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason

    def begin_match(self) -> None:
        self.current_turn = 0
        self.snapped_this_match = False
        self.snap_decision_made = False

    def complete_match(self) -> None:
        self.completed_matches += 1

    def begin_turn(self, turn: int) -> None:
        if turn < 1:
            raise ValueError("turn must be at least 1")
        self.current_turn = turn

    def should_retreat(self) -> bool:
        threshold = self.config.retreat_after_turn
        return threshold > 0 and self.current_turn > threshold

    def decide_snap(self, rng: RandomSource) -> bool:
        if self.snap_decision_made or self.snapped_this_match:
            return False

        self.snap_decision_made = True
        if self.config.snap_mode is SnapMode.OFF:
            return False
        if self.config.snap_mode is SnapMode.ALWAYS:
            self.snapped_this_match = True
            return True

        decision = rng.randrange(100) < self.config.snap_probability
        if decision:
            self.snapped_this_match = True
        return decision

    def mark_known(self, name: str) -> None:
        self.last_known_state = name
        self._reset_recovery_phase()

    def next_recovery_action(self, now: float) -> RecoveryAction:
        if self.unknown_since is None:
            self.unknown_since = now

        if self.retry_count < 3:
            self.retry_count += 1
            return RecoveryAction.RETRY
        if self.back_count < 3:
            self.back_count += 1
            return RecoveryAction.ANDROID_BACK
        if now - self.unknown_since < self.config.unknown_timeout_seconds:
            return RecoveryAction.WAIT

        if self.config.auto_restart and self.restart_count < self.config.max_restarts:
            self.restart_count += 1
            self._reset_recovery_phase()
            return RecoveryAction.RESTART

        self.request_stop(StopReason.RECOVERY_EXHAUSTED)
        return RecoveryAction.STOP

    def _reset_recovery_phase(self) -> None:
        self.retry_count = 0
        self.back_count = 0
        self.unknown_since = None
