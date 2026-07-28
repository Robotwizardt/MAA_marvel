from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from agent.session.config import SessionConfig, SnapMode


class RandomSource(Protocol):
    def randrange(self, stop: int) -> int: ...


class StopReason(str, Enum):
    """任务停止原因，便于 Pipeline 和日志区分正常上限与异常停止。"""
    MAX_MATCHES = "max_matches"
    MAX_RUNTIME = "max_runtime"
    ENTRY_UNAVAILABLE = "entry_unavailable"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    USER_STOPPED = "user_stopped"


class RecoveryAction(str, Enum):
    """异常页面出现时采用的有界恢复动作。"""
    RETRY = "retry"
    ANDROID_BACK = "android_back"
    WAIT = "wait"
    RESTART = "restart"
    STOP = "stop"


@dataclass(slots=True)
class SessionState:
    """本次运行中会不断变化的状态：局数、回合、SNAP 和恢复计数。"""
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
    # 同一场地连续拖牌打开详情时，记录为仅当前回合不可用。
    blocked_lanes: set[int] = field(default_factory=set)

    def should_stop(self, now: float) -> bool:
        """统一判断是否达到停止条件，并首次写入 stop_reason。"""
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
        """只记录第一个停止原因，避免后续错误覆盖真正根因。"""
        if self.stop_reason is None:
            self.stop_reason = reason

    def begin_match(self) -> None:
        """新对局开始时重置只属于单局的数据。"""
        self.current_turn = 0
        self.snapped_this_match = False
        self.snap_decision_made = False
        self.blocked_lanes.clear()

    def complete_match(self) -> None:
        self.completed_matches += 1

    def begin_turn(self, turn: int) -> None:
        if turn < 1:
            raise ValueError("turn must be at least 1")
        self.current_turn = turn
        # 场地是否可放牌会被地点效果、回合阶段和场上卡牌变化影响。
        # 上一回合判定已满的场地不能沿用到新回合。
        self.blocked_lanes.clear()

    def should_retreat(self) -> bool:
        """配置为完成第 N 回合后撤退，因此在第 N+1 回合返回 True。"""
        threshold = self.config.retreat_after_turn
        return threshold > 0 and self.current_turn > threshold

    def decide_snap(self, rng: RandomSource) -> bool:
        """每局只做一次 SNAP 决策，防止同一局重复点击。"""
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
        """识别回稳定页面后清空当前恢复阶段的重试计数。"""
        self.last_known_state = name
        self._reset_recovery_phase()

    def next_recovery_action(self, now: float) -> RecoveryAction:
        """按重试→返回→等待→重启→停止的顺序给出下一步，全部有上限。"""
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
