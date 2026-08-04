from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Protocol
from uuid import uuid4

from agent.session.config import SessionConfig, SnapMode


class RandomSource(Protocol):
    def randrange(self, stop: int) -> int: ...


class StopReason(str, Enum):
    """任务停止原因，便于 Pipeline 和日志区分正常上限与异常停止。"""
    ENTRY_UNAVAILABLE = "entry_unavailable"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    USER_STOPPED = "user_stopped"
    PIPELINE_STOPPED = "pipeline_stopped"
    TASK_SUCCEEDED = "task_succeeded"
    TASK_FAILED = "task_failed"
    EXTERNAL_OR_FRONTEND = "external_or_frontend"


class RecoveryAction(str, Enum):
    """异常页面出现时采用的有界恢复动作。"""
    RETRY = "retry"
    ANDROID_BACK = "android_back"
    WAIT = "wait"
    RESTART = "restart"


class SnapStage(str, Enum):
    """一局内允许独立决策的两个 SNAP 时机。"""

    FIRST = "first"
    FINAL = "final"


@dataclass(slots=True)
class SessionState:
    """本次运行中会不断变化的状态：局数、回合、SNAP 和恢复计数。"""
    config: SessionConfig
    started_at: float
    started_wall_time: float | None = None
    run_id: str = field(default_factory=lambda: uuid4().hex)
    completed_matches: int = 0
    current_turn: int = 0
    first_snap_decision_made: bool = False
    first_snap_committed: bool = False
    final_snap_decision_made: bool = False
    final_snap_committed: bool = False
    retry_count: int = 0
    back_count: int = 0
    restart_count: int = 0
    unknown_since: float | None = None
    last_known_state: str = "task_started"
    stop_reason: StopReason | None = None
    match_in_progress: bool = False
    last_task_rewards_check_at: float | None = None
    last_task_rewards_check_wall_time: float | None = None
    task_rewards_timer_started_at: float | None = None
    task_rewards_timer_started_wall_time: float | None = None
    deck_selection_completed: bool = False
    deck_selection_result: str | None = None
    daily_routine_completed_date: str | None = None
    last_stop_source: str | None = None
    last_stop_node: str | None = None
    last_stop_page: str | None = None
    last_stop_detail: str | None = None
    # 仅属于当前 Agent 进程和当前回合的安全门禁，不写入断点。重启恢复时
    # 默认拒绝结束回合，必须重新观察能量/手牌/落牌失败证据。
    end_turn_allowed: bool = False
    end_turn_reason: str | None = None

    def should_stop(self, now: float) -> bool:
        """只响应显式停止原因；正常运行不再受局数或时长限制。"""
        del now
        return self.stop_reason is not None

    def task_rewards_due(self, now: float) -> bool:
        """按配置周期检查任务奖励，但绝不在一场对局进行中打断流程。"""
        interval_hours = self.config.claim_task_rewards_hours
        if interval_hours <= 0 or self.match_in_progress:
            return False
        anchor = self.last_task_rewards_check_at
        if anchor is None:
            anchor = self.task_rewards_timer_started_at
        if anchor is None:
            anchor = self.started_at
        return now - anchor >= interval_hours * 60 * 60

    def start_task_rewards_timer(
        self,
        now: float,
        wall_time: float | None = None,
    ) -> None:
        """从本次根任务启动重新计算周期领奖，避免旧断点导致刚启动就领奖。"""
        self.task_rewards_timer_started_at = now
        self.task_rewards_timer_started_wall_time = wall_time
        self.last_task_rewards_check_at = None
        self.last_task_rewards_check_wall_time = None

    def mark_task_rewards_checked(
        self,
        now: float,
        wall_time: float | None = None,
    ) -> None:
        """记录本次领取/检查结束时间，作为下一周期的计时起点。"""
        self.last_task_rewards_check_at = now
        self.last_task_rewards_check_wall_time = wall_time

    def daily_routine_pending(self, today: date | None = None) -> bool:
        """一键日常每天只完成一次；中途重启时仍保持待处理。"""
        if not self.config.daily_routine:
            return False
        current = date.today() if today is None else today
        return self.daily_routine_completed_date != current.isoformat()

    def mark_daily_routine_completed(self, today: date | None = None) -> None:
        current = date.today() if today is None else today
        self.daily_routine_completed_date = current.isoformat()

    def reset_daily_routine(self) -> None:
        """手动启动一键日常时重新执行整条领奖链路。"""
        self.daily_routine_completed_date = None

    def request_stop(
        self,
        reason: StopReason,
        *,
        source: str | None = None,
        node: str | None = None,
        page: str | None = None,
        detail: str | None = None,
    ) -> None:
        """只记录第一个停止原因，避免后续错误覆盖真正根因。"""
        if self.stop_reason is None:
            self.stop_reason = reason
            self.last_stop_source = source
            self.last_stop_node = node
            self.last_stop_page = page
            self.last_stop_detail = detail

    def should_select_deck(self) -> bool:
        """卡组名为 0/空时禁用；其余名称每次运行只尝试一次。"""
        deck_name = self.config.deck_name.strip()
        return not self.deck_selection_completed and deck_name not in {"", "0"}

    def mark_deck_selection_completed(self, result: str = "succeeded") -> None:
        """记录本次运行唯一一次卡组选择尝试及其真实结果。"""
        if result not in {
            "succeeded",
            "fallback_not_found",
            "fallback_verification_failed",
        }:
            raise ValueError(f"unsupported deck selection result: {result}")
        self.deck_selection_completed = True
        self.deck_selection_result = result

    def begin_match(self) -> None:
        """新对局开始时重置只属于单局的数据。"""
        self.match_in_progress = True
        self.current_turn = 0
        self.first_snap_decision_made = False
        self.first_snap_committed = False
        self.final_snap_decision_made = False
        self.final_snap_committed = False
        self.deny_end_turn()

    def resume_match(self) -> None:
        """以实际战斗页面校正状态，但保留已恢复的回合和 SNAP 决策。"""
        self.match_in_progress = True

    def reconcile_home(self) -> None:
        """主页证明已离开对局；只补记一次尚未落盘的整场完成。"""
        if self.match_in_progress:
            self.complete_match()

    def complete_match(self) -> None:
        self.match_in_progress = False
        self.completed_matches += 1
        self.deny_end_turn()

    def begin_turn(self, turn: int) -> None:
        if turn < 1:
            raise ValueError("turn must be at least 1")
        self.current_turn = turn
        self.deny_end_turn()

    def allow_end_turn(self, reason: str) -> None:
        """记录当前回合已满足结束门禁，并保留可诊断的明确原因。"""
        normalized = reason.strip()
        if not normalized:
            raise ValueError("end turn reason must not be empty")
        self.end_turn_allowed = True
        self.end_turn_reason = normalized

    def deny_end_turn(self) -> None:
        self.end_turn_allowed = False
        self.end_turn_reason = None

    def should_retreat(self) -> bool:
        """配置为完成第 N 回合后撤退，因此在第 N+1 回合返回 True。"""
        threshold = self.config.retreat_after_turn
        return threshold > 0 and self.current_turn > threshold

    def decide_snap(self, stage: SnapStage, rng: RandomSource) -> bool:
        """首回合和最终回合各最多做一次 SNAP 决策。"""
        if stage is SnapStage.FIRST:
            # 首回合门禁即使被错误地放进其他回合，也不得补点 SNAP。
            if self.current_turn != 1 or self.first_snap_decision_made:
                return False
            self.first_snap_decision_made = True
        elif self.final_snap_decision_made:
            return False
        else:
            self.final_snap_decision_made = True

        if self.config.snap_mode is SnapMode.OFF:
            return False
        if self.config.snap_mode is SnapMode.ALWAYS:
            self._mark_snap_committed(stage)
            return True

        decision = rng.randrange(100) < self.config.snap_probability
        if decision:
            self._mark_snap_committed(stage)
        return decision

    def _mark_snap_committed(self, stage: SnapStage) -> None:
        if stage is SnapStage.FIRST:
            self.first_snap_committed = True
        else:
            self.final_snap_committed = True

    def mark_known(self, name: str) -> None:
        """识别回稳定页面后清空整轮连续恢复计数。"""
        self.last_known_state = name
        self._reset_recovery_phase()
        # max_restarts 限制的是一次连续异常，而不是整晚运行的累计闪退次数。
        # 否则长时间运行即使每次都恢复成功，第 4 次异常仍会被永久停止。
        self.restart_count = 0

    def next_recovery_action(self, now: float) -> RecoveryAction:
        """按有界恢复顺序给出下一步；对局中跳过危险的 Android 返回。"""
        if self.unknown_since is None:
            self.unknown_since = now

        if self.retry_count < 3:
            self.retry_count += 1
            return RecoveryAction.RETRY
        # Android 返回键在征服对局和“准备战斗？”轮间页会触发放弃/撤退。
        # 对局尚未完整结算时只允许重试、等待或重启重连，绝不发送返回键。
        if not self.match_in_progress and self.back_count < 3:
            self.back_count += 1
            return RecoveryAction.ANDROID_BACK
        if now - self.unknown_since < self.config.unknown_timeout_seconds:
            return RecoveryAction.WAIT

        if self.config.auto_restart:
            self.restart_count += 1
            self._reset_recovery_phase()
            return RecoveryAction.RESTART

        # 自动重启被关闭时仍应持续观察页面，而非把暂时的未知状态
        # 误报为任务完成。用户可随时从 Maa 手动停止任务。
        return RecoveryAction.WAIT

    def _reset_recovery_phase(self) -> None:
        self.retry_count = 0
        self.back_count = 0
        self.unknown_since = None
