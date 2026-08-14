from collections import deque
from collections.abc import Mapping
from threading import RLock
import time
from typing import Any

from agent.conquest.tier_policy import candidate_tiers
from agent.runtime.checkpoint import SessionCheckpointStore
from agent.session.config import ConquestTier, SessionConfig
from agent.session.state import SessionState


class RuntimeStore:
    """Agent 进程内的共享运行状态，连接配置、计数器和档位路由。"""

    def __init__(
        self,
        checkpoint_store: SessionCheckpointStore | None = None,
    ) -> None:
        # MaaFramework 可能从不同回调线程访问 Agent，使用可重入锁保护状态。
        self._lock = RLock()
        self._state: SessionState | None = None
        self._tier_candidates: deque[ConquestTier] = deque()
        self._current_tier: ConquestTier | None = None
        self._resume_current_tier = False
        self._checkpoint = (
            SessionCheckpointStore.for_runtime()
            if checkpoint_store is None
            else checkpoint_store
        )
        self._session_checkpoint_enabled = True
        self._last_configure_restored = False

    def configure(
        self,
        values: Mapping[str, Any],
        now: float,
        wall_time: float | None = None,
        checkpoint_enabled: bool = True,
        restore_checkpoint: bool | None = None,
    ) -> SessionState:
        """恢复相同配置的断点，否则创建一份全新的会话。"""
        config = SessionConfig.from_mapping(values)
        epoch_now = time.time() if wall_time is None else wall_time
        should_restore = (
            checkpoint_enabled
            if restore_checkpoint is None
            else restore_checkpoint
        )
        restored = (
            self._checkpoint.restore(
                config,
                now=now,
                wall_time=epoch_now,
            )
            if should_restore
            else None
        )
        with self._lock:
            self._session_checkpoint_enabled = checkpoint_enabled
            if restored is None:
                state = SessionState(
                    config,
                    started_at=now,
                    started_wall_time=epoch_now,
                )
                self._current_tier = None
                self._tier_candidates = self._build_tier_candidates(config)
                self._resume_current_tier = False
                self._last_configure_restored = False
            else:
                state = restored.state
                self._current_tier = restored.current_tier
                self._tier_candidates = deque(restored.tier_candidates)
                self._resume_current_tier = restored.current_tier is not None
                self._last_configure_restored = True
            self._state = state
            self._persist_locked(now=now, wall_time=epoch_now)
        return state

    def last_configure_restored(self) -> bool:
        with self._lock:
            return self._last_configure_restored

    def require_state(self) -> SessionState:
        """读取当前会话；未执行初始化节点时直接报错，避免使用脏默认值。"""
        with self._lock:
            if self._state is None:
                raise RuntimeError("session has not been configured")
            return self._state

    def state_or_none(self) -> SessionState | None:
        """事件监听器可能早于会话初始化收到通知，因此提供安全读取。"""
        with self._lock:
            return self._state

    def clear_state(self) -> None:
        """新的根任务开始时只清理内存；异常退出断点必须继续保留。"""
        with self._lock:
            self._state = None
            self._current_tier = None
            self._resume_current_tier = False
            self._tier_candidates.clear()
            self._session_checkpoint_enabled = False

    def persist_checkpoint(
        self,
        *,
        now: float | None = None,
        wall_time: float | None = None,
    ) -> bool:
        with self._lock:
            return self._persist_locked(now=now, wall_time=wall_time)

    def clear_checkpoint(self) -> bool:
        """显式安全停止只清除当前模式的断点，不影响另一个模式。"""
        with self._lock:
            if self._state is None or not self._session_checkpoint_enabled:
                return False
            cleared = self._checkpoint.clear(self._state.config.game_mode)
            print(
                "[MarvelCheckpoint] cleared "
                f"mode={self._state.config.game_mode.value} removed={cleared}",
                flush=True,
            )
            return cleared

    def reset_tier_candidates(self) -> None:
        """重新生成从最高允许档位向下尝试的队列。"""
        with self._lock:
            state = self.require_state()
            self._tier_candidates = self._build_tier_candidates(state.config)
            self._current_tier = None
            self._resume_current_tier = False
            self._persist_locked()

    def next_tier_candidate(self) -> ConquestTier | None:
        """取出下一个候选档位；候选耗尽时由调用方等待后重新检查。"""
        with self._lock:
            self.require_state()
            if self._resume_current_tier and self._current_tier is not None:
                self._resume_current_tier = False
                tier = self._current_tier
            elif not self._tier_candidates:
                self._current_tier = None
                tier = None
            else:
                self._current_tier = self._tier_candidates.popleft()
                tier = self._current_tier
            self._persist_locked()
            return tier

    def current_tier(self) -> ConquestTier | None:
        """返回 Pipeline 当前正在检查的档位。"""
        with self._lock:
            return self._current_tier

    def tier_candidates(self) -> tuple[ConquestTier, ...]:
        with self._lock:
            return tuple(self._tier_candidates)

    def _persist_locked(
        self,
        *,
        now: float | None = None,
        wall_time: float | None = None,
    ) -> bool:
        if self._state is None or not self._session_checkpoint_enabled:
            return False
        return self._checkpoint.save(
            self._state,
            current_tier=self._current_tier,
            tier_candidates=self._tier_candidates,
            now=now,
            wall_time=wall_time,
        )

    @staticmethod
    def _build_tier_candidates(config: SessionConfig) -> deque[ConquestTier]:
        """例如最高黄金时固定生成：黄金 → 白银 → 试炼之地。"""
        return deque(candidate_tiers(config.max_tier))


# 所有 CustomAction / CustomRecognition 共享同一个 STORE 实例。
STORE = RuntimeStore()
