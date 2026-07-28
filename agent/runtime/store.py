from collections import deque
from collections.abc import Mapping
from threading import RLock
from typing import Any

from agent.conquest.tier_policy import candidate_tiers
from agent.session.config import ConquestTier, SessionConfig
from agent.session.state import SessionState, StopReason


class RuntimeStore:
    """Agent 进程内的共享运行状态，连接配置、计数器和档位路由。"""

    def __init__(self) -> None:
        # MaaFramework 可能从不同回调线程访问 Agent，使用可重入锁保护状态。
        self._lock = RLock()
        self._state: SessionState | None = None
        self._tier_candidates: deque[ConquestTier] = deque()
        self._current_tier: ConquestTier | None = None

    def configure(self, values: Mapping[str, Any], now: float) -> SessionState:
        """根据 UI/Pipeline 参数创建全新的会话，清空上一次运行残留。"""
        config = SessionConfig.from_mapping(values)
        state = SessionState(config, started_at=now)
        with self._lock:
            self._state = state
            self._current_tier = None
            self._tier_candidates = self._build_tier_candidates(config)
        return state

    def require_state(self) -> SessionState:
        """读取当前会话；未执行初始化节点时直接报错，避免使用脏默认值。"""
        with self._lock:
            if self._state is None:
                raise RuntimeError("session has not been configured")
            return self._state

    def reset_tier_candidates(self) -> None:
        """重新生成从最高允许档位向下尝试的队列。"""
        with self._lock:
            state = self.require_state()
            self._tier_candidates = self._build_tier_candidates(state.config)
            self._current_tier = None

    def next_tier_candidate(self) -> ConquestTier | None:
        """取出下一个候选档位；连免费档也无法确认时才安全停止。"""
        with self._lock:
            state = self.require_state()
            if not self._tier_candidates:
                state.request_stop(StopReason.ENTRY_UNAVAILABLE)
                self._current_tier = None
                return None
            self._current_tier = self._tier_candidates.popleft()
            return self._current_tier

    def current_tier(self) -> ConquestTier | None:
        """返回 Pipeline 当前正在检查的档位。"""
        with self._lock:
            return self._current_tier

    @staticmethod
    def _build_tier_candidates(config: SessionConfig) -> deque[ConquestTier]:
        """例如最高黄金时固定生成：黄金 → 白银 → 试炼之地。"""
        return deque(candidate_tiers(config.max_tier))


# 所有 CustomAction / CustomRecognition 共享同一个 STORE 实例。
STORE = RuntimeStore()
