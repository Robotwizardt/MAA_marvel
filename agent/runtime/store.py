from collections import deque
from collections.abc import Mapping
from threading import RLock
from typing import Any

from agent.conquest.tier_policy import candidate_tiers
from agent.session.config import ConquestTier, NoTicketBehavior, SessionConfig
from agent.session.state import SessionState, StopReason


class RuntimeStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._state: SessionState | None = None
        self._tier_candidates: deque[ConquestTier] = deque()
        self._current_tier: ConquestTier | None = None

    def configure(self, values: Mapping[str, Any], now: float) -> SessionState:
        config = SessionConfig.from_mapping(values)
        state = SessionState(config, started_at=now)
        with self._lock:
            self._state = state
            self._current_tier = None
            self._tier_candidates = self._build_tier_candidates(config)
        return state

    def require_state(self) -> SessionState:
        with self._lock:
            if self._state is None:
                raise RuntimeError("session has not been configured")
            return self._state

    def reset_tier_candidates(self) -> None:
        with self._lock:
            state = self.require_state()
            self._tier_candidates = self._build_tier_candidates(state.config)
            self._current_tier = None

    def next_tier_candidate(self) -> ConquestTier | None:
        with self._lock:
            state = self.require_state()
            if not self._tier_candidates:
                state.request_stop(StopReason.NO_TICKET)
                self._current_tier = None
                return None
            self._current_tier = self._tier_candidates.popleft()
            return self._current_tier

    def current_tier(self) -> ConquestTier | None:
        with self._lock:
            return self._current_tier

    @staticmethod
    def _build_tier_candidates(config: SessionConfig) -> deque[ConquestTier]:
        if config.max_tier is ConquestTier.PROVING_GROUNDS:
            return deque((ConquestTier.PROVING_GROUNDS,))

        candidates = deque(
            tier
            for tier in candidate_tiers(config.max_tier)
            if tier is not ConquestTier.PROVING_GROUNDS
        )
        if config.no_ticket is NoTicketBehavior.FALLBACK:
            candidates.append(ConquestTier.PROVING_GROUNDS)
        return candidates


STORE = RuntimeStore()
