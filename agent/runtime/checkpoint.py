from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from agent.session.config import ConquestTier, GameMode, SessionConfig
from agent.session.state import SessionState


CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_PATH = Path("config") / "marvel_session_checkpoint.json"


def config_fingerprint(config: SessionConfig) -> str:
    """Return a stable fingerprint for every user-visible session setting."""
    values: dict[str, Any] = {}
    for item in fields(config):
        value = getattr(config, item.name)
        values[item.name] = getattr(value, "value", value)
    encoded = json.dumps(
        values,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RestoredSession:
    state: SessionState
    current_tier: ConquestTier | None
    tier_candidates: tuple[ConquestTier, ...]


class SessionCheckpointStore:
    """Versioned, atomic persistence for the minimum resumable session state."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled

    @classmethod
    def for_runtime(cls) -> "SessionCheckpointStore":
        # A packaged MFA installation owns config/instances. Source tests do not,
        # so importing the global STORE never creates runtime files in the repo.
        enabled = (DEFAULT_CHECKPOINT_PATH.parent / "instances").is_dir()
        return cls(DEFAULT_CHECKPOINT_PATH, enabled=enabled)

    @property
    def temporary_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".tmp")

    def restore(
        self,
        config: SessionConfig,
        *,
        now: float,
        wall_time: float,
    ) -> RestoredSession | None:
        if not self.enabled:
            return None
        document = self._read_document()
        if document is None:
            return None
        if document.get("version") != CHECKPOINT_VERSION:
            print(
                "[MarvelCheckpoint] restore_skipped reason=version_mismatch",
                flush=True,
            )
            return None
        sessions = document.get("sessions")
        if not isinstance(sessions, dict):
            return None
        payload = sessions.get(config.game_mode.value)
        if not isinstance(payload, dict):
            return None
        if payload.get("config_fingerprint") != config_fingerprint(config):
            print(
                "[MarvelCheckpoint] restore_skipped reason=config_changed "
                f"mode={config.game_mode.value}",
                flush=True,
            )
            return None
        try:
            return self._decode_session(config, payload, now, wall_time)
        except (KeyError, TypeError, ValueError) as error:
            print(
                "[MarvelCheckpoint] restore_failed "
                f"mode={config.game_mode.value} error={error}",
                flush=True,
            )
            return None

    def save(
        self,
        state: SessionState,
        *,
        current_tier: ConquestTier | None,
        tier_candidates: Iterable[ConquestTier],
        now: float | None = None,
        wall_time: float | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        monotonic_now = time.monotonic() if now is None else now
        epoch_now = time.time() if wall_time is None else wall_time
        payload = self._encode_session(
            state,
            current_tier=current_tier,
            tier_candidates=tier_candidates,
            now=monotonic_now,
            wall_time=epoch_now,
        )
        document = self._read_document()
        if document is None or document.get("version") != CHECKPOINT_VERSION:
            document = {"version": CHECKPOINT_VERSION, "sessions": {}}
        sessions = document.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}
            document["sessions"] = sessions
        sessions[state.config.game_mode.value] = payload
        return self._write_document(document)

    def clear(self, mode: GameMode) -> bool:
        if not self.enabled:
            return False
        document = self._read_document()
        if document is None or document.get("version") != CHECKPOINT_VERSION:
            return self._remove_files()
        sessions = document.get("sessions")
        if not isinstance(sessions, dict):
            return self._remove_files()
        sessions.pop(mode.value, None)
        if not sessions:
            return self._remove_files()
        return self._write_document(document)

    def _read_document(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError("checkpoint root must be an object")
            return value
        except (OSError, json.JSONDecodeError, TypeError) as error:
            print(f"[MarvelCheckpoint] restore_failed error={error}", flush=True)
            return None

    def _write_document(self, document: dict[str, Any]) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.temporary_path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.temporary_path.replace(self.path)
            return True
        except OSError as error:
            print(f"[MarvelCheckpoint] write_failed error={error}", flush=True)
            try:
                self.temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _remove_files(self) -> bool:
        removed = False
        for path in (self.path, self.temporary_path):
            try:
                if path.exists():
                    path.unlink()
                    removed = True
            except OSError as error:
                print(f"[MarvelCheckpoint] clear_failed error={error}", flush=True)
                return False
        return removed

    @staticmethod
    def _encode_session(
        state: SessionState,
        *,
        current_tier: ConquestTier | None,
        tier_candidates: Iterable[ConquestTier],
        now: float,
        wall_time: float,
    ) -> dict[str, Any]:
        started_wall_time = state.started_wall_time
        if started_wall_time is None:
            started_wall_time = wall_time - max(0.0, now - state.started_at)
        rewards_wall_time = state.last_task_rewards_check_wall_time
        if rewards_wall_time is None and state.last_task_rewards_check_at is not None:
            rewards_wall_time = wall_time - max(
                0.0,
                now - state.last_task_rewards_check_at,
            )
        rewards_timer_wall_time = state.task_rewards_timer_started_wall_time
        if (
            rewards_timer_wall_time is None
            and state.task_rewards_timer_started_at is not None
        ):
            rewards_timer_wall_time = wall_time - max(
                0.0,
                now - state.task_rewards_timer_started_at,
            )
        return {
            "config_fingerprint": config_fingerprint(state.config),
            "saved_at": wall_time,
            "state": {
                "run_id": state.run_id,
                "completed_matches": state.completed_matches,
                "current_turn": state.current_turn,
                "first_snap_decision_made": state.first_snap_decision_made,
                "first_snap_committed": state.first_snap_committed,
                "final_snap_decision_made": state.final_snap_decision_made,
                "final_snap_committed": state.final_snap_committed,
                "last_known_state": state.last_known_state,
                "match_in_progress": state.match_in_progress,
                "deck_selection_completed": state.deck_selection_completed,
                "deck_selection_result": state.deck_selection_result,
                "daily_routine_completed_date": (
                    state.daily_routine_completed_date
                ),
                "started_wall_time": started_wall_time,
                "task_rewards_timer_started_wall_time": (
                    rewards_timer_wall_time
                ),
                "last_task_rewards_check_wall_time": rewards_wall_time,
            },
            "conquest": {
                "current_tier": None if current_tier is None else current_tier.value,
                "tier_candidates": [tier.value for tier in tier_candidates],
            },
        }

    @staticmethod
    def _decode_session(
        config: SessionConfig,
        payload: dict[str, Any],
        now: float,
        wall_time: float,
    ) -> RestoredSession:
        values = payload["state"]
        if not isinstance(values, dict):
            raise TypeError("state must be an object")
        run_id = values["run_id"]
        last_known_state = values["last_known_state"]
        if not isinstance(run_id, str) or not run_id:
            raise TypeError("run_id must be a non-empty string")
        if not isinstance(last_known_state, str):
            raise TypeError("last_known_state must be a string")
        started_wall_time = float(values["started_wall_time"])
        elapsed = max(0.0, wall_time - started_wall_time)
        rewards_wall_value = values.get("last_task_rewards_check_wall_time")
        rewards_wall_time = (
            None if rewards_wall_value is None else float(rewards_wall_value)
        )
        rewards_at = None
        if rewards_wall_time is not None:
            rewards_at = now - max(0.0, wall_time - rewards_wall_time)
        rewards_timer_wall_value = values.get(
            "task_rewards_timer_started_wall_time"
        )
        rewards_timer_wall_time = (
            None
            if rewards_timer_wall_value is None
            else float(rewards_timer_wall_value)
        )
        rewards_timer_at = None
        if rewards_timer_wall_time is not None:
            rewards_timer_at = now - max(
                0.0,
                wall_time - rewards_timer_wall_time,
            )
        daily_completed_date = values.get("daily_routine_completed_date")
        if daily_completed_date is not None and not isinstance(
            daily_completed_date,
            str,
        ):
            raise TypeError("daily_routine_completed_date must be a string or null")
        deck_selection_result = values.get("deck_selection_result")
        valid_deck_results = {
            "succeeded",
            "fallback_not_found",
            "fallback_verification_failed",
        }
        if (
            deck_selection_result is not None
            and deck_selection_result not in valid_deck_results
        ):
            raise TypeError("deck_selection_result is invalid")
        deck_selection_completed = _bool(values, "deck_selection_completed")
        # 旧版本把“未找到回退”也伪装成 completed，且没有保存真实结果。
        # 这种旧断点升级后重新尝试一次，避免永远沿用错误卡组。
        if deck_selection_completed and deck_selection_result is None:
            deck_selection_completed = False

        state = SessionState(
            config=config,
            started_at=now - elapsed,
            started_wall_time=started_wall_time,
            run_id=run_id,
            completed_matches=_non_negative_int(values, "completed_matches"),
            current_turn=_non_negative_int(values, "current_turn"),
            first_snap_decision_made=_bool(values, "first_snap_decision_made"),
            first_snap_committed=_bool(values, "first_snap_committed"),
            final_snap_decision_made=_bool(values, "final_snap_decision_made"),
            final_snap_committed=_bool(values, "final_snap_committed"),
            last_known_state=last_known_state,
            match_in_progress=_bool(values, "match_in_progress"),
            last_task_rewards_check_at=rewards_at,
            last_task_rewards_check_wall_time=rewards_wall_time,
            task_rewards_timer_started_at=rewards_timer_at,
            task_rewards_timer_started_wall_time=rewards_timer_wall_time,
            deck_selection_completed=deck_selection_completed,
            deck_selection_result=deck_selection_result,
            daily_routine_completed_date=daily_completed_date,
        )
        conquest = payload.get("conquest", {})
        if not isinstance(conquest, dict):
            raise TypeError("conquest must be an object")
        current_value = conquest.get("current_tier")
        current_tier = None if current_value is None else ConquestTier(current_value)
        candidates_value = conquest.get("tier_candidates", [])
        if not isinstance(candidates_value, list):
            raise TypeError("tier_candidates must be an array")
        candidates = tuple(ConquestTier(value) for value in candidates_value)
        return RestoredSession(state, current_tier, candidates)


def _non_negative_int(values: dict[str, Any], name: str) -> int:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


def _bool(values: dict[str, Any], name: str) -> bool:
    value = values[name]
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value
