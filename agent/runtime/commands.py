import json
import time
from typing import Any

from agent.session.state import SessionState


def parse_json_object(raw: str) -> dict[str, Any]:
    """解析 Custom 参数，并保证顶层一定是 JSON 对象。"""
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("custom parameter must be a JSON object")
    return value


def apply_event(
    state: SessionState,
    event: str,
    value: object | None = None,
    now: float | None = None,
    wall_time: float | None = None,
) -> None:
    """集中处理 Pipeline 事件名，防止各 Action 随意修改 SessionState。"""
    if event == "match_started":
        state.begin_match()
        return
    if event == "match_resumed":
        state.resume_match()
        return
    if event == "turn_started":
        turn = state.current_turn + 1 if value is None else int(value)
        state.begin_turn(turn)
        return
    if event == "match_completed":
        state.complete_match()
        return
    if event == "task_rewards_checked":
        state.mark_task_rewards_checked(
            time.monotonic() if now is None else now,
            time.time() if wall_time is None else wall_time,
        )
        return
    if event == "daily_routine_completed":
        state.mark_daily_routine_completed()
        return
    if event == "page_home":
        state.reconcile_home()
        return
    if event == "known_state":
        state.mark_known("pipeline" if value is None else str(value))
        return
    if event in {"deck_selection_completed", "deck_selection_succeeded"}:
        state.mark_deck_selection_completed("succeeded")
        return
    if event == "deck_selection_fallback_not_found":
        state.mark_deck_selection_completed("fallback_not_found")
        return
    if event == "deck_selection_fallback_verification_failed":
        state.mark_deck_selection_completed("fallback_verification_failed")
        return
    raise ValueError(f"unsupported session event: {event}")
