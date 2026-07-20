import json
from typing import Any

from agent.session.state import SessionState


def parse_json_object(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("custom parameter must be a JSON object")
    return value


def apply_event(
    state: SessionState,
    event: str,
    value: object | None = None,
) -> None:
    if event == "match_started":
        state.begin_match()
        return
    if event == "turn_started":
        turn = state.current_turn + 1 if value is None else int(value)
        state.begin_turn(turn)
        return
    if event == "match_completed":
        state.complete_match()
        return
    if event == "known_state":
        state.mark_known("pipeline" if value is None else str(value))
        return
    raise ValueError(f"unsupported session event: {event}")
