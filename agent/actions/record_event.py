import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.commands import apply_event, parse_json_object
from agent.runtime.store import STORE


@AgentServer.custom_action("MarvelRecordEvent")
class RecordEvent(CustomAction):
    """把 Pipeline 已确认发生的比赛/回合事件同步到 SessionState。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        values = parse_json_object(argv.custom_action_param)
        event = str(values.get("event", ""))
        state = STORE.require_state()
        was_match_in_progress = state.match_in_progress
        # 此动作不自行识别画面；只有前置 Pipeline 节点命中后才应调用。
        apply_event(
            state,
            event,
            values.get("value"),
            now=time.monotonic(),
            wall_time=time.time(),
        )
        known_name = (
            event
            if values.get("value") is None
            else f"{event}:{values['value']}"
        )
        state.mark_known(known_name)
        deck_detail = (
            f" deck_name={state.config.deck_name!r} "
            f"deck_result={state.deck_selection_result!r}"
            if event.startswith("deck_selection_")
            else ""
        )
        print(
            "[MarvelRecordEvent] "
            f"event={event} value={values.get('value')} "
            f"match_in_progress={state.match_in_progress} "
            f"completed_matches={state.completed_matches} "
            f"turn={state.current_turn}{deck_detail}",
            flush=True,
        )
        # 每场结束后重新从用户允许的最高档位检查实时门票数。
        if event == "match_completed" or (
            event == "page_home" and was_match_in_progress
        ):
            STORE.reset_tier_candidates()
        else:
            STORE.persist_checkpoint()
        return True
