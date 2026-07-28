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
        # 此动作不自行识别画面；只有前置 Pipeline 节点命中后才应调用。
        apply_event(STORE.require_state(), event, values.get("value"))
        # 每场结束后重新从用户允许的最高档位检查实时门票数。
        if event == "match_completed":
            STORE.reset_tier_candidates()
        return True
