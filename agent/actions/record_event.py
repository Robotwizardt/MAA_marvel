from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.commands import apply_event, parse_json_object
from agent.runtime.store import STORE


@AgentServer.custom_action("MarvelRecordEvent")
class RecordEvent(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        values = parse_json_object(argv.custom_action_param)
        event = str(values.get("event", ""))
        apply_event(STORE.require_state(), event, values.get("value"))
        return True
