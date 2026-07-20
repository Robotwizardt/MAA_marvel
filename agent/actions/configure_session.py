import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.commands import parse_json_object
from agent.runtime.store import STORE


@AgentServer.custom_action("MarvelConfigureSession")
class ConfigureSession(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        values = parse_json_object(argv.custom_action_param)
        STORE.configure(values, time.monotonic())
        return True
