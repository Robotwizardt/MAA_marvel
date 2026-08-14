import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.commands import parse_json_object
from agent.runtime.store import STORE


@AgentServer.custom_action("MarvelConfigureSession")
class ConfigureSession(CustomAction):
    """接收 Pipeline/UI 参数，初始化本次任务共用的运行状态。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        # custom_action_param 由征服入口节点传入。
        values = parse_json_object(argv.custom_action_param)
        # STORE 保存本次征服任务的配置和计数状态，供出牌、撤退、SNAP、恢复读取。
        STORE.configure(values, time.monotonic())
        return True
