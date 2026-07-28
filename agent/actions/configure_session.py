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
        # custom_action_param 由“征服-初始化会话”节点传入。
        # parse_json_object 负责把 MaaFramework 的 JSON 字符串转换为 dict。
        values = parse_json_object(argv.custom_action_param)
        # 天梯与征服共用同一组选项定义。天梯任务执行时，界面选项仍会覆盖
        # “征服-初始化会话”节点；这里读取覆盖后的公共参数，再强制改成天梯模式。
        if values.get("game_mode") == "ladder":
            node = context.get_node_data("征服-初始化会话") or {}
            shared_values = (
                node.get("action", {})
                .get("param", {})
                .get("custom_action_param", {})
            )
            if isinstance(shared_values, dict):
                values = {**shared_values, **values}
        # STORE 是 Agent 进程内的共享内存。后续出牌、撤退、SNAP、恢复等
        # 动作都会从这里读取同一份配置和计数状态。
        STORE.configure(values, time.monotonic())
        # True 表示 CustomAction 成功，Pipeline 可以继续执行 next。
        return True
