import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.store import STORE
from agent.session.state import RecoveryAction


RECOVERY_NODE = {
    RecoveryAction.RETRY: "公共-恢复重试",
    RecoveryAction.ANDROID_BACK: "公共-恢复返回",
    RecoveryAction.WAIT: "公共-恢复等待",
    RecoveryAction.RESTART: "公共-恢复重启",
    RecoveryAction.STOP: "公共-安全停止",
}


@AgentServer.custom_action("MarvelRecoveryAction")
class RecoveryRoute(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        action = STORE.require_state().next_recovery_action(time.monotonic())
        return context.override_next(argv.node_name, [RECOVERY_NODE[action]])
