import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.store import STORE
from agent.runtime.diagnostics import DIAGNOSTICS
from agent.session.state import RecoveryAction


RECOVERY_NODE = {
    # Python 负责“决定下一步”，Pipeline 负责执行具体点击/等待/重启动作。
    RecoveryAction.RETRY: "公共-恢复重试",
    RecoveryAction.ANDROID_BACK: "公共-恢复返回",
    RecoveryAction.WAIT: "公共-恢复等待",
    RecoveryAction.RESTART: "公共-恢复重启",
}


@AgentServer.custom_action("MarvelRecoveryAction")
class RecoveryRoute(CustomAction):
    """根据恢复计数动态改写当前节点的 next。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        state = STORE.require_state()
        action = state.next_recovery_action(time.monotonic())
        STORE.persist_checkpoint()
        if state.retry_count == 1 or action is RecoveryAction.RESTART:
            DIAGNOSTICS.capture(
                context.tasker.controller,
                state,
                source="recovery",
                reason=(
                    "recovery_restart"
                    if action is RecoveryAction.RESTART
                    else "recovery_started"
                ),
                node=argv.node_name,
                detail={
                    "action": action.value,
                    "last_known_state": state.last_known_state,
                },
                throttle_seconds=10.0,
            )
        print(
            "[MarvelRecovery] "
            f"action={action.value} "
            f"match_in_progress={state.match_in_progress} "
            f"retry_count={state.retry_count} "
            f"back_count={state.back_count} "
            f"restart_count={state.restart_count} "
            f"last_known_state={state.last_known_state}",
            flush=True,
        )
        # override_next 只影响本次任务运行，不会修改磁盘上的 Pipeline JSON。
        return context.override_next(argv.node_name, [RECOVERY_NODE[action]])
