from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.commands import parse_json_object
from agent.runtime.diagnostics import DIAGNOSTICS
from agent.runtime.store import STORE
from agent.session.state import StopReason


@AgentServer.custom_action("MarvelTraceRuntime")
class TraceRuntime(CustomAction):
    """由 Pipeline 在停止等关键动作前登记来源并保存现场。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        values = parse_json_object(argv.custom_action_param)
        source = str(values.get("source", "pipeline"))
        reason = str(values.get("reason", "pipeline_event"))
        state = STORE.state_or_none()
        if values.get("stop", False):
            DIAGNOSTICS.register_stop_intent(
                source=source,
                reason=reason,
                node=argv.node_name,
            )
        if state is not None and values.get("stop", False):
            state.request_stop(
                StopReason.PIPELINE_STOPPED,
                source=source,
                node=argv.node_name,
                page=state.last_known_state,
                detail=reason,
            )
        DIAGNOSTICS.capture(
            context.tasker.controller,
            state,
            source=source,
            reason=reason,
            node=argv.node_name,
            detail=values.get("detail"),
            throttle_seconds=0.0,
        )
        if values.get("stop", False):
            STORE.clear_checkpoint()
        return True
