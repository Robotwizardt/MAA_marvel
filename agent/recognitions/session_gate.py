import random
import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition

from agent.runtime.commands import parse_json_object
from agent.runtime.store import STORE
from agent.session.config import AfterRetreat, ConquestTier


RNG = random.Random()


@AgentServer.custom_recognition("MarvelSessionGate")
class SessionGate(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        values = parse_json_object(argv.custom_recognition_param)
        command = str(values.get("command", ""))
        state = STORE.require_state()

        if command == "should_stop":
            matched = state.should_stop(time.monotonic())
        elif command == "should_retreat":
            matched = state.should_retreat()
        elif command == "should_snap":
            matched = state.decide_snap(RNG)
        elif command == "after_retreat_concede":
            matched = state.config.after_retreat is AfterRetreat.CONCEDE
        elif command == "can_auto_restart":
            matched = (
                state.config.auto_restart
                and state.restart_count < state.config.max_restarts
                and state.stop_reason is None
            )
        elif command.startswith("tier_available:"):
            requested = ConquestTier(command.split(":", 1)[1])
            matched = STORE.current_tier() is requested
        else:
            raise ValueError(f"unsupported session gate command: {command}")

        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 720, 1280) if matched else None,
            detail={"command": command, "matched": matched},
        )
