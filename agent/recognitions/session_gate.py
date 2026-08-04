import random
import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition

from agent.runtime.commands import parse_json_object
from agent.runtime.store import STORE
from agent.recognitions.card_selection import is_active_turn
from agent.session.config import AfterRetreat, ConquestTier, GameMode
from agent.session.state import SnapStage


RNG = random.Random()


@AgentServer.custom_recognition("MarvelSessionGate")
class SessionGate(CustomRecognition):
    """把运行状态转换成 Pipeline 可识别的“命中/未命中”分支信号。"""
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        values = parse_json_object(argv.custom_recognition_param)
        command = str(values.get("command", ""))
        state = STORE.require_state()

        # 同一个 CustomRecognition 通过 command 支持多个轻量布尔判断，
        # 避免为停止、撤退、SNAP 等分别创建大量重复类。
        if command == "should_stop":
            matched = state.should_stop(time.monotonic())
        elif command == "task_rewards_due":
            matched = state.task_rewards_due(time.monotonic())
        elif command == "daily_routine_pending":
            matched = state.daily_routine_pending()
        elif command == "should_retreat":
            matched = state.should_retreat()
        elif command in {"should_snap", "should_snap_first"}:
            # should_snap 保留为旧资源兼容别名；新 Pipeline 使用显式阶段名。
            matched = state.decide_snap(SnapStage.FIRST, RNG)
        elif command == "should_snap_final":
            matched = state.decide_snap(SnapStage.FINAL, RNG)
        elif command == "after_retreat_concede":
            matched = state.config.after_retreat is AfterRetreat.CONCEDE
        elif command == "can_auto_restart":
            matched = (
                state.config.auto_restart
                and state.stop_reason is None
            )
        elif command == "match_in_progress":
            matched = state.match_in_progress
        elif command == "can_end_turn":
            # 内存许可必须与当前截图仍为可操作回合同时成立。这样 SNAP 动画、
            # 回合切换或 Agent 重启都不能沿用一份过期许可误点按钮；同时用
            # “结束回合”文字排除同为紫色按钮的“准备战斗？”。
            matched = state.end_turn_allowed and is_active_turn(
                context,
                argv.image,
            )
        elif command == "must_continue_playing":
            matched = (
                not state.end_turn_allowed
                and is_active_turn(context, argv.image)
            )
        elif command == "should_select_deck":
            matched = state.should_select_deck()
        elif command == "stop_on_daily_pass_limit":
            matched = state.config.stop_on_daily_pass_limit
        elif command == "mode_is_ladder":
            matched = state.config.game_mode is GameMode.LADDER
        elif command.startswith("tier_available:"):
            requested = ConquestTier(command.split(":", 1)[1])
            matched = STORE.current_tier() is requested
        else:
            raise ValueError(f"unsupported session gate command: {command}")

        if command in {"should_snap", "should_snap_first", "should_snap_final"}:
            STORE.persist_checkpoint()

        # 全屏 box 只表示条件为真，Pipeline 不会点击这个识别框。
        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 1920, 1080) if matched else None,
            detail={
                "command": command,
                "matched": matched,
                "end_turn_reason": state.end_turn_reason,
            },
        )
