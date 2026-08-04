import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.commands import parse_json_object
from agent.runtime.diagnostics import DIAGNOSTICS
from agent.runtime.store import STORE


CONFIG_NODE_NAMES = (
    "Config_DailyBattleMode",
    "Config_PlayStrategy",
    "Config_LaneOrder",
    "Config_MaxTier",
    "Config_ReserveTickets",
    "Config_StopDailyPass",
    "Config_Retreat",
    "Config_ClaimTaskRewardsHours",
    "Config_MatchmakingTimeout",
    "Config_AutoRestart",
    "Config_DeckName",
)


def _node_custom_values(context: Context, name: str) -> dict:
    node = context.get_node_data(name) or {}
    values = (
        node.get("action", {})
        .get("param", {})
        .get("custom_action_param", {})
    )
    return dict(values) if isinstance(values, dict) else {}


@AgentServer.custom_action("MarvelConfigureSession")
class ConfigureSession(CustomAction):
    """接收 Pipeline/UI 参数，初始化本次任务共用的运行状态。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        # custom_action_param 由“征服-初始化会话”节点传入。
        # parse_json_object 负责把 MaaFramework 的 JSON 字符串转换为 dict。
        action_values = parse_json_object(argv.custom_action_param)
        values = dict(action_values)
        # MFA 对多个 option 同时覆盖同一个 custom_action_param 时会整对象替换。
        # 每个 UI 选项因此写入独立 Config_* 节点，再在这里汇总，防止最后一个
        # 选项（通常是 auto_restart）覆盖 lane_order 等全部前置配置。
        for node_name in CONFIG_NODE_NAMES:
            values.update(_node_custom_values(context, node_name))
        # 天梯入口的模式值属于入口自身，不允许被共享配置节点改回征服。
        if action_values.get("game_mode") == "ladder":
            values["game_mode"] = "ladder"
        print(f"[MarvelConfigureSession] values={values}", flush=True)
        # STORE 是 Agent 进程内的共享内存。后续出牌、撤退、SNAP、恢复等
        # 动作都会从这里读取同一份配置和计数状态。
        node_name = str(getattr(argv, "node_name", ""))
        manual_daily_start = (
            node_name == "日常-初始化会话"
            and action_values.get("daily_routine") is True
        )
        checkpoint_enabled = not (
            node_name.startswith("训练-")
            or node_name == "邮箱-初始化会话"
        )
        configured_now = time.monotonic()
        configured_wall_time = time.time()
        state = STORE.configure(
            values,
            configured_now,
            configured_wall_time,
            checkpoint_enabled=checkpoint_enabled,
            # 手动点击“一键日常”必须从领奖链起点开始，不能继承上一场
            # 对局的 turn_started/match_in_progress；新的日常会话仍会写入
            # checkpoint，游戏在同一次任务中重启时不会丢失进度。
            restore_checkpoint=not manual_daily_start,
        )
        # 一键日常是用户显式启动的根任务。即使当天曾被旧版本错误地标记为
        # 完成，也必须重新进入任务页和邮箱页核验；同一次运行中的游戏重启
        # 不会再次执行本初始化节点，因此不会破坏中途恢复。
        if manual_daily_start:
            state.reset_daily_routine()
        elif checkpoint_enabled:
            state.start_task_rewards_timer(configured_now, configured_wall_time)
        restored = STORE.last_configure_restored()
        DIAGNOSTICS.begin_run(state, restored=restored)
        # Fresh runs adopt the root task's diagnostic run_id in begin_run().
        # Persist again so an immediate process crash still resumes that same id.
        STORE.persist_checkpoint()
        DIAGNOSTICS.emit(
            state,
            event=(
                "checkpoint_disabled"
                if not checkpoint_enabled
                else ("checkpoint_restored" if restored else "checkpoint_started")
            ),
            source="agent",
            reason=(
                "transient_training_session"
                if not checkpoint_enabled
                else (
                    "manual_daily_fresh_start"
                    if manual_daily_start
                    else ("compatible_checkpoint" if restored else "new_session")
                )
            ),
        )
        # True 表示 CustomAction 成功，Pipeline 可以继续执行 next。
        return True
