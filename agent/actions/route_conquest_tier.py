from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.store import STORE
from agent.session.config import ConquestTier


TIER_NODE = {
    # 业务枚举与 Pipeline 节点名的唯一映射表。
    ConquestTier.PROVING_GROUNDS: "征服-准备试炼之地",
    ConquestTier.SILVER: "征服-准备白银",
    ConquestTier.GOLD: "征服-准备黄金",
    ConquestTier.INFINITE: "征服-准备无限",
}


@AgentServer.custom_action("MarvelRouteConquestTier")
class RouteConquestTier(CustomAction):
    """依次选择允许的征服档位，并把 Pipeline 路由到对应检查节点。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        tier = STORE.next_tier_candidate()
        state = STORE.require_state()
        # 一轮候选全部确认失败通常表示征服大厅已经失步（例如卡片标题
        # 被遮挡或页面没有完成切换）。重置队列后立即走已有恢复重启链，
        # 避免把空队列写入断点后在下一次恢复中再次无限循环。
        if tier is None:
            STORE.reset_tier_candidates()
            next_node = "征服-无可用档位等待"
            state.mark_known("conquest_tiers_exhausted")
            if state.config.auto_restart:
                next_node = "公共-恢复重启"
        else:
            next_node = TIER_NODE[tier]
            state.mark_known(f"conquest_tier:{tier.value}")
        STORE.persist_checkpoint()
        return context.override_next(argv.node_name, [next_node])
