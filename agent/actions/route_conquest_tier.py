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
        # 候选耗尽说明没有可用门票；安全停止，绝不跳转到付费入口。
        next_node = "公共-安全停止" if tier is None else TIER_NODE[tier]
        return context.override_next(argv.node_name, [next_node])
