from agent.strategies.model import TurnPlan


def build_agatha_plan() -> TurnPlan:
    """阿加莎会自行出牌，因此计划中不包含任何拖动，只允许结束回合。"""
    return TurnPlan(swipes=())
