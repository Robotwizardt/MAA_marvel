from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CardCandidate:
    """决策层使用的轻量手牌信息，不包含截图和 MaaFramework 对象。"""
    slot: int
    cost: int
    confidence: float


@dataclass(frozen=True, slots=True)
class CardDecision:
    """选牌结果；card=None 时 reason 说明为什么没有可出的牌。"""
    card: CardCandidate | None
    reason: str


def choose_card(
    energy: int,
    cards: Iterable[CardCandidate],
    minimum_confidence: float,
) -> CardDecision:
    """从可信且可支付的牌中选择费用最高者，同费用优先最左侧。"""
    candidates = tuple(cards)
    if energy < 0:
        return CardDecision(None, "no_energy")
    if not candidates:
        return CardDecision(None, "no_candidates")

    # OCR 置信度不足的费用不能用于自动拖牌，宁可停止也不盲操作。
    confident = tuple(
        card for card in candidates if card.confidence >= minimum_confidence
    )
    if not confident:
        return CardDecision(None, "low_confidence")

    # 只保留当前能量能够支付的牌。
    # 卡牌费用的业务范围为 0～20；负数和超范围值都视为 OCR 误识别。
    affordable = tuple(card for card in confident if 0 <= card.cost <= min(energy, 20))
    if not affordable:
        return CardDecision(None, "no_affordable_card")

    # max 先比较 cost；费用相同时 -slot 越大表示位置越靠左。
    selected = max(affordable, key=lambda card: (card.cost, -card.slot))
    return CardDecision(selected, "selected")
