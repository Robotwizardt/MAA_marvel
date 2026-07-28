from collections.abc import Mapping
from dataclasses import dataclass

from agent.session.config import ConquestTier


TIER_ORDER = (
    # 从低到高排序，candidate_tiers 会按需要反转为从高到低尝试。
    ConquestTier.PROVING_GROUNDS,
    ConquestTier.SILVER,
    ConquestTier.GOLD,
    ConquestTier.INFINITE,
)


def candidate_tiers(max_tier: ConquestTier) -> tuple[ConquestTier, ...]:
    """返回不超过最高档位的候选序列，例如黄金→白银→试炼。"""
    last = TIER_ORDER.index(max_tier)
    return tuple(reversed(TIER_ORDER[: last + 1]))


def choose_tier(
    max_tier: ConquestTier,
    ticket_counts: Mapping[ConquestTier, int],
    reserves: Mapping[ConquestTier, int],
) -> ConquestTier:
    """选择数量严格超过保留值的最高档位；最终总会回退到免费试炼。"""
    for tier in candidate_tiers(max_tier):
        if tier is ConquestTier.PROVING_GROUNDS:
            return tier
        if ticket_counts.get(tier, 0) > reserves.get(tier, 0):
            return tier
    return ConquestTier.PROVING_GROUNDS


@dataclass(frozen=True, slots=True)
class EntryEvidence:
    """进入按钮附近的安全证据；付费相关证据具有最高否决权。"""
    tier: ConquestTier
    free_label: bool
    ticket_count: int | None
    reserve_count: int
    gold_icon: bool
    gold_amount: bool
    paid_confirmation: bool


def is_safe_entry(evidence: EntryEvidence) -> bool:
    """只有明确免费或明确使用已有门票，且没有金块证据时才允许进入。"""
    # 任意付费证据出现都直接拒绝，避免 OCR 冲突时误花金块。
    if evidence.gold_icon or evidence.gold_amount or evidence.paid_confirmation:
        return False
    if evidence.tier is ConquestTier.PROVING_GROUNDS:
        return evidence.free_label and evidence.ticket_count is None
    return (
        evidence.ticket_count is not None
        and evidence.ticket_count > evidence.reserve_count
        and not evidence.free_label
    )
