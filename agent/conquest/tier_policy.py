from collections.abc import Collection
from dataclasses import dataclass

from agent.session.config import ConquestTier, NoTicketBehavior


TIER_ORDER = (
    ConquestTier.PROVING_GROUNDS,
    ConquestTier.SILVER,
    ConquestTier.GOLD,
    ConquestTier.INFINITE,
)


def candidate_tiers(max_tier: ConquestTier) -> tuple[ConquestTier, ...]:
    last = TIER_ORDER.index(max_tier)
    return tuple(reversed(TIER_ORDER[: last + 1]))


def choose_tier(
    max_tier: ConquestTier,
    available_tickets: Collection[ConquestTier],
    no_ticket: NoTicketBehavior,
) -> ConquestTier | None:
    if max_tier is ConquestTier.PROVING_GROUNDS:
        return ConquestTier.PROVING_GROUNDS

    for tier in candidate_tiers(max_tier):
        if tier is not ConquestTier.PROVING_GROUNDS and tier in available_tickets:
            return tier

    if no_ticket is NoTicketBehavior.FALLBACK:
        return ConquestTier.PROVING_GROUNDS
    return None


@dataclass(frozen=True, slots=True)
class EntryEvidence:
    tier: ConquestTier
    free_label: bool
    ticket_label: bool
    gold_icon: bool
    gold_amount: bool
    paid_confirmation: bool


def is_safe_entry(evidence: EntryEvidence) -> bool:
    if evidence.gold_icon or evidence.gold_amount or evidence.paid_confirmation:
        return False
    if evidence.tier is ConquestTier.PROVING_GROUNDS:
        return evidence.free_label and not evidence.ticket_label
    return evidence.ticket_label and not evidence.free_label
