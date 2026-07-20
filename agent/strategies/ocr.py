from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CardCandidate:
    slot: int
    cost: int
    confidence: float


@dataclass(frozen=True, slots=True)
class CardDecision:
    card: CardCandidate | None
    reason: str


def choose_card(
    energy: int,
    cards: Iterable[CardCandidate],
    minimum_confidence: float,
) -> CardDecision:
    candidates = tuple(cards)
    if energy <= 0:
        return CardDecision(None, "no_energy")
    if not candidates:
        return CardDecision(None, "no_candidates")

    confident = tuple(
        card for card in candidates if card.confidence >= minimum_confidence
    )
    if not confident:
        return CardDecision(None, "low_confidence")

    affordable = tuple(
        card for card in confident if 0 <= card.cost <= energy
    )
    if not affordable:
        return CardDecision(None, "no_affordable_card")

    selected = max(affordable, key=lambda card: (card.cost, -card.slot))
    return CardDecision(selected, "selected")
