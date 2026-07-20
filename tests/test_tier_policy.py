import unittest

from agent.conquest.tier_policy import (
    EntryEvidence,
    candidate_tiers,
    choose_tier,
    is_safe_entry,
)
from agent.session.config import ConquestTier, NoTicketBehavior


class TierPolicyTests(unittest.TestCase):
    def test_candidates_descend_from_allowed_maximum(self) -> None:
        self.assertEqual(
            candidate_tiers(ConquestTier.GOLD),
            (
                ConquestTier.GOLD,
                ConquestTier.SILVER,
                ConquestTier.PROVING_GROUNDS,
            ),
        )

    def test_highest_available_paid_tier_is_selected(self) -> None:
        selected = choose_tier(
            ConquestTier.INFINITE,
            {ConquestTier.SILVER, ConquestTier.GOLD},
            NoTicketBehavior.STOP,
        )
        self.assertEqual(selected, ConquestTier.GOLD)

    def test_proving_grounds_maximum_is_always_free_tier(self) -> None:
        selected = choose_tier(
            ConquestTier.PROVING_GROUNDS,
            set(),
            NoTicketBehavior.STOP,
        )
        self.assertEqual(selected, ConquestTier.PROVING_GROUNDS)

    def test_no_ticket_falls_back_or_stops(self) -> None:
        self.assertEqual(
            choose_tier(
                ConquestTier.INFINITE,
                set(),
                NoTicketBehavior.FALLBACK,
            ),
            ConquestTier.PROVING_GROUNDS,
        )
        self.assertIsNone(
            choose_tier(
                ConquestTier.INFINITE,
                set(),
                NoTicketBehavior.STOP,
            )
        )

    def test_proving_grounds_requires_only_free_evidence(self) -> None:
        safe = EntryEvidence(
            tier=ConquestTier.PROVING_GROUNDS,
            free_label=True,
            ticket_label=False,
            gold_icon=False,
            gold_amount=False,
            paid_confirmation=False,
        )
        self.assertTrue(is_safe_entry(safe))

    def test_paid_tier_requires_only_ticket_evidence(self) -> None:
        safe = EntryEvidence(
            tier=ConquestTier.SILVER,
            free_label=False,
            ticket_label=True,
            gold_icon=False,
            gold_amount=False,
            paid_confirmation=False,
        )
        self.assertTrue(is_safe_entry(safe))

    def test_any_paid_evidence_rejects_entry(self) -> None:
        for field in ("gold_icon", "gold_amount", "paid_confirmation"):
            values = {
                "tier": ConquestTier.INFINITE,
                "free_label": False,
                "ticket_label": True,
                "gold_icon": False,
                "gold_amount": False,
                "paid_confirmation": False,
            }
            values[field] = True
            with self.subTest(field=field):
                self.assertFalse(is_safe_entry(EntryEvidence(**values)))

    def test_conflicting_free_and_ticket_evidence_is_rejected(self) -> None:
        evidence = EntryEvidence(
            tier=ConquestTier.GOLD,
            free_label=True,
            ticket_label=True,
            gold_icon=False,
            gold_amount=False,
            paid_confirmation=False,
        )
        self.assertFalse(is_safe_entry(evidence))

    def test_missing_entry_evidence_is_rejected(self) -> None:
        evidence = EntryEvidence(
            tier=ConquestTier.SILVER,
            free_label=False,
            ticket_label=False,
            gold_icon=False,
            gold_amount=False,
            paid_confirmation=False,
        )
        self.assertFalse(is_safe_entry(evidence))
