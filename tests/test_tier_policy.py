import unittest

from agent.conquest.tier_policy import EntryEvidence, candidate_tiers, choose_tier, is_safe_entry
from agent.recognitions.safe_entry import parse_ticket_count
from agent.session.config import ConquestTier


class TierPolicyTests(unittest.TestCase):
    def test_candidates_descend_and_always_include_free_tier(self) -> None:
        self.assertEqual(
            candidate_tiers(ConquestTier.GOLD),
            (ConquestTier.GOLD, ConquestTier.SILVER, ConquestTier.PROVING_GROUNDS),
        )

    def test_highest_tier_must_exceed_its_reserve(self) -> None:
        selected = choose_tier(
            ConquestTier.INFINITE,
            {
                ConquestTier.INFINITE: 1,
                ConquestTier.GOLD: 3,
                ConquestTier.SILVER: 8,
            },
            {
                ConquestTier.INFINITE: 1,
                ConquestTier.GOLD: 2,
                ConquestTier.SILVER: 0,
            },
        )
        self.assertEqual(selected, ConquestTier.GOLD)

    def test_all_reserved_tickets_fall_back_to_proving_grounds(self) -> None:
        selected = choose_tier(
            ConquestTier.INFINITE,
            {ConquestTier.INFINITE: 1, ConquestTier.GOLD: 1, ConquestTier.SILVER: 1},
            {ConquestTier.INFINITE: 1, ConquestTier.GOLD: 1, ConquestTier.SILVER: 1},
        )
        self.assertEqual(selected, ConquestTier.PROVING_GROUNDS)

    def test_ticket_count_parser_accepts_observed_formats(self) -> None:
        self.assertEqual(parse_ticket_count(["6/1"]), 6)
        self.assertEqual(parse_ticket_count(["已拥有0/1"]), 0)
        self.assertIsNone(parse_ticket_count(["进入", "500"]))
        self.assertIsNone(parse_ticket_count(["2/1", "3/1"]))

    def test_free_tier_requires_free_evidence(self) -> None:
        evidence = EntryEvidence(
            tier=ConquestTier.PROVING_GROUNDS,
            free_label=True,
            ticket_count=None,
            reserve_count=0,
            gold_icon=False,
            gold_amount=False,
            paid_confirmation=False,
        )
        self.assertTrue(is_safe_entry(evidence))

    def test_paid_tier_requires_count_strictly_above_reserve(self) -> None:
        base = dict(
            tier=ConquestTier.GOLD,
            free_label=False,
            reserve_count=1,
            gold_icon=False,
            gold_amount=False,
            paid_confirmation=False,
        )
        self.assertTrue(is_safe_entry(EntryEvidence(ticket_count=2, **base)))
        self.assertFalse(is_safe_entry(EntryEvidence(ticket_count=1, **base)))
        self.assertFalse(is_safe_entry(EntryEvidence(ticket_count=0, **base)))

    def test_any_paid_currency_evidence_rejects_entry(self) -> None:
        for field in ("gold_icon", "gold_amount", "paid_confirmation"):
            values = {
                "tier": ConquestTier.INFINITE,
                "free_label": False,
                "ticket_count": 2,
                "reserve_count": 1,
                "gold_icon": False,
                "gold_amount": False,
                "paid_confirmation": False,
            }
            values[field] = True
            with self.subTest(field=field):
                self.assertFalse(is_safe_entry(EntryEvidence(**values)))


if __name__ == "__main__":
    unittest.main()
