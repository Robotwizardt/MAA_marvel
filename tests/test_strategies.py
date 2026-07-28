import unittest

from agent.actions.play_turn import LANE_TARGETS
from agent.strategies.agatha import build_agatha_plan
from agent.strategies.ocr import CardCandidate, choose_card


class StrategyTests(unittest.TestCase):
    def test_lane_targets_stay_in_player_area_not_location_cards(self) -> None:
        self.assertEqual({point.x for point in LANE_TARGETS}, {145, 360, 575})
        for point in LANE_TARGETS:
            self.assertTrue(800 <= point.y <= 900)
            self.assertNotEqual(point.y, 650)

    def test_agatha_only_ends_turn(self) -> None:
        plan = build_agatha_plan()
        self.assertEqual(plan.swipes, ())
        self.assertTrue(plan.end_turn)

    def test_ocr_chooses_highest_affordable_cost(self) -> None:
        cards = (
            CardCandidate(slot=0, cost=1, confidence=0.95),
            CardCandidate(slot=1, cost=4, confidence=0.91),
            CardCandidate(slot=2, cost=3, confidence=0.93),
        )
        decision = choose_card(energy=3, cards=cards, minimum_confidence=0.80)
        self.assertIsNotNone(decision.card)
        self.assertEqual(decision.card.slot, 2)
        self.assertEqual(decision.reason, "selected")

    def test_ocr_tie_prefers_leftmost_slot(self) -> None:
        cards = (
            CardCandidate(slot=2, cost=3, confidence=0.99),
            CardCandidate(slot=0, cost=3, confidence=0.90),
            CardCandidate(slot=1, cost=3, confidence=0.95),
        )
        decision = choose_card(energy=3, cards=cards, minimum_confidence=0.80)
        self.assertIsNotNone(decision.card)
        self.assertEqual(decision.card.slot, 0)

    def test_ocr_can_play_zero_cost_but_rejects_negative_cost(self) -> None:
        decision = choose_card(
            energy=0,
            cards=(
                CardCandidate(slot=0, cost=-2, confidence=1.0),
                CardCandidate(slot=1, cost=0, confidence=1.0),
            ),
            minimum_confidence=0.80,
        )
        self.assertIsNotNone(decision.card)
        self.assertEqual(decision.card.slot, 1)

    def test_ocr_rejects_low_confidence_and_unaffordable_cards(self) -> None:
        low_confidence = choose_card(
            energy=5,
            cards=(CardCandidate(slot=0, cost=2, confidence=0.79),),
            minimum_confidence=0.80,
        )
        self.assertIsNone(low_confidence.card)
        self.assertEqual(low_confidence.reason, "low_confidence")

        unaffordable = choose_card(
            energy=2,
            cards=(CardCandidate(slot=0, cost=3, confidence=0.99),),
            minimum_confidence=0.80,
        )
        self.assertIsNone(unaffordable.card)
        self.assertEqual(unaffordable.reason, "no_affordable_card")

    def test_ocr_reports_missing_candidates(self) -> None:
        decision = choose_card(energy=5, cards=(), minimum_confidence=0.80)
        self.assertIsNone(decision.card)
        self.assertEqual(decision.reason, "no_candidates")
