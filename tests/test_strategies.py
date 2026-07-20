import random
import unittest

from agent.strategies.agatha import build_agatha_plan
from agent.strategies.ocr import CardCandidate, choose_card
from agent.strategies.random_play import (
    HAND_SLOTS,
    build_random_plan,
)


class StrategyTests(unittest.TestCase):
    def test_random_plan_attempts_each_hand_slot_twice(self) -> None:
        plan = build_random_plan(random.Random(42))
        self.assertEqual(len(plan.swipes), 8)
        for round_start in (0, 4):
            starts = plan.swipes[round_start : round_start + 4]
            nearest_slots = {
                min(
                    range(len(HAND_SLOTS)),
                    key=lambda index: abs(swipe.start.x - HAND_SLOTS[index].x),
                )
                for swipe in starts
            }
            self.assertEqual(nearest_slots, {0, 1, 2, 3})

    def test_random_plan_is_seeded_and_reproducible(self) -> None:
        first = build_random_plan(random.Random(7))
        second = build_random_plan(random.Random(7))
        different = build_random_plan(random.Random(8))
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_random_plan_stays_inside_safe_bounds(self) -> None:
        for seed in range(50):
            plan = build_random_plan(random.Random(seed))
            for swipe in plan.swipes:
                self.assertTrue(55 <= swipe.start.x <= 665)
                self.assertTrue(1020 <= swipe.start.y <= 1090)
                self.assertTrue(85 <= swipe.end.x <= 635)
                self.assertTrue(600 <= swipe.end.y <= 720)
                self.assertEqual(swipe.duration_ms, 350)

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

    def test_ocr_reports_no_energy(self) -> None:
        decision = choose_card(
            energy=0,
            cards=(CardCandidate(slot=0, cost=0, confidence=1.0),),
            minimum_confidence=0.80,
        )
        self.assertIsNone(decision.card)
        self.assertEqual(decision.reason, "no_energy")

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
