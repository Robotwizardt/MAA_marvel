from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

from agent.actions.play_turn import (
    LANE_TARGETS,
    PlayTurn,
    _close_detail_overlay,
    _confirm_play_from_fresh_frames,
    _is_detail_overlay,
    _play_succeeded,
    _recognition_hit,
    _scan_with_retry,
    _wait_for_agatha,
    _wait_for_play_resolution,
    lane_targets_for_order,
)
from agent.recognitions.card_selection import (
    BattleHand,
    BLUE_BADGE_SCAN_ROI,
    CardSelection,
    DetectedCard,
    ENERGY_DIGIT_ROI,
    HandFrameTracker,
    HAND_COST_ROI,
    HAND_COST_WINDOWS,
    _build_cost_probe_rois,
    _detect_cards,
    _find_blue_cost_badges,
    _tight_blue_badge,
    is_active_turn,
    is_active_turn_frame,
    _results,
    parse_digit_results,
    scan_battle_hand,
)
from agent.recognitions.lane_state import LaneState
from agent.recognitions.session_gate import SessionGate
from agent.runtime.store import STORE
from agent.session.config import LaneOrder


def ocr_result(text: str, score: float = 0.95, box=(10, 10, 20, 20)):
    return SimpleNamespace(text=text, score=score, box=box)


def recognition_detail(*results):
    return SimpleNamespace(
        filtered_results=list(results),
        all_results=list(results),
    )


def card(slot: int, cost: int, x: int) -> DetectedCard:
    return DetectedCard(slot, cost, 0.95, (x - 45, 885, 90, 90))


def mark_active_turn(image: np.ndarray) -> np.ndarray:
    image[960:1030, 1670:1850] = (200, 20, 150)
    return image


class FakeDirectContext:
    def __init__(self, details: list[object | None]) -> None:
        self.details = list(details)
        self.reco_params: list[object] = []

    def run_recognition_direct(self, reco_type, reco_param, image):
        self.reco_params.append(reco_param)
        return self.details.pop(0)


class FakeJob:
    succeeded = True

    def wait(self):
        return self


class FakeScreenshotJob:
    def get(self, wait=False):
        return object()


class FakeController:
    def __init__(self) -> None:
        self.swipes: list[tuple[int, int, int, int, int]] = []
        self.clicks: list[tuple[int, int]] = []

    def post_screencap(self):
        return FakeScreenshotJob()

    def post_swipe(self, x1, y1, x2, y2, duration):
        self.swipes.append((x1, y1, x2, y2, duration))
        return FakeJob()

    def post_click(self, x, y):
        self.clicks.append((x, y))
        return FakeJob()


class FakePlayContext:
    def __init__(self, detail_match=None) -> None:
        self.controller = FakeController()
        self.tasker = SimpleNamespace(controller=self.controller)
        self.detail_match = detail_match
        self.recognition_entries: list[str] = []
        self.next_overrides: list[tuple[str, list[str]]] = []

    def run_recognition(self, entry, image):
        self.last_recognition_entry = entry
        self.recognition_entries.append(entry)
        if isinstance(self.detail_match, dict):
            matches = self.detail_match.get(entry)
            if isinstance(matches, list):
                return matches.pop(0)
            return matches
        if isinstance(self.detail_match, list):
            return self.detail_match.pop(0)
        return self.detail_match

    def override_next(self, node_name, next_list):
        self.next_overrides.append((node_name, list(next_list)))
        return True


class OcrAdapterTests(unittest.TestCase):
    def test_ocr_uses_raw_digit_when_expected_filter_is_empty(self) -> None:
        detail = SimpleNamespace(
            filtered_results=[],
            all_results=[ocr_result("2", score=0.97)],
        )
        parsed = parse_digit_results(_results(detail))
        self.assertEqual(parsed.value, 2)
        self.assertEqual(parsed.confidence, 0.97)

    def test_detail_overlay_requires_native_landscape_frame(self) -> None:
        overlay = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertTrue(_is_detail_overlay(overlay))

        portrait = np.zeros((1920, 1080, 3), dtype=np.uint8)
        self.assertFalse(_is_detail_overlay(portrait))

    def test_detail_overlay_click_requires_template_confirmation(self) -> None:
        overlay = np.zeros((1080, 1920, 3), dtype=np.uint8)

        rejected = FakePlayContext(detail_match=None)
        self.assertFalse(_close_detail_overlay(rejected, rejected.controller, overlay))
        self.assertEqual(rejected.controller.clicks, [])

        confirmed = FakePlayContext(
            detail_match=SimpleNamespace(hit=True, box=(20, 950, 120, 60))
        )
        with patch("agent.actions.play_turn.time.sleep"):
            self.assertTrue(
                _close_detail_overlay(confirmed, confirmed.controller, overlay)
            )
        self.assertEqual(confirmed.last_recognition_entry, "公共-详情关闭按钮")
        self.assertEqual(confirmed.controller.clicks, [(80, 980)])

    def test_lane_order_supports_all_three_modes(self) -> None:
        self.assertEqual(
            lane_targets_for_order(LaneOrder.LEFT_TO_RIGHT),
            LANE_TARGETS,
        )
        self.assertEqual(
            lane_targets_for_order(LaneOrder.RIGHT_TO_LEFT),
            tuple(reversed(LANE_TARGETS)),
        )
        with patch(
            "agent.actions.play_turn.random.sample",
            return_value=list(reversed(LANE_TARGETS)),
        ):
            self.assertEqual(
                lane_targets_for_order(LaneOrder.RANDOM),
                tuple(reversed(LANE_TARGETS)),
            )

    def test_parse_energy_supports_modified_two_digit_values(self) -> None:
        self.assertEqual(parse_digit_results([ocr_result("3")]).value, 3)
        self.assertEqual(parse_digit_results([ocr_result("9")]).value, 9)
        self.assertEqual(parse_digit_results([ocr_result("12")]).value, 12)
        self.assertEqual(parse_digit_results([ocr_result("20")]).value, 20)
        self.assertIsNone(
            parse_digit_results([ocr_result("2"), ocr_result("3")]).value
        )
        self.assertIsNone(parse_digit_results([ocr_result("21")]).value)

    def test_ocr_scan_reads_energy_then_only_ocr_blue_badge(self) -> None:
        image = mark_active_turn(np.zeros((1080, 1920, 3), dtype=np.uint8))
        image[850:900, 680:701] = (255, 40, 25)
        context = FakeDirectContext(
            [
                recognition_detail(ocr_result("1")),
                recognition_detail(ocr_result("1", box=(20, 25, 20, 20))),
            ]
        )
        hand = scan_battle_hand(context, image)
        self.assertEqual(hand.energy, 1)
        self.assertEqual([card.cost for card in hand.cards], [1])
        self.assertEqual(len(context.reco_params), 2)
        self.assertEqual(context.reco_params[0].roi, ENERGY_DIGIT_ROI)
        self.assertTrue(context.reco_params[0].only_rec)
        self.assertEqual(context.reco_params[1].roi, (668, 840, 44, 60))
        self.assertFalse(context.reco_params[1].only_rec)

    def test_hand_scan_records_energy_badge_and_card_ocr_timings(self) -> None:
        image = mark_active_turn(np.zeros((1080, 1920, 3), dtype=np.uint8))
        image[850:900, 680:701] = (255, 40, 25)
        context = FakeDirectContext(
            [
                recognition_detail(ocr_result("1")),
                recognition_detail(ocr_result("1", box=(20, 25, 20, 20))),
            ]
        )
        timings: list[tuple[str, float]] = []

        hand = scan_battle_hand(
            context,
            image,
            lambda name, elapsed: timings.append((name, elapsed)),
        )

        self.assertEqual([card.cost for card in hand.cards], [1])
        names = [name for name, _ in timings]
        self.assertIn("hand_scan.active_turn", names)
        self.assertIn("hand_scan.energy_ocr", names)
        self.assertIn("hand_scan.badge_color", names)
        self.assertIn("hand_scan.cost_ocr", names)
        self.assertEqual(names[-1], "hand_scan.total")
        self.assertTrue(all(elapsed >= 0.0 for _, elapsed in timings))

    def test_blue_badge_scan_ignores_orange_power_and_dark_cost(self) -> None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image[850:900, 680:701] = (255, 40, 25)
        image[850:900, 820:841] = (30, 100, 245)
        image[850:900, 960:981] = (55, 50, 45)
        badges = _find_blue_cost_badges(image)
        self.assertEqual([badge.center_x for badge in badges], [690])
        self.assertEqual(BLUE_BADGE_SCAN_ROI, (500, 850, 930, 50))

    def test_blue_badge_scan_matches_real_landscape_fixture(self) -> None:
        fixture = (
            Path(__file__).parent
            / "fixtures"
            / "screens"
            / "battle"
            / "normal_turn.png"
        )
        rgb = np.asarray(Image.open(fixture).convert("RGB"))
        badges = _find_blue_cost_badges(rgb[:, :, ::-1])
        self.assertEqual(len(badges), 1)
        self.assertTrue(820 <= badges[0].center_x <= 840)

    def test_multiple_blue_badges_are_each_ocr_scanned(self) -> None:
        image = mark_active_turn(np.zeros((1080, 1920, 3), dtype=np.uint8))
        image[850:900, 680:701] = (255, 40, 25)
        image[850:900, 820:841] = (255, 40, 25)
        context = FakeDirectContext(
            [
                recognition_detail(ocr_result("3")),
                recognition_detail(ocr_result("1", box=(20, 25, 20, 20))),
                recognition_detail(ocr_result("3", box=(20, 25, 20, 20))),
            ]
        )
        hand = scan_battle_hand(context, image)
        self.assertEqual([card.cost for card in hand.cards], [1, 3])
        self.assertEqual(hand.reason, "blue_badge_path")

    def test_blue_badge_cost_above_energy_is_retried_with_tight_roi(self) -> None:
        image = mark_active_turn(np.zeros((1080, 1920, 3), dtype=np.uint8))
        image[850:900, 680:701] = (255, 40, 25)
        context = FakeDirectContext(
            [
                recognition_detail(ocr_result("5")),
                recognition_detail(ocr_result("12", box=(0, 0, 44, 42))),
                recognition_detail(ocr_result("2", box=(2, 0, 28, 32))),
            ]
        )
        hand = scan_battle_hand(context, image)
        self.assertEqual([card.cost for card in hand.cards], [2])
        self.assertEqual(
            context.reco_params[2].roi,
            _tight_blue_badge(_find_blue_cost_badges(image)[0]).roi,
        )

    def test_card_detection_accepts_zero_to_twenty_only(self) -> None:
        detected = _detect_cards(
            [
                (ocr_result("-2", box=(690, 850, 20, 20)), HAND_COST_ROI),
                (ocr_result("0", box=(790, 850, 20, 20)), HAND_COST_ROI),
                (ocr_result("10", box=(890, 850, 20, 20)), HAND_COST_ROI),
                (ocr_result("21", box=(990, 850, 20, 20)), HAND_COST_ROI),
            ],
            image=object(),
        )
        self.assertEqual([item.cost for item in detected], [0, 10])

    def test_narrow_seven_is_corrected_to_one_cost(self) -> None:
        detected = _detect_cards(
            [
                (ocr_result("7", score=0.67, box=(745, 850, 6, 10)), HAND_COST_ROI),
                (ocr_result("7", score=0.99, box=(845, 850, 16, 20)), HAND_COST_ROI),
            ],
            image=object(),
        )
        self.assertEqual([item.cost for item in detected], [1, 7])

    def test_power_digit_builds_narrow_probe_over_cost_badge_to_its_left(self) -> None:
        rois = _build_cost_probe_rois(
            [(ocr_result("9", box=(901, 850, 19, 20)), HAND_COST_ROI)]
        )
        self.assertEqual(rois, ((730, 815, 150, 90),))

    def test_card_selection_chooses_highest_affordable_card(self) -> None:
        image = mark_active_turn(np.zeros((1080, 1920, 3), dtype=np.uint8))
        for center_x in (690, 830, 970):
            image[850:900, center_x - 10 : center_x + 11] = (255, 40, 25)
        argv = SimpleNamespace(image=image)
        result = CardSelection().analyze(
            FakeDirectContext(
                [
                    recognition_detail(ocr_result("3")),
                    recognition_detail(ocr_result("1", box=(20, 25, 20, 20))),
                    recognition_detail(ocr_result("4", box=(20, 25, 20, 20))),
                    recognition_detail(ocr_result("4", box=(2, 0, 28, 32))),
                    recognition_detail(ocr_result("3", box=(20, 25, 20, 20))),
                ]
            ),
            argv,
        )
        self.assertEqual(result.detail["energy"], 3)
        self.assertEqual(result.detail["selected_slot"], 1)
        self.assertEqual(result.box, (953, 885, 90, 90))

    def test_ocr_card_scan_normalizes_relative_boxes(self) -> None:
        image = mark_active_turn(np.zeros((1080, 1920, 3), dtype=np.uint8))
        image[850:900, 680:701] = (255, 40, 25)
        image[850:900, 820:841] = (255, 40, 25)
        context = FakeDirectContext(
            [
                recognition_detail(ocr_result("4")),
                recognition_detail(ocr_result("2", box=(20, 25, 20, 20))),
                recognition_detail(ocr_result("3", box=(20, 25, 20, 20))),
            ]
        )
        hand = scan_battle_hand(context, image)
        self.assertEqual([card.cost for card in hand.cards], [2, 3])

    def test_unchanged_hand_reuses_cost_ocr_across_frames(self) -> None:
        image = mark_active_turn(np.zeros((1080, 1920, 3), dtype=np.uint8))
        for center_x in (690, 830):
            image[850:900, center_x - 10 : center_x + 11] = (255, 40, 25)
        tracker = HandFrameTracker()
        first = FakeDirectContext(
            [
                recognition_detail(ocr_result("3")),
                recognition_detail(ocr_result("1", box=(20, 25, 20, 20))),
                recognition_detail(ocr_result("2", box=(20, 25, 20, 20))),
            ]
        )
        second = FakeDirectContext([recognition_detail(ocr_result("3"))])

        initial = scan_battle_hand(first, image, tracker=tracker)
        reused = scan_battle_hand(second, image.copy(), tracker=tracker)

        self.assertEqual([item.cost for item in initial.cards], [1, 2])
        self.assertEqual([item.cost for item in reused.cards], [1, 2])
        self.assertEqual(reused.reason, "tracked_blue_badge_path")
        self.assertEqual(len(second.reco_params), 1)

    def test_card_outline_change_forces_full_cost_ocr(self) -> None:
        image = mark_active_turn(np.zeros((1080, 1920, 3), dtype=np.uint8))
        for center_x in (690, 830):
            image[850:900, center_x - 10 : center_x + 11] = (255, 40, 25)
        tracker = HandFrameTracker()
        first = FakeDirectContext(
            [
                recognition_detail(ocr_result("3")),
                recognition_detail(ocr_result("1", box=(20, 25, 20, 20))),
                recognition_detail(ocr_result("2", box=(20, 25, 20, 20))),
            ]
        )
        scan_battle_hand(first, image, tracker=tracker)
        rearranged = image.copy()
        rearranged[902:940, 680:750] = 230
        second = FakeDirectContext(
            [
                recognition_detail(ocr_result("3")),
                recognition_detail(ocr_result("2", box=(20, 25, 20, 20))),
                recognition_detail(ocr_result("1", box=(20, 25, 20, 20))),
            ]
        )

        hand = scan_battle_hand(second, rearranged, tracker=tracker)

        self.assertEqual(hand.reason, "blue_badge_path")
        self.assertEqual([item.cost for item in hand.cards], [2, 1])
        self.assertEqual(len(second.reco_params), 3)

    def test_cost_region_change_forces_full_cost_ocr(self) -> None:
        image = mark_active_turn(np.zeros((1080, 1920, 3), dtype=np.uint8))
        for center_x in (690, 830):
            image[850:900, center_x - 10 : center_x + 11] = (255, 40, 25)
        tracker = HandFrameTracker()
        first = FakeDirectContext(
            [
                recognition_detail(ocr_result("3")),
                recognition_detail(ocr_result("1", box=(20, 25, 20, 20))),
                recognition_detail(ocr_result("2", box=(20, 25, 20, 20))),
            ]
        )
        scan_battle_hand(first, image, tracker=tracker)
        modified = image.copy()
        modified[842:849, 672:708] = 255
        second = FakeDirectContext(
            [
                recognition_detail(ocr_result("3")),
                recognition_detail(ocr_result("2", box=(20, 25, 20, 20))),
                recognition_detail(ocr_result("1", box=(20, 25, 20, 20))),
            ]
        )

        hand = scan_battle_hand(second, modified, tracker=tracker)

        self.assertEqual(hand.reason, "blue_badge_path")
        self.assertEqual([item.cost for item in hand.cards], [2, 1])
        self.assertEqual(len(second.reco_params), 3)

    def test_card_detection_keeps_blue_cost_and_rejects_orange_power(self) -> None:
        """OCR 同时读到费用和战力时，只保留蓝色费用徽标。"""
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image[838:884, 678:724] = (200, 70, 20)  # BGR 亮起的费用徽标
        image[838:884, 778:824] = (60, 65, 70)  # BGR 暗下的费用徽标

        detected = _detect_cards(
            [
                (ocr_result("1", box=(690, 850, 20, 20)), HAND_COST_ROI),
                (ocr_result("5", box=(790, 850, 20, 20)), HAND_COST_ROI),
            ],
            image=image,
        )

        self.assertEqual([(item.cost, item.box[0]) for item in detected], [(1, 683)])

    def test_orange_power_does_not_borrow_blue_from_the_next_card(self) -> None:
        """重叠手牌中，相邻费用的蓝色不能让前一张牌的战力通过校验。"""
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image[840:882, 894:926] = (20, 90, 210)  # BGR 橙色战力徽标
        image[840:882, 928:970] = (200, 70, 20)  # 下一张牌的蓝色费用徽标

        detected = _detect_cards(
            [
                (ocr_result("9", box=(900, 850, 20, 20)), HAND_COST_ROI),
                (ocr_result("1", box=(935, 850, 20, 20)), HAND_COST_ROI),
            ],
            image=image,
        )

        self.assertEqual([item.cost for item in detected], [1])

    def test_wide_power_box_still_cannot_reach_the_next_blue_badge(self) -> None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image[840:882, 894:928] = (20, 90, 210)
        image[840:882, 930:974] = (200, 70, 20)

        detected = _detect_cards(
            [
                (ocr_result("9", box=(900, 850, 30, 20)), HAND_COST_ROI),
                (ocr_result("3", box=(938, 850, 20, 20)), HAND_COST_ROI),
            ],
            image=image,
        )

        self.assertEqual([item.cost for item in detected], [3])

    def test_landscape_power_next_cost_pairs_keep_the_right_digit(self) -> None:
        detected = _detect_cards(
            [
                (ocr_result("4", box=(728, 850, 20, 20)), HAND_COST_ROI),
                (ocr_result("2", box=(758, 850, 20, 20)), HAND_COST_ROI),
                (ocr_result("2", box=(866, 850, 20, 20)), HAND_COST_ROI),
                (ocr_result("3", box=(898, 850, 20, 20)), HAND_COST_ROI),
                (ocr_result("4", box=(1010, 850, 20, 20)), HAND_COST_ROI),
                (ocr_result("3", box=(1042, 850, 20, 20)), HAND_COST_ROI),
            ],
            image=object(),
        )
        self.assertEqual([item.cost for item in detected], [2, 3, 3])
        self.assertEqual([item.box[0] for item in detected], [751, 891, 1035])

    def test_ocr_play_repeats_highest_affordable_until_none_remains(self) -> None:
        STORE.configure({"play_strategy": "ocr"}, now=0.0)
        context = FakePlayContext()
        hand_3 = BattleHand(3, (card(0, 1, 180), card(1, 2, 420)), "recognized")
        hand_1 = BattleHand(1, (card(0, 1, 300),), "recognized")
        hand_0 = BattleHand(0, (), "no_cards")
        # 第二张牌打出后的确认扫描和下一轮结束判断都会完整复查四帧。
        scans = [hand_3, hand_1, hand_1, *([hand_0] * 8)]
        with (
            patch("agent.actions.play_turn.scan_battle_hand", side_effect=scans),
            patch(
                "agent.actions.play_turn._wait_for_play_resolution",
                return_value="ready",
            ),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 2)
        self.assertEqual(context.controller.swipes[0][:2], (420, 930))
        self.assertEqual(context.controller.swipes[1][:2], (300, 930))

    def test_scan_retries_multiple_frames_until_bright_card_is_found(self) -> None:
        context = FakePlayContext()
        no_bright_card = BattleHand(1, (), "no_blue_badges")
        bright_card = BattleHand(1, (card(0, 1, 300),), "recognized")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                side_effect=[no_bright_card, bright_card],
            ),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            result = _scan_with_retry(context, context.controller)
        self.assertEqual(result.cards[0].cost, 1)

    def test_inactive_turn_is_retried_before_leaving_play_phase(self) -> None:
        context = FakePlayContext()
        inactive = BattleHand(None, (), "inactive_turn")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                return_value=inactive,
            ) as scan,
            patch("agent.actions.play_turn.time.sleep") as sleep,
        ):
            result = _scan_with_retry(context, context.controller)
        self.assertEqual(result.reason, "inactive_turn")
        self.assertGreater(scan.call_count, 1)
        self.assertEqual(sleep.call_count, scan.call_count - 1)

    def test_inactive_turn_recovers_to_active_hand(self) -> None:
        context = FakePlayContext()
        inactive = BattleHand(None, (), "inactive_turn")
        active = BattleHand(1, (card(0, 1, 300),), "recognized")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                side_effect=[inactive, active],
            ) as scan,
            patch("agent.actions.play_turn.time.sleep") as sleep,
        ):
            result = _scan_with_retry(context, context.controller)
        self.assertEqual(result, active)
        self.assertEqual(scan.call_count, 2)
        sleep.assert_called_once()

    def test_inactive_turn_bypasses_snap_and_routes_to_state_recovery(self) -> None:
        STORE.configure({"play_strategy": "ocr"}, now=0.0)
        context = FakePlayContext()
        inactive = BattleHand(None, (), "inactive_turn")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                return_value=inactive,
            ),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(
                PlayTurn().run(
                    context,
                    SimpleNamespace(node_name="公共-执行出牌"),
                )
            )
        self.assertEqual(
            context.next_overrides,
            [("公共-执行出牌", ["公共-出牌后状态"])],
        )

    def test_final_turn_placing_transition_still_plays_remaining_card(self) -> None:
        STORE.configure({"play_strategy": "ocr"}, now=0.0)
        context = FakePlayContext()
        inactive = BattleHand(None, (), "inactive_turn")
        playable = BattleHand(1, (card(0, 1, 300),), "recognized")
        empty = BattleHand(0, (), "energy_zero")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                side_effect=[inactive, inactive, playable, empty],
            ),
            patch(
                "agent.actions.play_turn._wait_for_play_resolution",
                return_value="placed",
            ),
            patch(
                "agent.actions.play_turn._confirm_play_from_fresh_frames",
                return_value=True,
            ),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 1)

    def test_positive_cost_requires_energy_drop_not_ocr_reflow(self) -> None:
        before = BattleHand(1, (card(0, 1, 300),), "recognized")
        shifted = BattleHand(1, (card(0, 1, 420),), "recognized")
        self.assertFalse(_play_succeeded(before, shifted, played_cost=1))

    def test_zero_cost_can_use_hand_count_drop(self) -> None:
        before = BattleHand(1, (card(0, 0, 300),), "recognized")
        after = BattleHand(1, (), "no_cards")
        self.assertTrue(_play_succeeded(before, after, played_cost=0))

    def test_zero_cost_can_use_replaced_candidate_with_same_count(self) -> None:
        before = BattleHand(1, (card(0, 0, 300),), "recognized")
        after = BattleHand(1, (card(0, 1, 430),), "recognized")
        self.assertTrue(_play_succeeded(before, after, played_cost=0))

    def test_unchanged_zero_cost_candidate_is_not_success(self) -> None:
        before = BattleHand(1, (card(0, 0, 300),), "recognized")
        unchanged = BattleHand(1, (card(0, 0, 320),), "recognized")
        self.assertFalse(_play_succeeded(before, unchanged, played_cost=0))

    def test_agatha_waits_only_for_end_turn_without_scanning_cards(self) -> None:
        context = FakePlayContext(
            detail_match=SimpleNamespace(hit=True, box=(550, 1149, 144, 55))
        )
        with patch("agent.actions.play_turn.scan_battle_hand") as scan:
            self.assertTrue(_wait_for_agatha(context, context.controller))
        scan.assert_not_called()
        self.assertEqual(context.last_recognition_entry, "公共-结束回合")

    def test_recognition_miss_with_box_is_not_a_hit(self) -> None:
        self.assertFalse(
            _recognition_hit(
                SimpleNamespace(hit=False, box=(550, 1149, 144, 55))
            )
        )

    def test_play_resolution_requires_stable_end_turn_without_placing(self) -> None:
        context = FakePlayContext(
            detail_match={
                "公共-放置中状态": [
                    SimpleNamespace(hit=False, box=(0, 0, 0, 0))
                    for _ in range(4)
                ],
                "公共-结束回合": [
                    SimpleNamespace(hit=True, box=(550, 950, 144, 55))
                    for _ in range(4)
                ],
            }
        )
        with patch("agent.actions.play_turn.time.sleep"):
            self.assertEqual(
                _wait_for_play_resolution(context, context.controller),
                "ready",
            )
        self.assertEqual(
            context.recognition_entries,
            [
                "公共-放置中状态",
                "公共-结束回合",
                "公共-放置中状态",
                "公共-结束回合",
                "公共-放置中状态",
                "公共-结束回合",
                "公共-放置中状态",
                "公共-结束回合",
            ],
        )
        self.assertEqual(context.controller.clicks, [])

    def test_play_resolution_waits_until_placing_disappears(self) -> None:
        context = FakePlayContext(
            detail_match={
                "公共-放置中状态": [
                    SimpleNamespace(hit=True, box=(1600, 930, 150, 60)),
                    SimpleNamespace(hit=False, box=(0, 0, 0, 0)),
                ],
                "公共-结束回合": [
                    SimpleNamespace(hit=True, box=(1600, 930, 180, 60)),
                ],
            }
        )
        with patch("agent.actions.play_turn.time.sleep"):
            self.assertEqual(
                _wait_for_play_resolution(context, context.controller),
                "placed",
            )
        self.assertEqual(
            context.recognition_entries,
            ["公共-放置中状态", "公共-放置中状态", "公共-结束回合"],
        )

    def test_play_resolution_keeps_placing_success_when_animation_times_out(self) -> None:
        context = FakePlayContext(
            detail_match={
                "公共-放置中状态": [
                    SimpleNamespace(hit=True, box=(1600, 930, 150, 60)),
                ],
            }
        )
        with (
            patch("agent.actions.play_turn.time.monotonic", side_effect=[0.0, 0.0, 9.0]),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertEqual(
                _wait_for_play_resolution(context, context.controller),
                "placed",
            )
        self.assertEqual(context.recognition_entries, ["公共-放置中状态"])

    def test_failed_expensive_card_falls_back_to_cheaper_card(self) -> None:
        STORE.configure(
            {
                "play_strategy": "ocr",
                "snap_mode": "off",
            },
            now=0.0,
        )
        context = FakePlayContext(
            detail_match=SimpleNamespace(box=(550, 1149, 144, 55))
        )
        initial = BattleHand(
            3,
            (card(0, 3, 200), card(1, 1, 430)),
            "recognized",
        )
        after = BattleHand(2, (card(0, 3, 250),), "recognized")
        no_card = BattleHand(2, (card(0, 3, 250),), "recognized")
        # 3 费牌三个场地均失败，随后 1 费牌成功；下一轮确认无可支付牌。
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                side_effect=[initial, no_card, no_card],
            ),
            patch(
                "agent.actions.play_turn._wait_for_play_resolution",
                return_value="ready",
            ),
            patch(
                "agent.actions.play_turn._confirm_play_from_fresh_frames",
                side_effect=[False, False, False, True],
            ),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 4)
        self.assertEqual(context.controller.swipes[-1][:2], (430, 930))
        self.assertEqual(context.controller.clicks, [])

    def test_positive_energy_without_blue_badges_waits_for_turn_animation(self) -> None:
        context = FakePlayContext()
        no_bright_card = BattleHand(1, (), "no_blue_badges")
        with patch(
            "agent.actions.play_turn.scan_battle_hand",
            return_value=no_bright_card,
        ) as scan:
            result = _scan_with_retry(context, context.controller)
        self.assertEqual(result.energy, no_bright_card.energy)
        self.assertEqual(result.cards, no_bright_card.cards)
        self.assertEqual(result.reason, no_bright_card.reason)
        self.assertEqual(scan.call_count, 4)
        self.assertEqual(result.no_badge_frames, 4)

    def test_zero_energy_opens_end_turn_gate_without_lane_scan(self) -> None:
        state = STORE.configure({"play_strategy": "ocr"}, now=0.0)
        context = FakePlayContext()
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                return_value=BattleHand(0, (), "energy_zero"),
            ),
            patch("agent.actions.play_turn.scan_lane_states") as lane_scan,
        ):
            self.assertTrue(
                PlayTurn().run(
                    context,
                    SimpleNamespace(node_name="公共-执行出牌"),
                )
            )

        lane_scan.assert_not_called()
        self.assertTrue(state.end_turn_allowed)
        self.assertEqual(state.end_turn_reason, "energy_zero")
        self.assertEqual(context.controller.swipes, [])

    def test_four_no_badge_frames_open_end_turn_gate(self) -> None:
        state = STORE.configure({"play_strategy": "ocr"}, now=0.0)
        state.begin_turn(1)
        context = FakePlayContext()
        no_badge = BattleHand(2, (), "no_blue_badges")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                return_value=no_badge,
            ) as scan,
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(
                PlayTurn().run(
                    context,
                    SimpleNamespace(node_name="公共-执行出牌"),
                )
            )

        self.assertEqual(scan.call_count, 4)
        self.assertTrue(state.end_turn_allowed)
        self.assertEqual(
            state.end_turn_reason,
            "stable_no_highlighted_badges",
        )

    def test_final_turn_no_badge_frames_do_not_open_end_turn_gate(self) -> None:
        state = STORE.configure({"play_strategy": "ocr"}, now=0.0)
        state.begin_match()
        state.begin_turn(6)
        context = FakePlayContext()
        no_badge = BattleHand(6, (), "no_blue_badges")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                return_value=no_badge,
            ) as scan,
            patch("agent.actions.play_turn.scan_lane_states", return_value=()),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(
                PlayTurn().run(
                    context,
                    SimpleNamespace(node_name="鍏叡-鎵ц鍑虹墝"),
                )
            )

        self.assertEqual(scan.call_count, 4)
        self.assertFalse(state.end_turn_allowed)
        self.assertTrue(context.next_overrides)

    def test_all_prechecked_full_lanes_skip_swipes_and_open_gate(self) -> None:
        state = STORE.configure({"play_strategy": "ocr"}, now=0.0)
        context = FakePlayContext()
        hand = BattleHand(2, (card(0, 2, 300),), "blue_badge_path")
        full = tuple(
            LaneState(x, (True, True, True, True), (2.0, 2.0, 2.0, 2.0))
            for x in (710, 960, 1210)
        )
        with (
            patch("agent.actions.play_turn.scan_battle_hand", return_value=hand),
            patch("agent.actions.play_turn.scan_lane_states", return_value=full),
        ):
            self.assertTrue(
                PlayTurn().run(
                    context,
                    SimpleNamespace(node_name="公共-执行出牌"),
                )
            )

        self.assertEqual(context.controller.swipes, [])
        self.assertTrue(state.end_turn_allowed)
        self.assertEqual(state.end_turn_reason, "all_lanes_full")

    def test_unresolved_highlighted_badge_blocks_failure_gate(self) -> None:
        state = STORE.configure({"play_strategy": "ocr"}, now=0.0)
        context = FakePlayContext()
        hand = BattleHand(
            2,
            (card(0, 2, 300),),
            "blue_badge_path",
            unresolved_badges=1,
        )
        full = tuple(
            LaneState(x, (True, True, True, True), (2.0, 2.0, 2.0, 2.0))
            for x in (710, 960, 1210)
        )
        with (
            patch("agent.actions.play_turn.scan_battle_hand", return_value=hand),
            patch("agent.actions.play_turn.scan_lane_states", return_value=full),
        ):
            self.assertTrue(
                PlayTurn().run(
                    context,
                    SimpleNamespace(node_name="公共-执行出牌"),
                )
            )

        self.assertFalse(state.end_turn_allowed)
        self.assertEqual(
            context.next_overrides,
            [("公共-执行出牌", ["公共-出牌后状态"])],
        )

    def test_end_turn_custom_gate_requires_permission_and_active_frame(self) -> None:
        state = STORE.configure({"play_strategy": "ocr"}, now=0.0)
        active = mark_active_turn(np.zeros((1080, 1920, 3), dtype=np.uint8))
        self.assertIs(type(is_active_turn_frame(active)), bool)
        context = FakePlayContext(
            detail_match=SimpleNamespace(
                hit=True,
                box=(1580, 900, 340, 180),
            )
        )
        argv = SimpleNamespace(
            custom_recognition_param='{"command":"can_end_turn"}',
            image=active,
        )

        denied = SessionGate().analyze(context, argv)
        state.allow_end_turn("energy_zero")
        allowed = SessionGate().analyze(context, argv)
        inactive = SessionGate().analyze(
            FakePlayContext(
                detail_match=SimpleNamespace(hit=False, box=None)
            ),
            SimpleNamespace(
                custom_recognition_param='{"command":"can_end_turn"}',
                image=active,
            ),
        )

        self.assertIsNone(denied.box)
        self.assertIsNotNone(allowed.box)
        self.assertIsNone(inactive.box)

    def test_active_turn_uses_text_to_separate_ready_battle_button(self) -> None:
        purple_button = mark_active_turn(
            np.zeros((1080, 1920, 3), dtype=np.uint8)
        )
        ready_battle = FakePlayContext(
            detail_match=SimpleNamespace(hit=False, box=None)
        )
        end_turn = FakePlayContext(
            detail_match=SimpleNamespace(
                hit=True,
                box=(1580, 900, 340, 180),
            )
        )

        self.assertFalse(is_active_turn(ready_battle, purple_button))
        self.assertTrue(
            is_active_turn(
                end_turn,
                np.zeros((1080, 1920, 3), dtype=np.uint8),
            )
        )
        self.assertEqual(
            ready_battle.last_recognition_entry,
            "公共-结束回合文字",
        )

    def test_full_field_tries_each_lane_once_then_stops(self) -> None:
        STORE.configure({"play_strategy": "ocr"}, now=0.0)
        context = FakePlayContext()
        unchanged = BattleHand(2, (card(0, 2, 300),), "recognized")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                return_value=unchanged,
            ),
            patch(
                "agent.actions.play_turn._wait_for_play_resolution",
                return_value="ready",
            ),
            patch(
                "agent.actions.play_turn._confirm_play_from_fresh_frames",
                return_value=False,
            ),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 3)
        self.assertEqual(
            {swipe[2:4] for swipe in context.controller.swipes},
            {(point.x, point.y) for point in LANE_TARGETS},
        )

    def test_detail_overlay_does_not_count_as_lane_failure(self) -> None:
        STORE.configure({"play_strategy": "ocr"}, now=0.0)
        context = FakePlayContext()
        hand = BattleHand(
            3,
            (card(0, 3, 300), card(1, 2, 430), card(2, 1, 560)),
            "recognized",
        )
        with (
            patch("agent.actions.play_turn.scan_battle_hand", return_value=hand),
            patch(
                "agent.actions.play_turn._wait_for_play_resolution",
                return_value="detail",
            ),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 9)
        self.assertEqual(
            [swipe[2:4] for swipe in context.controller.swipes],
            [(point.x, point.y) for point in LANE_TARGETS] * 3,
        )

    def test_placing_signal_confirms_zero_cost_even_when_ocr_is_stale(self) -> None:
        STORE.configure({"play_strategy": "ocr"}, now=0.0)
        context = FakePlayContext()
        zero = BattleHand(1, (card(0, 0, 300),), "recognized")
        no_card = BattleHand(1, (), "no_blue_badges")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                side_effect=[zero, no_card, no_card, no_card, no_card],
            ),
            patch(
                "agent.actions.play_turn._wait_for_play_resolution",
                return_value="placed",
            ),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 1)

    def test_placing_signal_confirms_positive_cost_even_when_ocr_is_stale(self) -> None:
        context = FakePlayContext()
        before = BattleHand(3, (card(0, 3, 300),), "recognized")
        stale = BattleHand(3, (card(0, 3, 300),), "recognized")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                return_value=stale,
            ) as scan,
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(
                _confirm_play_from_fresh_frames(
                    context,
                    context.controller,
                    before,
                    3,
                    "placed",
                )
            )
        scan.assert_not_called()

    def test_ready_rejection_uses_bounded_fresh_frame_checks(self) -> None:
        context = FakePlayContext()
        unchanged = BattleHand(2, (card(0, 2, 300),), "recognized")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                return_value=unchanged,
            ) as scan,
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertFalse(
                _confirm_play_from_fresh_frames(
                    context,
                    context.controller,
                    unchanged,
                    2,
                    "ready",
                )
            )
        self.assertEqual(scan.call_count, 3)

    def test_post_play_confirmation_waits_for_energy_drop(self) -> None:
        context = FakePlayContext()
        before = BattleHand(6, (card(0, 6, 300),), "recognized")
        stale = BattleHand(6, (card(0, 6, 300),), "recognized")
        updated = BattleHand(0, (), "energy_zero")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                side_effect=[stale, stale, updated],
            ) as scan,
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(
                _confirm_play_from_fresh_frames(
                    context,
                    context.controller,
                    before,
                    6,
                    "timeout",
                )
            )
        self.assertEqual(scan.call_count, 3)


if __name__ == "__main__":
    unittest.main()
