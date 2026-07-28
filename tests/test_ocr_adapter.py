from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from agent.actions.play_turn import (
    LANE_TARGETS,
    PlayTurn,
    _close_detail_overlay,
    _click_end_turn,
    _is_detail_overlay,
    _scan_with_retry,
    _wait_for_agatha,
    lane_targets_for_order,
)
from agent.recognitions.card_selection import (
    BattleHand,
    CardSelection,
    DetectedCard,
    ENERGY_DIGIT_ROI,
    HAND_COST_ROI,
    HAND_COST_WINDOWS,
    _build_cost_probe_rois,
    _detect_cards,
    _results,
    parse_digit_results,
    scan_battle_hand,
)
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
    return DetectedCard(slot, cost, 0.95, (x - 35, 1010, 70, 100))


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

    def run_recognition(self, entry, image):
        self.last_recognition_entry = entry
        return self.detail_match


class OcrAdapterTests(unittest.TestCase):
    def test_ocr_uses_raw_digit_when_expected_filter_is_empty(self) -> None:
        detail = SimpleNamespace(
            filtered_results=[],
            all_results=[ocr_result("2", score=0.97)],
        )
        parsed = parse_digit_results(_results(detail))
        self.assertEqual(parsed.value, 2)
        self.assertEqual(parsed.confidence, 0.97)

    def test_detail_overlay_requires_red_button_and_white_x(self) -> None:
        overlay = np.zeros((1280, 720, 3), dtype=np.uint8)
        overlay[1145:1245, 305:415] = (30, 35, 170)
        overlay[1180:1220, 345:371] = (230, 230, 230)
        self.assertTrue(_is_detail_overlay(overlay))

        normal_battle = np.zeros((1280, 720, 3), dtype=np.uint8)
        normal_battle[1145:1245, 305:415] = (220, 100, 30)
        normal_battle[1180:1220, 345:371] = (230, 230, 230)
        self.assertFalse(_is_detail_overlay(normal_battle))

    def test_detail_overlay_click_requires_template_confirmation(self) -> None:
        overlay = np.zeros((1280, 720, 3), dtype=np.uint8)
        overlay[1145:1245, 305:415] = (30, 35, 170)
        overlay[1180:1220, 345:371] = (230, 230, 230)

        rejected = FakePlayContext(detail_match=None)
        self.assertFalse(_close_detail_overlay(rejected, rejected.controller, overlay))
        self.assertEqual(rejected.controller.clicks, [])

        confirmed = FakePlayContext(
            detail_match=SimpleNamespace(box=(300, 1145, 118, 100))
        )
        with patch("agent.actions.play_turn.time.sleep"):
            self.assertTrue(
                _close_detail_overlay(confirmed, confirmed.controller, overlay)
            )
        self.assertEqual(confirmed.last_recognition_entry, "公共-详情关闭按钮")
        self.assertEqual(confirmed.controller.clicks, [(359, 1195)])

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

    def test_ocr_scan_reads_energy_then_uses_one_hand_fast_scan(self) -> None:
        context = FakeDirectContext(
            [
                recognition_detail(ocr_result("1")),
                recognition_detail(ocr_result("1", box=(100, 70, 20, 20))),
            ]
        )
        hand = scan_battle_hand(context, object())
        self.assertEqual(hand.energy, 1)
        self.assertEqual([card.cost for card in hand.cards], [1])
        self.assertEqual(len(context.reco_params), 2)
        self.assertEqual(context.reco_params[0].roi, ENERGY_DIGIT_ROI)
        self.assertTrue(context.reco_params[0].only_rec)
        self.assertEqual(context.reco_params[1].roi, HAND_COST_ROI)
        self.assertEqual(context.reco_params[1].threshold, 0.20)

    def test_overlapping_windows_recover_digit_missed_by_full_hand_scan(self) -> None:
        context = FakeDirectContext(
            [
                recognition_detail(ocr_result("1")),
                recognition_detail(),
                recognition_detail(ocr_result("1", box=(100, 70, 20, 20))),
                *[recognition_detail() for _ in range(len(HAND_COST_WINDOWS) - 1)],
            ]
        )
        hand = scan_battle_hand(context, object())
        self.assertEqual(hand.energy, 1)
        self.assertEqual([card.cost for card in hand.cards], [1])
        self.assertEqual(hand.reason, "window_path")

    def test_card_detection_accepts_zero_to_twenty_only(self) -> None:
        detected = _detect_cards(
            [
                (ocr_result("-2", box=(90, 1000, 20, 20)), HAND_COST_ROI),
                (ocr_result("0", box=(190, 1000, 20, 20)), HAND_COST_ROI),
                (ocr_result("10", box=(290, 1000, 20, 20)), HAND_COST_ROI),
                (ocr_result("21", box=(390, 1000, 20, 20)), HAND_COST_ROI),
            ],
            image=object(),
        )
        self.assertEqual([item.cost for item in detected], [0, 10])

    def test_narrow_seven_is_corrected_to_one_cost(self) -> None:
        detected = _detect_cards(
            [
                (ocr_result("7", score=0.67, box=(245, 971, 6, 10)), HAND_COST_ROI),
                (ocr_result("7", score=0.99, box=(345, 971, 16, 20)), HAND_COST_ROI),
            ],
            image=object(),
        )
        self.assertEqual([item.cost for item in detected], [1, 7])

    def test_power_digit_builds_narrow_probe_over_cost_badge_to_its_left(self) -> None:
        rois = _build_cost_probe_rois(
            [(ocr_result("9", box=(301, 962, 19, 20)), HAND_COST_ROI)]
        )
        self.assertEqual(rois, ((190, 937, 98, 70),))

    def test_card_selection_chooses_highest_affordable_card(self) -> None:
        argv = SimpleNamespace(image=object())
        result = CardSelection().analyze(
            FakeDirectContext(
                [
                    recognition_detail(ocr_result("3")),
                    recognition_detail(
                        ocr_result("1", box=(20, 70, 20, 20)),
                        ocr_result("4", box=(200, 70, 20, 20)),
                        ocr_result("3", box=(380, 70, 20, 20)),
                    ),
                ]
            ),
            argv,
        )
        self.assertEqual(result.detail["energy"], 3)
        self.assertEqual(result.detail["selected_slot"], 2)
        self.assertEqual(result.box, (397, 1040, 70, 100))

    def test_ocr_card_scan_normalizes_relative_boxes(self) -> None:
        context = FakeDirectContext(
            [
                recognition_detail(ocr_result("4")),
                recognition_detail(
                    ocr_result("2", box=(110, 70, 20, 20)),
                    ocr_result("3", box=(300, 70, 20, 20)),
                ),
            ]
        )
        hand = scan_battle_hand(context, object())
        self.assertEqual([card.cost for card in hand.cards], [2, 3])

    def test_card_detection_keeps_blue_cost_and_rejects_orange_power(self) -> None:
        """OCR 同时读到费用和战力时，只保留蓝色费用徽标。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        image[988:1034, 78:124] = (200, 70, 20)  # BGR 亮起的费用徽标
        image[988:1034, 178:224] = (60, 65, 70)  # BGR 暗下的费用徽标

        detected = _detect_cards(
            [
                (ocr_result("1", box=(90, 1000, 20, 20)), HAND_COST_ROI),
                (ocr_result("5", box=(190, 1000, 20, 20)), HAND_COST_ROI),
            ],
            image=image,
        )

        self.assertEqual([(item.cost, item.box[0]) for item in detected], [(1, 107)])

    def test_ocr_play_repeats_highest_affordable_until_none_remains(self) -> None:
        STORE.configure(
            {"play_strategy": "ocr", "max_matches": 0, "max_minutes": 0}, now=0.0
        )
        context = FakePlayContext()
        hand_3 = BattleHand(3, (card(0, 1, 180), card(1, 2, 420)), "recognized")
        hand_1 = BattleHand(1, (card(0, 1, 300),), "recognized")
        hand_0 = BattleHand(0, (), "no_cards")
        # 第二张牌打出后的确认扫描和下一轮结束判断都会完整复查四帧。
        scans = [hand_3, hand_1, hand_1, *([hand_0] * 8)]
        with (
            patch("agent.actions.play_turn.scan_battle_hand", side_effect=scans),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 2)
        self.assertEqual(context.controller.swipes[0][:2], (420, 1060))
        self.assertEqual(context.controller.swipes[1][:2], (300, 1060))

    def test_scan_retries_multiple_frames_until_bright_card_is_found(self) -> None:
        context = FakePlayContext()
        no_bright_card = BattleHand(1, (), "no_cards")
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

    def test_agatha_waits_only_for_end_turn_without_scanning_cards(self) -> None:
        context = FakePlayContext(
            detail_match=SimpleNamespace(box=(550, 1149, 144, 55))
        )
        with patch("agent.actions.play_turn.scan_battle_hand") as scan:
            self.assertTrue(_wait_for_agatha(context, context.controller))
        scan.assert_not_called()
        self.assertEqual(context.last_recognition_entry, "公共-结束回合")

    def test_already_snapped_turn_clicks_end_turn_directly(self) -> None:
        context = FakePlayContext(
            detail_match=SimpleNamespace(box=(550, 1149, 144, 55))
        )
        with patch("agent.actions.play_turn.time.sleep"):
            self.assertTrue(
                _click_end_turn(context, context.controller)
            )
        self.assertEqual(context.controller.clicks, [(622, 1176)])

    def test_failed_expensive_card_falls_back_to_cheaper_card(self) -> None:
        STORE.configure(
            {
                "play_strategy": "ocr",
                "snap_mode": "off",
                "max_matches": 0,
                "max_minutes": 0,
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
                side_effect=[initial, initial, initial, initial, after, no_card],
            ),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 4)
        self.assertEqual(context.controller.swipes[-1][:2], (430, 1060))
        self.assertEqual(context.controller.clicks, [(622, 1176)])

    def test_definite_unaffordable_hand_skips_second_scan(self) -> None:
        context = FakePlayContext()
        unaffordable = BattleHand(1, (card(0, 2, 300),), "cost_probe_path")
        with patch(
            "agent.actions.play_turn.scan_battle_hand",
            return_value=unaffordable,
        ) as scan:
            result = _scan_with_retry(context, context.controller)
        self.assertEqual(result, unaffordable)
        scan.assert_called_once()

    def test_full_field_tries_each_lane_once_then_stops(self) -> None:
        STORE.configure(
            {"play_strategy": "ocr", "max_matches": 0, "max_minutes": 0}, now=0.0
        )
        context = FakePlayContext()
        unchanged = BattleHand(2, (card(0, 2, 300),), "recognized")
        with (
            patch(
                "agent.actions.play_turn.scan_battle_hand",
                side_effect=[unchanged, unchanged, unchanged, unchanged],
            ),
            patch("agent.actions.play_turn.time.sleep"),
        ):
            self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 3)
        self.assertEqual(
            {swipe[2:4] for swipe in context.controller.swipes},
            {(point.x, point.y) for point in LANE_TARGETS},
        )


if __name__ == "__main__":
    unittest.main()
