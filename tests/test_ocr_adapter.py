from types import SimpleNamespace
import unittest

from agent.actions.play_turn import PlayTurn
from agent.recognitions.card_selection import (
    HAND_BOXES,
    CardSelection,
    parse_digit_results,
)
from agent.runtime.store import STORE


def ocr_result(text: str, score: float = 0.95, box=(10, 10, 20, 20)):
    return SimpleNamespace(text=text, score=score, box=box)


def recognition_detail(*results):
    return SimpleNamespace(filtered_results=list(results))


class FakeDirectContext:
    def __init__(self, details: list[object | None]) -> None:
        self.details = list(details)

    def run_recognition_direct(self, reco_type, reco_param, image):
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

    def post_screencap(self):
        return FakeScreenshotJob()

    def post_swipe(self, x1, y1, x2, y2, duration):
        self.swipes.append((x1, y1, x2, y2, duration))
        return FakeJob()


class FakePlayContext:
    def __init__(self, recognition_results: list[object | None]) -> None:
        self.controller = FakeController()
        self.tasker = SimpleNamespace(controller=self.controller)
        self.recognition_results = list(recognition_results)

    def run_recognition(self, entry, image):
        self.last_entry = entry
        return self.recognition_results.pop(0)


class OcrAdapterTests(unittest.TestCase):
    def test_parse_digit_requires_one_unambiguous_digit(self) -> None:
        parsed = parse_digit_results([ocr_result("3")])
        self.assertEqual(parsed.value, 3)
        self.assertEqual(parsed.reason, "recognized")

        conflict = parse_digit_results([ocr_result("2"), ocr_result("3")])
        self.assertIsNone(conflict.value)
        self.assertEqual(conflict.reason, "conflicting_digits")

        invalid = parse_digit_results([ocr_result("12"), ocr_result("A")])
        self.assertIsNone(invalid.value)
        self.assertEqual(invalid.reason, "no_single_digit")

    def analyze(self, details: list[object | None]):
        argv = SimpleNamespace(image=object())
        return CardSelection().analyze(FakeDirectContext(details), argv)

    def test_card_selection_chooses_highest_affordable_slot(self) -> None:
        result = self.analyze(
            [
                recognition_detail(ocr_result("3")),
                recognition_detail(ocr_result("1", box=(20, 950, 20, 20))),
                recognition_detail(ocr_result("4", box=(200, 950, 20, 20))),
                recognition_detail(ocr_result("3", box=(380, 950, 20, 20))),
                None,
            ]
        )
        self.assertEqual(result.box, HAND_BOXES[2])
        self.assertEqual(result.detail["energy"], 3)
        self.assertEqual(result.detail["selected_slot"], 2)
        self.assertEqual(result.detail["reason"], "selected")

    def test_card_selection_reports_missing_energy(self) -> None:
        result = self.analyze([None])
        self.assertIsNone(result.box)
        self.assertIsNone(result.detail["energy"])
        self.assertEqual(result.detail["reason"], "energy_missing")

    def test_card_selection_rejects_low_confidence_cards(self) -> None:
        result = self.analyze(
            [
                recognition_detail(ocr_result("5")),
                recognition_detail(ocr_result("2", score=0.79)),
                None,
                None,
                None,
            ]
        )
        self.assertIsNone(result.box)
        self.assertEqual(result.detail["reason"], "low_confidence")

    def test_ocr_play_failure_never_falls_back_to_random_swipes(self) -> None:
        STORE.configure({"play_strategy": "ocr", "max_matches": 0, "max_minutes": 0}, now=0.0)
        context = FakePlayContext([None])
        self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(context.last_entry, "公共-OCR选牌")
        self.assertEqual(context.controller.swipes, [])

    def test_ocr_play_drags_selected_card_then_stops_on_next_miss(self) -> None:
        STORE.configure({"play_strategy": "ocr", "max_matches": 0, "max_minutes": 0}, now=0.0)
        selected = SimpleNamespace(box=SimpleNamespace(x=360, y=1010, w=80, h=100))
        context = FakePlayContext([selected, None])
        self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 1)
        x1, y1, x2, y2, duration = context.controller.swipes[0]
        self.assertEqual((x1, y1), (400, 1060))
        self.assertTrue(85 <= x2 <= 635)
        self.assertTrue(600 <= y2 <= 720)
        self.assertEqual(duration, 350)

    def test_random_play_ignores_failed_zero_energy_result(self) -> None:
        STORE.configure({"play_strategy": "random", "max_matches": 0, "max_minutes": 0}, now=0.0)
        failed = SimpleNamespace(box=None)
        context = FakePlayContext([failed] * 8)
        self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 8)

    def test_random_play_stops_only_on_zero_energy_hit(self) -> None:
        STORE.configure({"play_strategy": "random", "max_matches": 0, "max_minutes": 0}, now=0.0)
        failed = SimpleNamespace(box=None)
        hit = SimpleNamespace(box=SimpleNamespace(x=300, y=1130, w=80, h=80))
        context = FakePlayContext([failed, failed, hit])
        self.assertTrue(PlayTurn().run(context, SimpleNamespace()))
        self.assertEqual(len(context.controller.swipes), 3)


if __name__ == "__main__":
    unittest.main()
