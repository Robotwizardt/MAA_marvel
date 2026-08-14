from types import SimpleNamespace
import json
from pathlib import Path
import unittest

import numpy as np

from agent.recognitions.daily_task_reward import (
    DAILY_TASK_PROGRESS_ROIS,
    DAILY_TASK_ROW_BOXES,
    TASK_REWARD_BADGE_ROI,
    DailyTaskClear,
    DailyTaskReward,
    completed_task_progress,
    has_pending_task_reward,
)


def ocr_detail(*texts: str):
    results = [
        SimpleNamespace(text=text, score=0.95, box=(10, 10, 20, 20))
        for text in texts
    ]
    return SimpleNamespace(filtered_results=results, all_results=results)


class FakeRewardContext:
    def __init__(self, details, *, on_page: bool = True) -> None:
        self.details = list(details)
        self.on_page = on_page
        self.ocr_params = []

    def run_recognition(self, entry, image):
        assert entry == "公共-领奖-任务页证据"
        return SimpleNamespace(
            hit=self.on_page,
            box=(700, 120, 500, 120) if self.on_page else None,
        )

    def run_recognition_direct(self, reco_type, reco_param, image):
        self.ocr_params.append(reco_param)
        return self.details.pop(0)


def task_page_image(*, pending: bool) -> np.ndarray:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    if pending:
        x, y, width, height = TASK_REWARD_BADGE_ROI
        image[y : y + height, x : x + width, 2] = 255
    return image


class TaskRewardRecognitionTests(unittest.TestCase):
    def test_home_daily_task_entry_covers_button_and_has_safe_fallback(self) -> None:
        pipeline_path = (
            Path(__file__).parents[1]
            / "assets"
            / "resource"
            / "pipeline"
            / "common"
            / "task_rewards.json"
        )
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))

        daily_task_text = pipeline["公共-领奖-查看所有每日任务"]
        self.assertEqual(daily_task_text["recognition"]["type"], "OCR")
        roi = daily_task_text["recognition"]["param"]["roi"]
        x, y, width, height = roi
        self.assertLessEqual(x, 1500)
        self.assertLessEqual(y, 320)
        self.assertGreaterEqual(x + width, 1920)
        self.assertGreaterEqual(y + height, 1060)
        self.assertEqual(
            daily_task_text["recognition"]["param"]["expected"],
            ["^查看所有每日任务$", "^查看所有任务$"],
        )

        home_entry = pipeline["公共-领奖-首页入口"]
        self.assertEqual(
            home_entry["recognition"]["param"]["all_of"],
            ["公共-主界面", "公共-领奖-查看所有每日任务"],
        )
        self.assertEqual(home_entry["recognition"]["param"]["box_index"], 1)
        self.assertEqual(home_entry["action"]["type"], "Click")
        self.assertNotIn("target", home_entry["action"])
        self.assertEqual(
            pipeline["公共-领奖-首页稳定态"]["next"],
            ["公共-领奖-首页入口"],
        )
        no_reward = pipeline["公共-领奖-任务页无可领取"]
        self.assertEqual(no_reward["recognition"]["type"], "Custom")
        self.assertEqual(
            no_reward["recognition"]["param"]["custom_recognition"],
            "MarvelDailyTaskClear",
        )

    def test_progress_parser_accepts_supported_ocr_shapes(self) -> None:
        for texts, expected in (
            (["5/5"], (5, 5)),
            (["5", "5"], (5, 5)),
            (["55"], (5, 5)),
            (["10／10"], (10, 10)),
            (["1010"], (10, 10)),
        ):
            with self.subTest(texts=texts):
                self.assertEqual(completed_task_progress(texts), expected)

    def test_progress_parser_rejects_incomplete_or_ambiguous_values(self) -> None:
        for texts in (
            [],
            ["0/0"],
            ["1/2"],
            ["1", "2"],
            ["12"],
            ["5", "5", "50"],
            ["任务"],
        ):
            with self.subTest(texts=texts):
                self.assertIsNone(completed_task_progress(texts))

    def test_recognition_returns_first_completed_daily_row(self) -> None:
        context = FakeRewardContext(
            [
                ocr_detail("1"),
                ocr_detail("2"),
                ocr_detail("5"),
                ocr_detail("5"),
            ]
        )
        result = DailyTaskReward().analyze(
            context,
            SimpleNamespace(
                custom_recognition_param="{}",
                image=task_page_image(pending=True),
            ),
        )
        self.assertEqual(result.box, DAILY_TASK_ROW_BOXES[1])
        self.assertEqual(result.detail["row"], 1)
        self.assertEqual(len(context.ocr_params), 4)
        self.assertEqual(
            tuple(context.ocr_params[0].roi),
            DAILY_TASK_PROGRESS_ROIS[0][0],
        )

    def test_recognition_rejects_non_task_page_without_ocr(self) -> None:
        context = FakeRewardContext([], on_page=False)
        result = DailyTaskReward().analyze(
            context,
            SimpleNamespace(
                custom_recognition_param="{}",
                image=task_page_image(pending=True),
            ),
        )
        self.assertIsNone(result.box)
        self.assertEqual(result.detail["reason"], "not_daily_task_page")
        self.assertEqual(context.ocr_params, [])

    def test_pending_badge_falls_back_to_first_completed_row_when_ocr_is_ambiguous(
        self,
    ) -> None:
        context = FakeRewardContext(
            [ocr_detail("1"), ocr_detail("N")] * 3
        )
        result = DailyTaskReward().analyze(
            context,
            SimpleNamespace(
                custom_recognition_param="{}",
                image=task_page_image(pending=True),
            ),
        )
        self.assertEqual(result.box, DAILY_TASK_ROW_BOXES[0])
        self.assertEqual(
            result.detail["reason"],
            "pending_badge_first_row_fallback",
        )
        self.assertEqual(len(context.ocr_params), 6)

    def test_no_badge_skips_ocr_and_clear_gate_succeeds(self) -> None:
        image = task_page_image(pending=False)
        context = FakeRewardContext([])
        reward = DailyTaskReward().analyze(
            context,
            SimpleNamespace(custom_recognition_param="{}", image=image),
        )
        self.assertIsNone(reward.box)
        self.assertEqual(reward.detail["reason"], "no_pending_task_badge")
        self.assertEqual(context.ocr_params, [])

        clear = DailyTaskClear().analyze(
            context,
            SimpleNamespace(custom_recognition_param="{}", image=image),
        )
        self.assertIsNotNone(clear.box)
        self.assertEqual(clear.detail["reason"], "task_rewards_clear")

    def test_pending_badge_blocks_clear_gate(self) -> None:
        image = task_page_image(pending=True)
        self.assertTrue(has_pending_task_reward(image))
        context = FakeRewardContext([])
        clear = DailyTaskClear().analyze(
            context,
            SimpleNamespace(custom_recognition_param="{}", image=image),
        )
        self.assertIsNone(clear.box)
        self.assertEqual(clear.detail["reason"], "pending_task_badge")


if __name__ == "__main__":
    unittest.main()
