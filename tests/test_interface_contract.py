from pathlib import Path
import unittest

from tools.validate_schema import load_jsonc


ROOT = Path(__file__).resolve().parents[1]


class InterfaceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.interface = load_jsonc(ROOT / "assets" / "interface.json")
        cls.task_file = load_jsonc(ROOT / "assets" / "tasks" / "征服模式.json")

    def test_project_metadata_targets_android_cn_client(self) -> None:
        self.assertEqual(self.interface["name"], "MAA_marvel")
        self.assertEqual(
            self.interface["github"],
            "https://github.com/Robotwizardt/MAA_marvel",
        )
        self.assertEqual(self.interface["license"], "MIT")
        self.assertNotIn("contact", self.interface)

        self.assertEqual(len(self.interface["controller"]), 1)
        controller = self.interface["controller"][0]
        self.assertEqual(controller["type"], "Adb")
        self.assertEqual(controller["display_short_side"], 720)

    def test_agent_runs_as_a_package(self) -> None:
        self.assertEqual(self.interface["agent"]["child_exec"], "python")
        self.assertEqual(
            self.interface["agent"]["child_args"], ["-m", "agent.main"]
        )

    def test_only_conquest_task_file_is_imported(self) -> None:
        self.assertEqual(self.interface["import"], ["tasks/征服模式.json"])
        self.assertEqual(len(self.task_file["task"]), 3)
        task = self.task_file["task"][0]
        self.assertEqual(task["name"], "征服模式自动对战")
        self.assertEqual(task["entry"], "征服-任务入口")
        self.assertTrue(task["default_check"])
        self.assertEqual(task["resource"], ["官服"])
        self.assertEqual(task["controller"], ["安卓端"])
        ladder = self.task_file["task"][1]
        self.assertEqual(ladder["name"], "天梯模式自动对战")
        self.assertEqual(ladder["entry"], "天梯-任务入口")
        self.assertFalse(ladder["default_check"])
        training_tasks = self.task_file["task"][2:]
        self.assertEqual(
            [item["entry"] for item in training_tasks],
            ["训练-OCR选牌测试入口"],
        )
        self.assertTrue(all(not item["default_check"] for item in training_tasks))

    def test_all_approved_options_are_exposed(self) -> None:
        options = self.task_file["option"]
        expected = {
            "征服-选择卡组",
            "征服-出牌策略",
            "征服-放牌场地顺序",
            "征服-最高档位",
            "征服-保留门票数",
            "征服-每日经验上限后结束",
            "征服-自动撤退",
            "征服-最大对局数",
            "征服-最大运行分钟",
            "征服-匹配超时",
            "征服-自动重启",
        }
        self.assertEqual(set(options), expected)

    def test_defaults_match_the_approved_design(self) -> None:
        options = self.task_file["option"]
        deck_option = options["征服-选择卡组"]
        self.assertEqual(deck_option["inputs"][0]["default"], "")
        self.assertEqual(
            deck_option["pipeline_override"]["征服-选择指定卡组"]["recognition"][
                "param"
            ]["expected"],
            ["^{卡组名称}$"],
        )
        self.assertEqual(options["征服-出牌策略"]["default_case"], "ocr")
        self.assertEqual(
            options["征服-放牌场地顺序"]["default_case"], "left_to_right"
        )
        self.assertEqual(
            options["征服-最高档位"]["default_case"], "proving_grounds"
        )
        reserves = options["征服-保留门票数"]["inputs"]
        self.assertEqual(len(reserves), 3)
        self.assertTrue(all(item["default"].isdigit() for item in reserves))
        self.assertEqual(options["征服-每日经验上限后结束"]["default_case"], "Yes")
        self.assertEqual(options["征服-自动撤退"]["default_case"], "off")
        self.assertNotIn("征服-撤退后", options)
        self.assertNotIn("征服-SNAP", options)
        self.assertNotIn("征服-SNAP概率", options)
        self.assertEqual(options["征服-最大对局数"]["inputs"][0]["default"], "1")
        self.assertEqual(
            options["征服-最大运行分钟"]["inputs"][0]["default"], "30"
        )
        self.assertEqual(options["征服-匹配超时"]["inputs"][0]["default"], "600")
        self.assertEqual(options["征服-自动重启"]["default_case"], "Yes")

    def test_strategy_cases_are_agatha_and_ocr(self) -> None:
        cases = self.task_file["option"]["征服-出牌策略"]["cases"]
        self.assertEqual([case["name"] for case in cases], ["agatha", "ocr"])
        self.assertIn("智能", cases[1]["label"])

    def test_retreat_only_exposes_late_turns_and_always_continues(self) -> None:
        cases = self.task_file["option"]["征服-自动撤退"]["cases"]
        self.assertEqual(
            [case["name"] for case in cases],
            ["off", "turn_4", "turn_5", "turn_6"],
        )
        self.assertNotIn("征服-撤退后", self.task_file["option"])

    def test_lane_order_exposes_random_and_both_directions(self) -> None:
        option = self.task_file["option"]["征服-放牌场地顺序"]
        self.assertEqual(
            [case["name"] for case in option["cases"]],
            ["random", "left_to_right", "right_to_left"],
        )
        training = self.task_file["task"][2]
        self.assertIn("征服-放牌场地顺序", training["option"])


if __name__ == "__main__":
    unittest.main()
