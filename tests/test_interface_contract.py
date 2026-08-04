from pathlib import Path
import re
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
        self.assertEqual(controller["display_short_side"], 1080)

    def test_agent_runs_as_a_package(self) -> None:
        self.assertEqual(self.interface["agent"]["child_exec"], "python")
        self.assertEqual(
            self.interface["agent"]["child_args"], ["-m", "agent.main"]
        )

    def test_conquest_mail_and_daily_tasks_are_registered(self) -> None:
        self.assertEqual(self.interface["import"], ["tasks/征服模式.json"])
        self.assertEqual(len(self.task_file["task"]), 5)
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
        training = self.task_file["task"][2]
        self.assertEqual(training["entry"], "训练-OCR选牌测试入口")
        self.assertFalse(training["default_check"])
        mailbox = self.task_file["task"][3]
        self.assertEqual(mailbox["name"], "领取邮箱奖励")
        self.assertEqual(mailbox["entry"], "邮箱-任务入口")
        self.assertFalse(mailbox["default_check"])
        self.assertEqual(mailbox["option"], [])
        daily = self.task_file["task"][4]
        self.assertEqual(daily["name"], "一键日常任务")
        self.assertEqual(daily["entry"], "日常-任务入口")
        self.assertFalse(daily["default_check"])
        self.assertEqual(daily["resource"], ["官服"])
        self.assertEqual(daily["controller"], ["安卓端"])
        self.assertEqual(daily["option"][0], "日常-完成后模式")

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
            "征服-领取任务奖励小时",
            "征服-匹配超时",
            "征服-自动重启",
            "日常-完成后模式",
        }
        self.assertEqual(set(options), expected)

    def test_defaults_match_the_approved_design(self) -> None:
        options = self.task_file["option"]
        deck_option = options["征服-选择卡组"]
        self.assertEqual(deck_option["inputs"][0]["default"], "0")
        self.assertIn("填写 0 表示不选择卡组", deck_option["description"])
        self.assertEqual(
            deck_option["pipeline_override"]["征服-选择指定卡组"]["recognition"][
                "param"
            ]["expected"],
            ["^(?!0$){卡组名称}$"],
        )
        self.assertEqual(
            deck_option["pipeline_override"]["征服-验证指定卡组"]["recognition"][
                "param"
            ]["expected"],
            ["^(?!0$){卡组名称}$"],
        )
        self.assertEqual(
            deck_option["pipeline_override"]["Config_DeckName"]["action"]
            ["param"]["custom_action_param"]["deck_name"],
            "{卡组名称}",
        )
        deck_pattern = deck_option["pipeline_override"]["征服-选择指定卡组"][
            "recognition"
        ]["param"]["expected"][0]
        self.assertIsNone(re.fullmatch(deck_pattern.replace("{卡组名称}", "0"), "0"))
        self.assertIsNotNone(
            re.fullmatch(
                deck_pattern.replace("{卡组名称}", "测试卡组"),
                "测试卡组",
            )
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
        self.assertEqual(options["征服-每日经验上限后结束"]["default_case"], "No")
        self.assertEqual(options["征服-自动撤退"]["default_case"], "off")
        self.assertNotIn("征服-撤退后", options)
        self.assertNotIn("征服-SNAP", options)
        self.assertNotIn("征服-SNAP概率", options)
        reward_hours = options["征服-领取任务奖励小时"]
        self.assertEqual(reward_hours["label"], "领取任务奖励小时")
        self.assertEqual(reward_hours["inputs"][0]["default"], "0")
        self.assertEqual(reward_hours["inputs"][0]["pipeline_type"], "int")
        self.assertEqual(
            reward_hours["inputs"][0]["verify"],
            "^(0|[1-9][0-9]*)$",
        )
        self.assertEqual(
            reward_hours["pipeline_override"]["Config_ClaimTaskRewardsHours"][
                "action"
            ]["param"]["custom_action_param"]["claim_task_rewards_hours"],
            "{领取任务奖励小时}",
        )
        self.assertNotIn("征服-最大对局数", options)
        self.assertNotIn("征服-最大运行分钟", options)
        self.assertEqual(options["征服-匹配超时"]["inputs"][0]["default"], "600")
        self.assertEqual(options["征服-自动重启"]["default_case"], "Yes")
        daily_mode = options["日常-完成后模式"]
        self.assertEqual(daily_mode["default_case"], "conquest")
        self.assertEqual(
            [case["name"] for case in daily_mode["cases"]],
            ["conquest", "ladder"],
        )
        for case in daily_mode["cases"]:
            self.assertEqual(
                case["pipeline_override"]["Config_DailyBattleMode"]["action"]
                ["param"]["custom_action_param"]["game_mode"],
                case["name"],
            )

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
        for case in option["cases"]:
            self.assertEqual(
                set(case["pipeline_override"]),
                {"Config_LaneOrder"},
            )

    def test_reward_schedule_is_shared_by_battle_tasks_only(self) -> None:
        conquest, ladder, training, mailbox, daily = self.task_file["task"]
        option_name = "征服-领取任务奖励小时"
        self.assertIn(option_name, conquest["option"])
        self.assertIn(option_name, ladder["option"])
        self.assertIn(option_name, daily["option"])
        self.assertNotIn(option_name, mailbox["option"])
        self.assertNotIn(option_name, training["option"])
        for task in (conquest, ladder, training, daily):
            self.assertNotIn("征服-最大对局数", task["option"])
            self.assertNotIn("征服-最大运行分钟", task["option"])


if __name__ == "__main__":
    unittest.main()
