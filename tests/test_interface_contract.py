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
        self.assertEqual(len(self.task_file["task"]), 1)
        task = self.task_file["task"][0]
        self.assertEqual(task["name"], "征服模式自动对战")
        self.assertEqual(task["entry"], "征服-任务入口")
        self.assertEqual(task["resource"], ["官服"])
        self.assertEqual(task["controller"], ["安卓端"])

    def test_all_approved_options_are_exposed(self) -> None:
        options = self.task_file["option"]
        expected = {
            "征服-出牌策略",
            "征服-最高档位",
            "征服-无票行为",
            "征服-自动撤退",
            "征服-撤退后",
            "征服-SNAP",
            "征服-SNAP概率",
            "征服-最大对局数",
            "征服-最大运行分钟",
            "征服-匹配超时",
            "征服-自动重启",
        }
        self.assertEqual(set(options), expected)

    def test_defaults_match_the_approved_design(self) -> None:
        options = self.task_file["option"]
        self.assertEqual(options["征服-出牌策略"]["default_case"], "random")
        self.assertEqual(
            options["征服-最高档位"]["default_case"], "proving_grounds"
        )
        self.assertEqual(options["征服-无票行为"]["default_case"], "fallback")
        self.assertEqual(options["征服-自动撤退"]["default_case"], "off")
        self.assertEqual(options["征服-撤退后"]["default_case"], "continue")
        self.assertEqual(options["征服-SNAP"]["default_case"], "off")
        self.assertEqual(options["征服-SNAP概率"]["inputs"][0]["default"], "46")
        self.assertEqual(options["征服-最大对局数"]["inputs"][0]["default"], "0")
        self.assertEqual(
            options["征服-最大运行分钟"]["inputs"][0]["default"], "0"
        )
        self.assertEqual(options["征服-匹配超时"]["inputs"][0]["default"], "600")
        self.assertEqual(options["征服-自动重启"]["default_case"], "Yes")

    def test_strategy_cases_are_random_agatha_and_experimental_ocr(self) -> None:
        cases = self.task_file["option"]["征服-出牌策略"]["cases"]
        self.assertEqual([case["name"] for case in cases], ["random", "agatha", "ocr"])
        self.assertIn("实验", cases[2]["label"])


if __name__ == "__main__":
    unittest.main()
