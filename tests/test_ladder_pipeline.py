from pathlib import Path
from types import SimpleNamespace
import unittest

from agent.actions.configure_session import ConfigureSession
from agent.runtime.store import STORE
from agent.session.config import GameMode, PlayStrategy, SnapMode
from tools.validate_schema import load_jsonc


ROOT = Path(__file__).resolve().parents[1]


class FakeConfigContext:
    def get_node_data(self, name: str):
        assert name == "征服-初始化会话"
        return {
            "action": {
                "param": {
                    "custom_action_param": {
                        "play_strategy": "agatha",
                        "snap_mode": "always",
                        "max_matches": 7,
                    }
                }
            }
        }


class LadderPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_jsonc(
            ROOT / "assets/resource/pipeline/ladder/ladder.json"
        )
        cls.tasks = load_jsonc(ROOT / "assets/tasks/征服模式.json")

    def test_ladder_task_exposes_only_common_battle_options(self) -> None:
        task = next(item for item in self.tasks["task"] if item["entry"] == "天梯-任务入口")
        self.assertNotIn("征服-最高档位", task["option"])
        self.assertNotIn("征服-保留门票数", task["option"])
        self.assertIn("征服-出牌策略", task["option"])
        self.assertIn("征服-放牌场地顺序", task["option"])
        self.assertIn("征服-每日经验上限后结束", task["option"])

    def test_ladder_initialization_reuses_overridden_common_values(self) -> None:
        ConfigureSession().run(
            FakeConfigContext(),
            SimpleNamespace(custom_action_param='{"game_mode":"ladder"}'),
        )
        config = STORE.require_state().config
        self.assertEqual(config.game_mode, GameMode.LADDER)
        self.assertEqual(config.play_strategy, PlayStrategy.AGATHA)
        self.assertEqual(config.snap_mode, SnapMode.ALWAYS)
        self.assertEqual(config.max_matches, 7)

    def test_training_mode_green_state_is_disabled_before_battle(self) -> None:
        node = self.nodes["天梯-训练模式已开启"]
        self.assertEqual(node["recognition"]["type"], "ColorMatch")
        self.assertEqual(node["recognition"]["param"]["roi"], [420, 900, 90, 60])
        self.assertEqual(node["action"]["type"], "Click")
        self.assertEqual(node["next"], ["天梯-主页可开战"])

    def test_ladder_result_is_mode_gated_and_returns_to_home(self) -> None:
        result = self.nodes["天梯-整场结果"]["recognition"]
        self.assertEqual(result["type"], "And")
        self.assertEqual(
            result["param"]["all_of"],
            ["天梯-模式证据", "天梯-结果文字"],
        )
        self.assertEqual(self.nodes["天梯-记录完成"]["next"], ["天梯-停止判断"])
        self.assertEqual(self.nodes["天梯-继续下一局"]["next"], ["天梯-主页模式命中"])

        conquest_results = load_jsonc(
            ROOT / "assets/resource/pipeline/conquest/results.json"
        )
        result_after = conquest_results["征服-结果后状态"]["next"]
        self.assertIn("天梯-整场结果", result_after)
        self.assertIn("天梯-返回主页", result_after)


if __name__ == "__main__":
    unittest.main()
