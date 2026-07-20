from pathlib import Path
import unittest

from tools.validate_schema import load_jsonc


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "assets" / "resource" / "pipeline"


def load_nodes() -> dict[str, dict[str, object]]:
    nodes: dict[str, dict[str, object]] = {}
    for path in PIPELINE_ROOT.rglob("*.json"):
        nodes.update(load_jsonc(path))
    return nodes


def next_names(node: dict[str, object]) -> list[str]:
    values = node.get("next", [])
    if isinstance(values, str):
        return [values]
    return [value for value in values if isinstance(value, str)]


class BattlePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes()

    def predecessors(self, target: str) -> set[str]:
        return {
            name for name, node in self.nodes.items() if target in next_names(node)
        }

    def test_common_battle_sequence_nodes_exist(self) -> None:
        required = {
            "公共-比赛开始",
            "公共-战斗继续",
            "公共-首回合",
            "公共-停止判断",
            "公共-执行出牌",
            "公共-SNAP判断",
            "公共-点击SNAP",
            "公共-结束回合",
            "公共-等待对手",
            "公共-新回合",
            "公共-撤退判断",
            "公共-点击撤退",
            "公共-确认撤退",
            "征服-轮间结果",
            "征服-整场结果",
            "征服-记录整场完成",
        }
        self.assertTrue(required.issubset(self.nodes), required - set(self.nodes))

    def test_bootstrap_and_battle_wait_can_resume_live_battle(self) -> None:
        bootstrap = next_names(self.nodes["公共-识别当前页面"])
        self.assertIn("公共-战斗继续", bootstrap)
        self.assertIn("公共-首回合", bootstrap)
        self.assertIn("公共-等待对手", bootstrap)
        self.assertIn("征服-轮间结果", bootstrap)
        self.assertIn("征服-整场结果", bootstrap)

        battle_wait = next_names(self.nodes["公共-等待战斗状态"])
        self.assertIn("公共-战斗继续", battle_wait)

        node = self.nodes["公共-战斗继续"]
        self.assertEqual(node["recognition"]["type"], "OCR")
        self.assertIn("^继续$", node["recognition"]["param"]["expected"])
        self.assertEqual(node["action"]["type"], "Click")
        self.assertNotIn("target", node["action"].get("param", {}))

    def test_play_turn_is_reached_only_after_stop_gate(self) -> None:
        self.assertEqual(self.predecessors("公共-执行出牌"), {"公共-停止跳过"})
        recognition = self.nodes["公共-停止命中"]["recognition"]
        self.assertEqual(
            recognition["param"]["custom_recognition_param"]["command"],
            "should_stop",
        )

    def test_snap_click_is_reached_only_from_snap_gate(self) -> None:
        self.assertEqual(self.predecessors("公共-点击SNAP"), {"公共-SNAP命中"})
        recognition = self.nodes["公共-SNAP命中"]["recognition"]
        self.assertEqual(
            recognition["param"]["custom_recognition_param"]["command"],
            "should_snap",
        )

    def test_retreat_click_is_reached_only_from_retreat_gate(self) -> None:
        self.assertEqual(self.predecessors("公共-点击撤退"), {"公共-撤退命中"})
        recognition = self.nodes["公共-撤退命中"]["recognition"]
        self.assertEqual(
            recognition["param"]["custom_recognition_param"]["command"],
            "should_retreat",
        )

    def test_concede_is_reached_only_from_after_retreat_gate(self) -> None:
        self.assertEqual(
            self.predecessors("公共-点击整场认输"), {"公共-整场认输命中"}
        )
        recognition = self.nodes["公共-整场认输命中"]["recognition"]
        self.assertEqual(
            recognition["param"]["custom_recognition_param"]["command"],
            "after_retreat_concede",
        )

    def test_critical_buttons_click_recognized_text_boxes(self) -> None:
        for name in (
            "公共-点击SNAP",
            "公共-结束回合",
            "公共-点击撤退",
            "公共-确认撤退",
            "公共-点击整场认输",
            "公共-确认整场认输",
            "征服-轮间继续",
            "征服-结果继续",
        ):
            with self.subTest(name=name):
                node = self.nodes[name]
                self.assertEqual(node["recognition"]["type"], "OCR")
                self.assertEqual(node["action"]["type"], "Click")
                self.assertNotIn("target", node["action"].get("param", {}))

    def test_end_turn_accepts_live_ocr_variant_in_button_area(self) -> None:
        params = self.nodes["公共-结束回合"]["recognition"]["param"]
        self.assertIn("结束回[合会]", params["expected"])
        x, y, width, height = params["roi"]
        self.assertGreaterEqual(x, 500)
        self.assertGreaterEqual(y, 1120)
        self.assertEqual(x + width, 720)
        self.assertLessEqual(y + height, 1280)

    def test_match_completion_records_before_next_tier(self) -> None:
        record = self.nodes["征服-记录整场完成"]
        self.assertEqual(
            record["action"]["param"]["custom_action_param"]["event"],
            "match_completed",
        )
        self.assertEqual(next_names(record), ["征服-结束后停止判断"])
        self.assertIn(
            "征服-选择档位候选",
            next_names(self.nodes["征服-结束后继续"]),
        )

    def test_round_result_accepts_live_next_step_label(self) -> None:
        for name in ("征服-轮间结果", "征服-轮间继续"):
            params = self.nodes[name]["recognition"]["param"]
            expected = params["expected"]
            self.assertIn("下一步", expected, name)
            x, _, width, _ = params["roi"]
            self.assertEqual(x + width, 720, name)

    def test_whole_match_result_does_not_match_lobby_victory_count(self) -> None:
        expected = self.nodes["征服-整场结果"]["recognition"]["param"]["expected"]
        self.assertNotIn("胜利", expected)
        self.assertNotIn("失败", expected)
        self.assertIn("战斗胜利", expected)
        self.assertIn("战斗失败", expected)
        self.assertIn(
            "征服-返回征服大厅",
            next_names(self.nodes["征服-整场结果"]),
        )
        self.assertIn(
            "征服-返回征服大厅",
            next_names(self.nodes["公共-等待新状态"]),
        )

    def test_zero_energy_recognition_exists_for_random_strategy(self) -> None:
        node = self.nodes["公共-零能量"]
        self.assertEqual(node["recognition"]["type"], "OCR")
        self.assertIn("^0$", node["recognition"]["param"]["expected"])


if __name__ == "__main__":
    unittest.main()
