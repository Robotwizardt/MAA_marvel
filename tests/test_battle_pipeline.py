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
    def test_detail_close_uses_template_not_energy_ball_color_alone(self) -> None:
        recognition = self.nodes["公共-详情关闭按钮"]["recognition"]
        self.assertEqual(recognition["type"], "TemplateMatch")
        self.assertEqual(
            recognition["param"]["template"],
            ["common/battle/detail_close.png"],
        )
        self.assertEqual(recognition["param"]["roi"], [270, 1120, 180, 150])

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
        self.assertIn("征服-结果继续", battle_wait)

        state_wait = next_names(self.nodes["公共-等待新状态"])
        self.assertIn("征服-结果继续", state_wait)

        node = self.nodes["公共-战斗继续"]
        self.assertEqual(node["recognition"]["type"], "OCR")
        self.assertIn("^继续$", node["recognition"]["param"]["expected"])
        self.assertEqual(node["action"]["type"], "Click")
        self.assertNotIn("target", node["action"].get("param", {}))

    def test_bootstrap_waits_on_loading_screen_and_recognizes_all_tier_lobbies(self) -> None:
        bootstrap = next_names(self.nodes["公共-识别当前页面"])
        self.assertEqual(bootstrap[0], "公共-启动加载中")
        self.assertEqual(bootstrap[1], "公共-启动活动弹窗")
        self.assertEqual(bootstrap[2], "公共-累计签到奖励")
        self.assertIn("征服-白银标题", bootstrap)
        self.assertIn("征服-黄金标题", bootstrap)
        loading = self.nodes["公共-启动加载中"]
        self.assertEqual(
            loading["recognition"]["param"]["expected"],
            ["prod-[0-9.]+"],
        )
        self.assertEqual(next_names(loading), ["公共-识别当前页面"])
        popup = self.nodes["公共-启动活动弹窗"]
        self.assertEqual(
            popup["recognition"]["param"]["expected"],
            ["立即前往"],
        )
        self.assertEqual(
            popup["action"],
            {"type": "Click", "param": {"target": [570, 140, 100, 120]}},
        )
        reward = self.nodes["公共-累计签到奖励"]
        self.assertEqual(reward["recognition"]["param"]["expected"], ["累签大奖"])
        self.assertEqual(
            reward["action"],
            {"type": "Click", "param": {"target": [300, 1140, 120, 130]}},
        )

    def test_exit_confirmation_cancel_uses_live_exact_ocr(self) -> None:
        node = self.nodes["公共-退出游戏取消"]
        self.assertEqual(node["recognition"]["type"], "OCR")
        self.assertEqual(node["recognition"]["param"]["roi"], [100, 630, 290, 170])
        self.assertEqual(node["recognition"]["param"]["expected"], ["^否$"])

    def test_reconnect_uses_live_bottom_button(self) -> None:
        node = self.nodes["公共-重新连接"]
        self.assertEqual(node["recognition"]["param"]["roi"], [150, 900, 420, 250])
        self.assertEqual(
            node["recognition"]["param"]["expected"],
            ["^重新连接$", "^重连$"],
        )

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
        click_recognition = self.nodes["公共-点击SNAP"]["recognition"]
        self.assertEqual(click_recognition["param"]["roi"], [280, 40, 160, 170])
        self.assertEqual(click_recognition["param"]["expected"], ["^[1248]$"])

    def test_post_play_state_router_handles_end_turn_and_transitions(self) -> None:
        next_nodes = self.nodes["公共-出牌后状态"]["next"]
        self.assertEqual(next_nodes[0], "征服-每日经验上限")
        self.assertEqual(next_nodes[1], "公共-结束回合")
        self.assertEqual(self.nodes["公共-出牌后状态"]["rate_limit"], 200)
        self.assertIn("公共-战斗继续", next_nodes)
        self.assertIn("公共-新回合", next_nodes)
        self.assertNotIn("公共-结束回合", self.nodes["公共-等待新状态"]["next"])
        self.assertEqual(self.nodes["公共-SNAP跳过"]["next"], ["公共-出牌后状态"])
        self.assertEqual(self.nodes["公共-点击SNAP"]["next"], ["公共-出牌后状态"])

    def test_daily_pass_limit_can_stop_or_continue_from_result_flow(self) -> None:
        limit = self.nodes["征服-每日经验上限"]
        self.assertEqual(limit["recognition"]["type"], "OCR")
        self.assertEqual(limit["recognition"]["param"]["roi"], [0, 0, 720, 1280])
        gate = self.nodes["征服-每日经验上限停止"]["recognition"]
        self.assertEqual(
            gate["param"]["custom_recognition_param"]["command"],
            "stop_on_daily_pass_limit",
        )
        self.assertEqual(next_names(self.nodes["征服-每日经验上限停止"]), ["公共-安全停止"])
        self.assertEqual(self.nodes["征服-每日经验上限继续"]["action"]["type"], "Click")
        self.assertIn("征服-每日经验上限", next_names(self.nodes["征服-结果后状态"]))

    def test_waiting_opponent_accepts_live_waiting_label(self) -> None:
        expected = self.nodes["公共-等待对手"]["recognition"]["param"]["expected"]
        self.assertIn("等待中", expected)

    def test_zero_energy_accepts_live_ocr_letter_variant(self) -> None:
        recognition = self.nodes["公共-零能量"]["recognition"]
        self.assertEqual(recognition["param"]["roi"], [290, 1120, 140, 160])
        self.assertEqual(recognition["param"]["expected"], ["^[0O]$"])

    def test_retreat_click_is_reached_only_from_retreat_gate(self) -> None:
        self.assertEqual(self.predecessors("公共-点击撤退"), {"公共-撤退命中"})
        recognition = self.nodes["公共-撤退命中"]["recognition"]
        self.assertEqual(
            recognition["param"]["custom_recognition_param"]["command"],
            "should_retreat",
        )

    def test_retreat_click_uses_live_bottom_left_button(self) -> None:
        recognition = self.nodes["公共-点击撤退"]["recognition"]
        self.assertEqual(recognition["param"]["roi"], [0, 1050, 250, 230])
        self.assertEqual(recognition["param"]["expected"], ["^(撤退|放弃)$"])
        confirmation = self.nodes["公共-确认撤退"]["recognition"]
        self.assertEqual(confirmation["param"]["roi"], [40, 780, 340, 280])
        self.assertEqual(confirmation["param"]["expected"], ["现在撤退"])

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

    def test_round_result_accepts_prepare_battle_button_without_waiting(self) -> None:
        for name in ("征服-轮间结果", "征服-轮间继续"):
            expected = self.nodes[name]["recognition"]["param"]["expected"]
            self.assertEqual(expected[0], "^准备战斗$")
        self.assertEqual(self.nodes["公共-等待新状态"]["rate_limit"], 300)

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

    def test_whole_match_continue_accepts_clipped_live_label(self) -> None:
        params = self.nodes["征服-结果继续"]["recognition"]["param"]
        self.assertEqual(params["roi"], [80, 650, 640, 590])
        self.assertIn("^下一$", params["expected"])

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

    def test_zero_energy_recognition_exists_for_battle_flow(self) -> None:
        node = self.nodes["公共-零能量"]
        self.assertEqual(node["recognition"]["type"], "OCR")
        self.assertIn("^[0O]$", node["recognition"]["param"]["expected"])


if __name__ == "__main__":
    unittest.main()
