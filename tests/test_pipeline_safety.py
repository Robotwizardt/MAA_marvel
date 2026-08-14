from pathlib import Path
import unittest

from tools.validate_schema import load_jsonc


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "assets" / "resource" / "pipeline"


def action_type(node: dict[str, object]) -> str:
    action = node.get("action", "DoNothing")
    return action if isinstance(action, str) else str(action.get("type", "DoNothing"))


def action_param(node: dict[str, object]) -> dict[str, object]:
    action = node.get("action", {})
    if not isinstance(action, dict):
        return {}
    param = action.get("param", {})
    return param if isinstance(param, dict) else {}


def recognition_type(node: dict[str, object]) -> str:
    recognition = node.get("recognition", "DirectHit")
    if isinstance(recognition, str):
        return recognition
    return str(recognition.get("type", "DirectHit"))


class PipelineSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes: dict[str, dict[str, object]] = {}
        for path in PIPELINE_ROOT.rglob("*.json"):
            cls.nodes.update(load_jsonc(path))

    def test_coordinates_are_1920_by_1080_only(self) -> None:
        coordinate_keys = {"roi", "target", "begin", "end"}

        def check(value: object, key: str | None, node_name: str) -> None:
            if isinstance(value, dict):
                for child_key, child in value.items():
                    check(child, child_key, node_name)
            elif isinstance(value, list):
                if key in coordinate_keys and len(value) in (2, 4):
                    if len(value) == 2:
                        self.assertTrue(0 <= value[0] <= 1920, node_name)
                        self.assertTrue(0 <= value[1] <= 1080, node_name)
                    else:
                        x, y, width, height = value
                        self.assertTrue(0 <= x <= 1920, node_name)
                        self.assertTrue(0 <= y <= 1080, node_name)
                        self.assertTrue(0 <= width <= 1920, node_name)
                        self.assertTrue(0 <= height <= 1080, node_name)
                        self.assertLessEqual(x + width, 1920, node_name)
                        self.assertLessEqual(y + height, 1080, node_name)
                else:
                    for child in value:
                        check(child, key, node_name)

        for name, node in self.nodes.items():
            check(node, None, name)

    def test_clicks_are_never_blind_direct_hits(self) -> None:
        for name, node in self.nodes.items():
            if action_type(node) == "Click":
                self.assertNotEqual(recognition_type(node), "DirectHit", name)

    def test_task_reward_navigation_never_uses_blind_back(self) -> None:
        for name, node in self.nodes.items():
            if not name.startswith("公共-领奖-"):
                continue
            if action_type(node) == "ClickKey":
                self.assertNotEqual(recognition_type(node), "DirectHit", name)

    def test_only_android_back_key_is_allowed(self) -> None:
        for name, node in self.nodes.items():
            if action_type(node) == "ClickKey":
                self.assertEqual(action_param(node).get("key"), 4, name)

    def test_startup_popup_requires_page_evidence_and_clicks_close_region(self) -> None:
        node = self.nodes["公共-启动活动弹窗"]
        self.assertEqual(recognition_type(node), "And")
        self.assertEqual(action_type(node), "Click")
        self.assertEqual(
            node["recognition"]["param"]["all_of"],
            ["公共-启动活动弹窗页面证据", "公共-启动活动弹窗关闭图标"],
        )
        self.assertEqual(action_param(node)["target"], [1135, 115, 85, 90])
        self.assertEqual(node["next"], ["公共-识别当前页面"])

        evidence = self.nodes["公共-启动活动弹窗页面证据"]
        self.assertEqual(recognition_type(evidence), "Or")
        self.assertEqual(
            evidence["recognition"]["param"]["any_of"],
            ["公共-启动活动弹窗文案", "公共-启动活动弹窗前往文案"],
        )

    def test_launcher_opens_game_by_clicking_the_recognized_icon(self) -> None:
        node = self.nodes["公共-模拟器桌面重启游戏"]
        self.assertEqual(recognition_type(node), "OCR")
        self.assertEqual(action_type(node), "Click")
        self.assertEqual(node["max_hit"], 2)

    def test_sign_in_claim_and_close_click_only_exact_recognized_text(self) -> None:
        for node_name in (
            "公共-累计签到领取",
            "公共-累计签到奖励",
            "公共-活动中心关闭",
        ):
            node = self.nodes[node_name]
            self.assertEqual(recognition_type(node), "And")
            self.assertEqual(action_type(node), "Click")
            self.assertNotIn("target", action_param(node))
            self.assertEqual(node["recognition"]["param"]["box_index"], 1)
            self.assertEqual(node["next"], ["公共-识别当前页面"])

        claim_expected = self.nodes["公共-累计签到领取按钮"]["recognition"][
            "param"
        ]["expected"]
        self.assertEqual(claim_expected, ["^领取$", "^领取奖励$"])
        self.assertNotIn("^可补签$", claim_expected)
        self.assertEqual(
            self.nodes["公共-累计签到领取"]["recognition"]["param"][
                "all_of"
            ],
            ["公共-累计签到领取按钮紫色背景", "公共-累计签到领取按钮"],
        )
        self.assertEqual(
            self.nodes["公共-累计签到关闭按钮"]["recognition"]["param"][
                "expected"
            ],
            ["^关闭$"],
        )

    def test_app_actions_target_cn_package(self) -> None:
        for name, node in self.nodes.items():
            if action_type(node) in {"StartApp", "StopApp"}:
                self.assertEqual(
                    action_param(node).get("package"), "com.netease.ms", name
                )

    def test_recovery_never_routes_to_entry_clicks(self) -> None:
        for name, node in self.nodes.items():
            if "恢复" not in name:
                continue
            next_nodes = node.get("next", [])
            if isinstance(next_nodes, str):
                next_nodes = [next_nodes]
            for next_node in next_nodes:
                if isinstance(next_node, str):
                    self.assertNotIn("点击进入", next_node, name)

    def test_recovery_never_ends_the_task(self) -> None:
        for name, node in self.nodes.items():
            if "恢复" not in name and name != "公共-重新启动游戏":
                continue
            next_nodes = node.get("next", [])
            on_error = node.get("on_error", [])
            if isinstance(next_nodes, str):
                next_nodes = [next_nodes]
            if isinstance(on_error, str):
                on_error = [on_error]
            self.assertNotIn("公共-安全停止", next_nodes, name)
            self.assertNotIn("公共-安全停止", on_error, name)

    def test_conquest_loop_boundaries_always_recover_on_error(self) -> None:
        for name in (
            "公共-主界面",
            "公共-模式列表",
            "征服-打开模式列表",
            "征服-滚动模式列表",
            "征服-滚动模式列表-第2次",
            "征服-滚动模式列表-第3次",
            "征服-返回征服大厅",
        ):
            self.assertEqual(
                self.nodes[name].get("on_error"),
                ["公共-恢复决策"],
                name,
            )

    def test_paid_evidence_nodes_never_click(self) -> None:
        for name, node in self.nodes.items():
            if "金块" in name or "付费" in name:
                self.assertNotEqual(action_type(node), "Click", name)

    def test_task_reward_routers_have_finite_recovery_paths(self) -> None:
        for name in (
            "公共-领取任务奖励入口",
            "公共-领奖-等待任务页",
            "公共-领奖-扫描奖励",
            "公共-领奖-返回主页",
        ):
            node = self.nodes[name]
            self.assertGreater(node["timeout"], 0, name)
            self.assertTrue(node.get("on_error"), name)


if __name__ == "__main__":
    unittest.main()
