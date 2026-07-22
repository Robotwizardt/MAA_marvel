from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image

from agent.recognitions.safe_entry import SafeEntry
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


class FakeRecognitionContext:
    def __init__(self, matches: set[str]) -> None:
        self.matches = matches

    def run_recognition(
        self,
        entry: str,
        image: object,
        pipeline_override: object = None,
    ) -> object:
        box = (1, 1, 10, 10) if entry in self.matches else None
        return SimpleNamespace(box=box)


class ConquestPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes()

    def test_navigation_and_tier_nodes_exist(self) -> None:
        required = {
            "征服-任务入口",
            "征服-初始化会话",
            "征服-打开模式列表",
            "征服-查找模式卡片",
            "征服-选择档位候选",
            "征服-准备试炼之地",
            "征服-准备白银",
            "征服-准备黄金",
            "征服-准备无限",
            "征服-安全入口确认",
            "征服-点击免费进入",
            "征服-点击门票进入",
            "征服-点击开战",
            "征服-确认卡组页面",
            "征服-确认使用卡组",
        }
        self.assertTrue(required.issubset(self.nodes), required - set(self.nodes))

    def test_mode_list_falls_back_to_safe_scroll(self) -> None:
        self.assertEqual(
            next_names(self.nodes["公共-模式列表"]),
            ["征服-查找模式卡片", "征服-滚动模式列表"],
        )

    def test_conquest_card_uses_exact_mode_title_ocr(self) -> None:
        node = self.nodes["征服-查找模式卡片"]
        self.assertEqual(node["recognition"]["type"], "OCR")
        self.assertEqual(node["recognition"]["param"]["roi"], [40, 150, 640, 1050])
        self.assertEqual(node["recognition"]["param"]["expected"], ["征服模式"])

    def test_mode_entry_waits_for_transition_then_requires_mode_list(self) -> None:
        opening = self.nodes["征服-打开模式列表"]
        self.assertGreaterEqual(opening["post_delay"], 3000)
        self.assertEqual(next_names(opening), ["公共-模式列表"])
        self.assertEqual(
            opening["recognition"]["param"]["roi"],
            [450, 1120, 140, 140],
        )

    def test_entry_clicks_have_only_safe_gate_as_direct_predecessor(self) -> None:
        for target in ("征服-点击免费进入", "征服-点击门票进入"):
            predecessors = {
                name for name, node in self.nodes.items() if target in next_names(node)
            }
            self.assertEqual(predecessors, {"征服-安全入口确认"})

    def test_ticket_evidence_and_click_share_current_button_ocr(self) -> None:
        evidence = self.nodes["征服-证据-门票可用"]["recognition"]["param"]
        click = self.nodes["征服-点击门票进入"]["recognition"]["param"]

        self.assertEqual(evidence, click)
        self.assertIn("[1-9][0-9]*/[1-9][0-9]*", evidence["expected"])
        self.assertEqual(evidence["roi"], [230, 930, 300, 150])
        self.assertEqual(self.nodes["征服-点击门票进入"]["action"]["type"], "Click")

    def test_proving_grounds_title_uses_live_ocr_text(self) -> None:
        for name in ("征服-试炼之地标题", "征服-确认试炼候选"):
            recognition = self.nodes[name]["recognition"]
            self.assertEqual(recognition["type"], "OCR")
            self.assertEqual(recognition["param"]["expected"], ["试炼之地"])

    def test_safe_gate_uses_custom_recognition(self) -> None:
        recognition = self.nodes["征服-安全入口确认"]["recognition"]
        self.assertEqual(recognition["type"], "Custom")
        self.assertEqual(
            recognition["param"]["custom_recognition"], "MarvelSafeEntry"
        )

    def test_paid_evidence_routes_only_to_rejection(self) -> None:
        for name in ("征服-金块入口", "征服-付费确认"):
            self.assertEqual(next_names(self.nodes[name]), ["征服-拒绝当前档位"])
            self.assertNotEqual(self.nodes[name].get("action"), "Click")

    def test_battle_click_uses_exact_ocr_inside_center_safe_area(self) -> None:
        node = self.nodes["征服-点击开战"]
        self.assertEqual(node["recognition"]["type"], "OCR")
        self.assertEqual(node["recognition"]["param"]["expected"], ["^开战$"])
        self.assertEqual(node["action"]["type"], "Click")
        self.assertNotIn("target", node["action"].get("param", {}))

        x, y, width, height = node["recognition"]["param"]["roi"]
        self.assertGreaterEqual(x, 250)
        self.assertLessEqual(x + width, 480)
        self.assertGreaterEqual(y, 950)
        self.assertLessEqual(y + height, 1100)

    def test_battle_click_requires_deck_confirmation_before_match_start(self) -> None:
        self.assertEqual(
            next_names(self.nodes["征服-点击开战"]),
            ["征服-确认卡组页面"],
        )
        page = self.nodes["征服-确认卡组页面"]
        self.assertEqual(page["recognition"]["type"], "OCR")
        self.assertIn("确认卡组", page["recognition"]["param"]["expected"])
        self.assertEqual(next_names(page), ["征服-确认使用卡组"])

        confirm = self.nodes["征服-确认使用卡组"]
        self.assertEqual(confirm["recognition"]["type"], "OCR")
        self.assertIn("^确认$", confirm["recognition"]["param"]["expected"])
        self.assertEqual(confirm["action"]["type"], "Click")
        self.assertNotIn("target", confirm["action"].get("param", {}))
        self.assertEqual(next_names(confirm), ["公共-比赛开始"])

    def test_prematch_shop_negative_sample_uses_exact_ocr(self) -> None:
        recognition = self.nodes["征服-赛前商店负样本"]["recognition"]
        self.assertEqual(recognition["type"], "OCR")
        self.assertEqual(recognition["param"]["expected"], ["^商店$"])

    def test_deck_confirmation_fixture_matches_reference_resolution(self) -> None:
        path = ROOT / "tests" / "fixtures" / "screens" / "conquest" / "deck_confirmation.png"
        with Image.open(path) as image:
            self.assertEqual(image.size, (720, 1280))

    def test_session_initialization_contains_all_default_fields(self) -> None:
        values = self.nodes["征服-初始化会话"]["action"]["param"][
            "custom_action_param"
        ]
        self.assertEqual(
            set(values),
            {
                "play_strategy",
                "max_tier",
                "no_ticket",
                "retreat_after_turn",
                "after_retreat",
                "snap_mode",
                "snap_probability",
                "max_matches",
                "max_minutes",
                "matchmaking_timeout_seconds",
                "auto_restart",
            },
        )

    def analyze_entry(self, tier: str, matches: set[str]):
        argv = SimpleNamespace(
            custom_recognition_param=f'{{"tier": "{tier}"}}',
            image=object(),
        )
        return SafeEntry().analyze(FakeRecognitionContext(matches), argv)

    def test_safe_entry_accepts_free_proving_grounds(self) -> None:
        result = self.analyze_entry(
            "proving_grounds", {"征服-证据-免费进入"}
        )
        self.assertIsNotNone(result.box)
        self.assertTrue(result.detail["safe"])

    def test_safe_entry_accepts_ticket_without_paid_evidence(self) -> None:
        result = self.analyze_entry("silver", {"征服-证据-门票可用"})
        self.assertIsNotNone(result.box)
        self.assertTrue(result.detail["safe"])

    def test_safe_entry_rejects_gold_even_when_ticket_is_visible(self) -> None:
        result = self.analyze_entry(
            "infinite",
            {"征服-证据-门票可用", "征服-证据-金块图标"},
        )
        self.assertIsNone(result.box)
        self.assertFalse(result.detail["safe"])

    def test_safe_entry_rejects_missing_evidence(self) -> None:
        result = self.analyze_entry("gold", set())
        self.assertIsNone(result.box)
        self.assertFalse(result.detail["safe"])


if __name__ == "__main__":
    unittest.main()
