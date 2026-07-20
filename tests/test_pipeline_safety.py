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

    def test_coordinates_are_720_by_1280_only(self) -> None:
        coordinate_keys = {"roi", "target", "begin", "end"}

        def check(value: object, key: str | None, node_name: str) -> None:
            if isinstance(value, dict):
                for child_key, child in value.items():
                    check(child, child_key, node_name)
            elif isinstance(value, list):
                if key in coordinate_keys and len(value) in (2, 4):
                    if len(value) == 2:
                        self.assertTrue(0 <= value[0] <= 720, node_name)
                        self.assertTrue(0 <= value[1] <= 1280, node_name)
                    else:
                        x, y, width, height = value
                        self.assertTrue(0 <= x <= 720, node_name)
                        self.assertTrue(0 <= y <= 1280, node_name)
                        self.assertTrue(0 <= width <= 720, node_name)
                        self.assertTrue(0 <= height <= 1280, node_name)
                        self.assertLessEqual(x + width, 720, node_name)
                        self.assertLessEqual(y + height, 1280, node_name)
                else:
                    for child in value:
                        check(child, key, node_name)

        for name, node in self.nodes.items():
            check(node, None, name)

    def test_clicks_are_never_blind_direct_hits(self) -> None:
        for name, node in self.nodes.items():
            if action_type(node) == "Click":
                self.assertNotEqual(recognition_type(node), "DirectHit", name)

    def test_only_android_back_key_is_allowed(self) -> None:
        for name, node in self.nodes.items():
            if action_type(node) == "ClickKey":
                self.assertEqual(action_param(node).get("key"), 4, name)

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

    def test_paid_evidence_nodes_never_click(self) -> None:
        for name, node in self.nodes.items():
            if "金块" in name or "付费" in name:
                self.assertNotEqual(action_type(node), "Click", name)


if __name__ == "__main__":
    unittest.main()
