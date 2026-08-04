from pathlib import Path
import unittest

from tools.validate_schema import load_jsonc


ROOT = Path(__file__).resolve().parents[1]


def next_names(node: dict[str, object]) -> list[str]:
    values = node.get("next", [])
    if isinstance(values, str):
        return [values]
    return [value for value in values if isinstance(value, str)]


class MailPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_jsonc(ROOT / "assets/resource/pipeline/mail/rewards.json")

    def test_entry_always_opens_mail_and_selects_inbox(self) -> None:
        entry = self.nodes["邮箱-任务入口"]
        self.assertEqual(
            entry["action"]["param"]["custom_action"],
            "MarvelWarmupScreencap",
        )
        self.assertEqual(next_names(entry), ["邮箱-初始化会话"])
        initializer = self.nodes["邮箱-初始化会话"]
        self.assertEqual(
            initializer["action"]["param"]["custom_action_param"],
            {"daily_routine": False},
        )
        self.assertEqual(next_names(initializer), ["邮箱-流程入口"])

        entry = self.nodes["邮箱-流程入口"]
        self.assertEqual(
            next_names(entry),
            ["邮箱-信息中心打开收件箱", "邮箱-主界面打开邮箱"],
        )
        self.assertGreater(entry["timeout"], 0)
        self.assertEqual(entry["on_error"], ["公共-安全停止"])

        opener = self.nodes["邮箱-主界面打开邮箱"]
        self.assertEqual(opener["recognition"]["type"], "And")
        self.assertEqual(
            opener["recognition"]["param"]["all_of"],
            ["公共-主界面"],
        )
        self.assertEqual(opener["recognition"]["param"]["box_index"], 0)
        self.assertEqual(
            opener["action"]["param"]["target"],
            [180, 20, 16, 16],
        )
        self.assertEqual(next_names(opener), ["邮箱-信息中心打开收件箱"])

        inbox_tab = self.nodes["邮箱-信息中心打开收件箱"]
        self.assertEqual(
            inbox_tab["recognition"]["param"]["all_of"],
            ["邮箱-信息中心标题", "邮箱-收件箱标签"],
        )
        self.assertEqual(inbox_tab["recognition"]["param"]["box_index"], 1)
        self.assertEqual(inbox_tab["action"]["type"], "Click")
        self.assertEqual(next_names(inbox_tab), ["邮箱-收件箱状态"])

    def test_red_dot_regions_are_small_and_clicks_require_page_evidence(self) -> None:
        for name, roi in (("邮箱-收件箱奖励红点", [1100, 145, 115, 800]),):
            with self.subTest(name=name):
                recognition = self.nodes[name]["recognition"]
                self.assertEqual(recognition["type"], "ColorMatch")
                self.assertEqual(recognition["param"]["roi"], roi)
                self.assertEqual(recognition["param"]["method"], 40)
                self.assertTrue(recognition["param"]["connected"])
                self.assertEqual(
                    recognition["param"]["lower"],
                    [[0, 150, 130], [170, 150, 130]],
                )
                self.assertEqual(
                    recognition["param"]["upper"],
                    [[15, 255, 255], [180, 255, 255]],
                )

        opener = self.nodes["邮箱-打开红点邮件"]
        self.assertEqual(opener["recognition"]["type"], "And")
        self.assertEqual(
            opener["recognition"]["param"]["all_of"],
            ["邮箱-收件箱页面", "邮箱-收件箱奖励红点"],
        )
        self.assertEqual(opener["recognition"]["param"]["box_index"], 1)
        self.assertEqual(opener["action"]["type"], "Click")

    def test_inbox_router_requires_the_inbox_page(self) -> None:
        inbox = self.nodes["邮箱-收件箱状态"]
        self.assertEqual(inbox["recognition"]["type"], "And")
        self.assertEqual(
            inbox["recognition"]["param"]["all_of"],
            ["邮箱-信息中心标题", "邮箱-收件箱标签"],
        )
        self.assertEqual(inbox["recognition"]["param"]["box_index"], 0)

    def test_claim_requires_both_visible_buttons_and_observes_claimed_state(self) -> None:
        all_claim = self.nodes["邮箱-全部领取奖励"]
        self.assertEqual(all_claim["recognition"]["param"]["expected"], ["^全部领取$"])
        self.assertEqual(all_claim["recognition"]["param"]["roi"], [740, 760, 440, 160])
        self.assertEqual(next_names(all_claim), ["邮箱-领取后状态"])

        claim = self.nodes["邮箱-奖励展示领取"]
        self.assertEqual(claim["recognition"]["param"]["expected"], ["^领取$"])
        self.assertEqual(claim["recognition"]["param"]["roi"], [760, 890, 400, 160])
        self.assertEqual(next_names(claim), ["邮箱-领取后状态"])

        claimed = self.nodes["邮箱-已领取状态"]
        self.assertEqual(claimed["recognition"]["param"]["expected"], ["^已领取$"])
        self.assertEqual(claimed["recognition"]["param"]["roi"], [740, 760, 440, 160])
        self.assertEqual(
            next_names(self.nodes["邮箱-领取后状态"]),
            ["邮箱-奖励展示领取", "邮箱-已领取详情关闭"],
        )

    def test_claimed_detail_returns_to_inbox_and_closes_when_no_red_dot_remains(self) -> None:
        detail_close = self.nodes["邮箱-已领取详情关闭"]
        self.assertEqual(detail_close["recognition"]["type"], "And")
        self.assertEqual(
            detail_close["recognition"]["param"]["all_of"],
            ["邮箱-已领取状态", "邮箱-关闭文本"],
        )
        self.assertEqual(detail_close["recognition"]["param"]["box_index"], 1)
        self.assertEqual(next_names(detail_close), ["邮箱-收件箱状态"])

        inbox = self.nodes["邮箱-收件箱状态"]
        self.assertEqual(
            next_names(inbox),
            ["邮箱-打开红点邮件", "邮箱-收件箱关闭"],
        )
        self.assertEqual(next_names(self.nodes["邮箱-收件箱关闭"]), ["邮箱-首页确认"])
        self.assertEqual(
            next_names(self.nodes["邮箱-首页确认"]),
            ["日常-邮箱完成路由", "日常-普通子任务完成"],
        )
        self.assertEqual(self.nodes["邮箱-首页确认"]["recognition"]["type"], "And")


if __name__ == "__main__":
    unittest.main()
