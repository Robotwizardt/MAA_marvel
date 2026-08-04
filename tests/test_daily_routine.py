from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from agent.actions.configure_session import ConfigureSession
from agent.runtime.event_listener import ROOT_ENTRIES
from agent.session.config import SessionConfig
from agent.session.state import SessionState
from tools.validate_schema import load_jsonc


ROOT = Path(__file__).resolve().parents[1]


def next_names(node: dict[str, object]) -> list[str]:
    values = node.get("next", [])
    if isinstance(values, str):
        return [values]
    return [value for value in values if isinstance(value, str)]


class EmptyConfigContext:
    def get_node_data(self, name: str) -> dict[str, object]:
        return {}


class DailyRoutinePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pipeline = ROOT / "assets" / "resource" / "pipeline"
        cls.daily = load_jsonc(pipeline / "daily" / "routine.json")
        cls.bootstrap = load_jsonc(pipeline / "common" / "bootstrap.json")
        cls.home = load_jsonc(pipeline / "common" / "home.json")
        cls.rewards = load_jsonc(pipeline / "common" / "task_rewards.json")
        cls.mail = load_jsonc(pipeline / "mail" / "rewards.json")
        cls.conquest_results = load_jsonc(
            pipeline / "conquest" / "results.json"
        )
        cls.ladder = load_jsonc(pipeline / "ladder" / "ladder.json")

    def test_entry_initializes_daily_session_before_bootstrap(self) -> None:
        self.assertIn("日常-任务入口", ROOT_ENTRIES)
        self.assertEqual(
            self.daily["日常-任务入口"]["action"]["param"]["custom_action"],
            "MarvelWarmupScreencap",
        )
        self.assertEqual(
            next_names(self.daily["日常-任务入口"]),
            ["日常-初始化会话"],
        )
        initializer = self.daily["日常-初始化会话"]
        values = initializer["action"]["param"]["custom_action_param"]
        self.assertTrue(values["daily_routine"])
        self.assertEqual(values["play_strategy"], "ocr")
        self.assertEqual(next_names(initializer), ["公共-启动游戏"])

    def test_bootstrap_handles_popups_then_prefers_daily_recovery(self) -> None:
        route = next_names(self.bootstrap["公共-识别当前页面"])
        self.assertLess(
            route.index("公共-启动品牌加载中"),
            route.index("日常-首页开始"),
        )
        for popup in (
            "公共-启动活动弹窗",
            "公共-累计签到领取",
            "公共-累计签到奖励",
            "公共-活动中心关闭",
        ):
            self.assertLess(route.index(popup), route.index("日常-首页开始"))
        for daily_node, normal_node in (
            ("日常-恢复任务页", "公共-主界面"),
            ("日常-恢复邮箱页", "公共-主界面"),
            ("日常-恢复通行证页", "公共-主界面"),
            ("日常-首页开始", "公共-主界面"),
        ):
            with self.subTest(daily_node=daily_node):
                self.assertLess(route.index(daily_node), route.index(normal_node))

    def test_reward_mail_pass_and_battle_handoffs_form_one_chain(self) -> None:
        self.assertEqual(
            next_names(self.daily["日常-首页开始"]),
            ["公共-领取任务奖励入口"],
        )
        self.assertEqual(
            next_names(self.rewards["公共-领奖-记录检查完成"]),
            ["日常-任务奖励完成路由", "公共-主界面"],
        )
        self.assertEqual(
            next_names(self.daily["日常-任务奖励完成路由"]),
            ["邮箱-流程入口"],
        )
        self.assertEqual(
            next_names(self.mail["邮箱-首页确认"]),
            ["日常-邮箱完成路由", "日常-普通子任务完成"],
        )
        self.assertEqual(
            next_names(self.daily["日常-邮箱完成路由"]),
            ["通行证-任务入口"],
        )
        self.assertEqual(
            next_names(self.daily["日常-记录完成"]),
            ["公共-主界面"],
        )
        self.assertEqual(
            next_names(self.home["公共-主界面"]),
            ["天梯-主页模式命中", "征服-关闭训练模式", "征服-打开模式列表"],
        )

    def test_missing_rewards_and_pass_entry_are_safe_skip_paths(self) -> None:
        self.assertEqual(
            next_names(self.rewards["公共-领奖-首页稳定态"]),
            ["公共-领奖-首页入口"],
        )
        self.assertEqual(
            next_names(self.daily["通行证-任务入口"]),
            ["通行证-首页打开入口", "通行证-首页无入口", "通行证-页面处理"],
        )
        page = self.daily["通行证-页面处理"]
        self.assertEqual(page["recognition"]["type"], "And")
        self.assertEqual(
            page["recognition"]["param"]["all_of"],
            ["通行证-页面证据"],
        )
        self.assertEqual(
            next_names(self.daily["通行证-首页无入口"]),
            ["日常-记录完成"],
        )

    def test_completed_battle_returns_to_pending_daily_before_next_match(
        self,
    ) -> None:
        self.assertEqual(
            next_names(self.conquest_results["征服-结束后停止判断"])[0],
            "日常-对局后继续处理",
        )
        self.assertEqual(
            next_names(self.ladder["天梯-停止判断"])[0],
            "日常-对局后继续处理",
        )
        daily_route = self.daily["日常-对局后继续处理"]
        self.assertEqual(
            daily_route["recognition"]["param"]["custom_recognition_param"],
            {"command": "daily_routine_pending"},
        )
        self.assertEqual(
            next_names(daily_route),
            ["日常-返回主页状态"],
        )
        self.assertEqual(
            next_names(self.daily["日常-返回主页状态"]),
            [
                "日常-首页开始",
                "日常-模式列表返回主页",
                "日常-征服大厅返回主页",
            ],
        )
        mode_back = self.daily["日常-模式列表返回主页"]
        self.assertEqual(
            mode_back["recognition"]["param"]["all_of"],
            ["日常-待处理", "公共-模式列表"],
        )
        self.assertEqual(mode_back["action"]["type"], "ClickKey")
        self.assertEqual(mode_back["action"]["param"]["key"], 4)
        self.assertEqual(next_names(mode_back), ["日常-返回主页状态"])
        back = self.daily["日常-征服大厅返回主页"]
        self.assertEqual(
            back["recognition"]["param"]["all_of"],
            ["日常-待处理", "征服-大厅档位标题"],
        )
        self.assertEqual(back["action"]["type"], "ClickKey")
        self.assertEqual(back["action"]["param"]["key"], 4)
        self.assertEqual(next_names(back), ["日常-返回主页状态"])

    def test_pass_claims_only_exact_free_reward_text(self) -> None:
        claim = self.daily["通行证-领取奖励"]
        self.assertEqual(
            claim["recognition"]["param"]["all_of"],
            ["通行证-页面证据", "通行证-领取文字"],
        )
        for name in ("通行证-奖励展示领取", "通行证-领取文字"):
            self.assertEqual(
                self.daily[name]["recognition"]["param"]["expected"],
                ["^领取$"],
            )
        serialized = str(self.daily)
        for forbidden in ("购买", "解锁", "充值"):
            self.assertNotIn(forbidden, serialized)

    @patch("agent.actions.configure_session.DIAGNOSTICS")
    @patch("agent.actions.configure_session.STORE")
    def test_independent_mail_does_not_restore_or_overwrite_battle_checkpoint(
        self,
        store: MagicMock,
        diagnostics: MagicMock,
    ) -> None:
        store.configure.return_value = SessionState(SessionConfig(), started_at=0.0)
        store.last_configure_restored.return_value = False

        result = ConfigureSession().run(
            EmptyConfigContext(),
            SimpleNamespace(
                node_name="邮箱-初始化会话",
                custom_action_param='{"daily_routine": false}',
            ),
        )

        self.assertTrue(result)
        self.assertFalse(store.configure.call_args.kwargs["checkpoint_enabled"])

    @patch("agent.actions.configure_session.DIAGNOSTICS")
    @patch("agent.actions.configure_session.STORE")
    def test_manual_daily_start_skips_stale_battle_checkpoint(
        self,
        store: MagicMock,
        diagnostics: MagicMock,
    ) -> None:
        state = SessionState(SessionConfig(daily_routine=True), started_at=0.0)
        state.daily_routine_completed_date = "2026-08-02"
        store.configure.return_value = state
        store.last_configure_restored.return_value = True

        result = ConfigureSession().run(
            EmptyConfigContext(),
            SimpleNamespace(
                node_name="日常-初始化会话",
                custom_action_param='{"daily_routine": true}',
            ),
        )

        self.assertTrue(result)
        self.assertIsNone(state.daily_routine_completed_date)
        self.assertTrue(store.configure.call_args.kwargs["checkpoint_enabled"])
        self.assertFalse(store.configure.call_args.kwargs["restore_checkpoint"])
        store.persist_checkpoint.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
