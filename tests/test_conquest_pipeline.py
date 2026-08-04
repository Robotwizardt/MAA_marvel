from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image

from agent.actions.route_conquest_tier import RouteConquestTier
from agent.recognitions.safe_entry import SafeEntry
from agent.session.config import ConquestTier
from agent.runtime.store import STORE
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
    def __init__(self, matches: set[str], ticket_texts: tuple[str, ...] = ()) -> None:
        self.matches = matches
        self.ticket_texts = ticket_texts

    def run_recognition(
        self,
        entry: str,
        image: object,
        pipeline_override: object = None,
    ) -> object:
        box = (1, 1, 10, 10) if entry in self.matches else None
        return SimpleNamespace(hit=entry in self.matches, box=box)

    def run_recognition_direct(self, reco_type, reco_param, image):
        return SimpleNamespace(
            filtered_results=[SimpleNamespace(text=text) for text in self.ticket_texts]
        )


class FakeActionContext:
    def override_next(self, node_name: str, next_nodes: list[str]) -> list[str]:
        del node_name
        return next_nodes


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
            "征服-无可用档位等待",
            "征服-大厅回流判断",
            "征服-大厅中对局已完成",
            "征服-准备试炼之地",
            "征服-准备白银",
            "征服-准备黄金",
            "征服-准备无限",
            "征服-安全入口确认",
            "征服-点击免费进入",
            "征服-点击门票进入",
            "征服-选择指定卡组",
            "征服-验证指定卡组",
            "征服-点击开战",
            "征服-确认卡组页面",
            "征服-确认使用卡组",
        }
        self.assertTrue(required.issubset(self.nodes), required - set(self.nodes))

    def test_root_entry_warms_up_screencap_before_routing(self) -> None:
        entry = self.nodes["征服-任务入口"]
        action = entry["action"]
        self.assertEqual(action["type"], "Custom")
        self.assertEqual(
            action["param"]["custom_action"],
            "MarvelWarmupScreencap",
        )
        self.assertGreaterEqual(
            action["param"]["custom_action_param"]["timeout_ms"],
            60000,
        )
        self.assertEqual(next_names(entry), ["征服-初始化会话"])

    def test_mode_list_falls_back_to_safe_scroll(self) -> None:
        self.assertEqual(
            next_names(self.nodes["公共-模式列表"]),
            ["征服-查找模式卡片", "征服-滚动模式列表"],
        )
        self.assertEqual(
            self.nodes["公共-模式列表"]["on_error"],
            ["公共-恢复决策"],
        )

        scroll_chain = (
            ("征服-滚动模式列表", "征服-滚动模式列表-第2次"),
            ("征服-滚动模式列表-第2次", "征服-滚动模式列表-第3次"),
            ("征服-滚动模式列表-第3次", "公共-恢复决策"),
        )
        for name, fallback in scroll_chain:
            with self.subTest(name=name):
                node = self.nodes[name]
                self.assertNotIn("max_hit", node)
                self.assertEqual(
                    next_names(node),
                    ["征服-查找模式卡片", fallback],
                )
                self.assertEqual(node["on_error"], ["公共-恢复决策"])

    def test_conquest_card_uses_exact_mode_title_ocr(self) -> None:
        node = self.nodes["征服-查找模式卡片"]
        self.assertEqual(node["recognition"]["type"], "OCR")
        self.assertEqual(node["recognition"]["param"]["roi"], [620, 100, 700, 900])
        self.assertEqual(node["recognition"]["param"]["expected"], ["^征服模式$"])

    def test_mode_entry_waits_for_transition_then_requires_mode_list(self) -> None:
        opening = self.nodes["征服-打开模式列表"]
        self.assertGreaterEqual(opening["post_delay"], 3000)
        self.assertEqual(
            next_names(opening),
            [
                "征服-进行中页面",
                "征服-试炼之地标题",
                "征服-白银标题",
                "征服-黄金标题",
                "征服-无限标题",
                "征服-赛前页面",
                "公共-模式列表",
            ],
        )
        self.assertEqual(opening["recognition"]["type"], "OCR")
        self.assertEqual(opening["recognition"]["param"]["expected"], ["^游戏模式$"])
        self.assertNotIn("target", opening["action"].get("param", {}))

    def test_home_requires_battle_button_and_home_tab(self) -> None:
        recognition = self.nodes["公共-主界面"]["recognition"]
        self.assertEqual(recognition["type"], "And")
        self.assertEqual(
            recognition["param"]["all_of"],
            ["公共-首页开战按钮", "公共-首页标签"],
        )
        self.assertEqual(
            self.nodes["公共-首页标签"]["recognition"]["param"]["expected"],
            ["主页$"],
        )

    def test_conquest_disables_training_before_opening_mode_list(self) -> None:
        self.assertEqual(
            next_names(self.nodes["公共-主界面"]),
            ["天梯-主页模式命中", "征服-关闭训练模式", "征服-打开模式列表"],
        )
        toggle = self.nodes["征服-关闭训练模式"]
        self.assertEqual(toggle["recognition"]["type"], "ColorMatch")
        self.assertEqual(toggle["recognition"]["param"]["roi"], [1030, 760, 150, 120])
        self.assertEqual(toggle["action"]["type"], "Click")
        self.assertEqual(next_names(toggle), ["征服-打开模式列表"])

    def test_entry_clicks_have_only_safe_gate_as_direct_predecessor(self) -> None:
        for target in ("征服-点击免费进入", "征服-点击门票进入"):
            predecessors = {
                name for name, node in self.nodes.items() if target in next_names(node)
            }
            self.assertEqual(predecessors, {"征服-安全入口确认"})

    def test_ticket_evidence_reads_zero_but_click_requires_positive_count(self) -> None:
        evidence = self.nodes["征服-证据-门票可用"]["recognition"]["param"]
        click = self.nodes["征服-点击门票进入"]["recognition"]["param"]

        self.assertEqual(evidence["roi"], click["roi"])
        self.assertEqual(evidence["roi"], [800, 760, 360, 190])
        self.assertIn("[0-9]+/1", evidence["expected"])
        self.assertIn("[1-9][0-9]*/1", click["expected"])
        self.assertEqual(self.nodes["征服-点击门票进入"]["action"]["type"], "Click")

    def test_proving_grounds_title_uses_live_ocr_text(self) -> None:
        for name in ("征服-试炼之地标题", "征服-确认试炼候选"):
            recognition = self.nodes[name]["recognition"]
            self.assertEqual(recognition["type"], "OCR")
            self.assertEqual(recognition["param"]["expected"], ["试炼之地"])
            self.assertEqual(recognition["param"]["roi"], [700, 40, 520, 180])

    def test_exhausted_tier_candidates_restart_and_reseed_when_enabled(self) -> None:
        STORE.configure({}, now=0.0, checkpoint_enabled=False)
        self.assertEqual(STORE.next_tier_candidate(), ConquestTier.PROVING_GROUNDS)

        result = RouteConquestTier().run(
            FakeActionContext(),
            SimpleNamespace(node_name="征服-选择档位候选"),
        )

        self.assertEqual(result, ["公共-恢复重启"])
        self.assertEqual(STORE.tier_candidates(), (ConquestTier.PROVING_GROUNDS,))

    def test_exhausted_tier_candidates_wait_when_restart_is_disabled(self) -> None:
        STORE.configure({"auto_restart": False}, now=0.0, checkpoint_enabled=False)
        self.assertEqual(STORE.next_tier_candidate(), ConquestTier.PROVING_GROUNDS)

        result = RouteConquestTier().run(
            FakeActionContext(),
            SimpleNamespace(node_name="征服-选择档位候选"),
        )

        self.assertEqual(result, ["征服-无可用档位等待"])
        self.assertEqual(STORE.tier_candidates(), (ConquestTier.PROVING_GROUNDS,))

    def test_lobby_title_records_an_interrupted_match_before_reselecting_a_tier(self) -> None:
        for name in (
            "征服-试炼之地标题",
            "征服-白银标题",
            "征服-黄金标题",
            "征服-无限标题",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    next_names(self.nodes[name]), ["征服-大厅回流判断"]
                )

        gate = self.nodes["征服-大厅中对局已完成"]["recognition"]["param"]
        self.assertEqual(gate["custom_recognition"], "MarvelSessionGate")
        self.assertEqual(
            gate["custom_recognition_param"]["command"], "match_in_progress"
        )
        self.assertEqual(
            next_names(self.nodes["征服-大厅中对局已完成"]),
            ["征服-记录整场完成"],
        )

    def test_exhausted_tier_candidates_wait_then_retry_without_safe_stop(self) -> None:
        selector = self.nodes["征服-选择档位候选"]
        self.assertEqual(
            next_names(selector)[-1], "征服-无可用档位等待"
        )
        wait = self.nodes["征服-无可用档位等待"]
        self.assertEqual(wait["action"]["type"], "DoNothing")
        self.assertGreaterEqual(wait["post_delay"], 20_000)
        self.assertEqual(next_names(wait), ["征服-选择档位候选"])

    def test_tier_selection_uses_carousel_swipes_instead_of_dots(self) -> None:
        confirmations = {
            "征服-准备试炼之地": "征服-确认试炼候选",
            "征服-准备白银": "征服-从试炼选择白银",
            "征服-准备黄金": "征服-从试炼选择黄金第一步",
            "征服-准备无限": "征服-从试炼选择无限第一步",
        }
        for base_name, confirmation in confirmations.items():
            chain = [
                base_name,
                f"{base_name}-第2次滑动",
                f"{base_name}-第3次滑动",
                f"{base_name}-第4次滑动",
            ]
            for index, name in enumerate(chain):
                with self.subTest(name=name):
                    node = self.nodes[name]
                    self.assertNotIn("max_hit", node)
                    self.assertEqual(node["timeout"], 10_000)
                    self.assertEqual(
                        node["on_error"],
                        ["征服-胜利结算页", "公共-恢复决策"],
                    )
                    if index == 0:
                        self.assertEqual(node["action"]["type"], "Swipe")
                        self.assertEqual(
                            node["action"]["param"]["begin"], [850, 450]
                        )
                        self.assertEqual(
                            node["action"]["param"]["end"], [1150, 450]
                        )
                    else:
                        self.assertEqual(node["recognition"]["type"], "OCR")
                        self.assertEqual(node["action"]["type"], "Swipe")
                        self.assertEqual(
                            node["action"]["param"]["begin"], [850, 450]
                        )
                        self.assertEqual(
                            node["action"]["param"]["end"], [1150, 450]
                        )

                    fallback = (
                        chain[index + 1]
                        if index + 1 < len(chain)
                        else "征服-拒绝当前档位"
                    )
                    self.assertEqual(next_names(node), [confirmation, fallback])

    def test_higher_tiers_are_selected_from_canonical_proving_ground(self) -> None:
        expected_swipes = {
            "征服-从试炼选择白银": "征服-确认白银候选",
            "征服-从白银选择黄金": "征服-确认黄金候选",
            "征服-从黄金选择无限": "征服-确认无限候选",
        }
        for name, next_name in expected_swipes.items():
            with self.subTest(name=name):
                node = self.nodes[name]
                self.assertEqual(node["action"]["type"], "Swipe")
                self.assertEqual(node["action"]["param"]["begin"], [1150, 450])
                self.assertEqual(node["action"]["param"]["end"], [850, 450])
                self.assertEqual(node["next"], [next_name])
                self.assertEqual(node["timeout"], 10_000)
                self.assertEqual(
                    node["on_error"],
                    ["征服-胜利结算页", "公共-恢复决策"],
                )

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
        self.assertEqual(node["recognition"]["type"], "And")
        self.assertEqual(
            node["recognition"]["param"]["all_of"],
            ["征服-赛前对战编号", "征服-赛前开战按钮"],
        )
        self.assertEqual(node["recognition"]["param"]["box_index"], 1)
        self.assertEqual(node["action"]["type"], "Click")
        self.assertEqual(node["pre_delay"], 2500)
        self.assertNotIn("target", node["action"].get("param", {}))

        button = self.nodes["征服-赛前开战按钮"]["recognition"]["param"]
        self.assertEqual(button["expected"], ["^开战$"])
        x, y, width, height = button["roi"]
        self.assertGreaterEqual(x, 800)
        self.assertLessEqual(x + width, 1120)
        self.assertGreaterEqual(y, 760)
        self.assertLessEqual(y + height, 980)

    def test_deck_selection_is_optional_and_falls_back_to_current_deck(self) -> None:
        self.assertEqual(
            next_names(self.nodes["征服-赛前页面"]),
            ["征服-首次选择卡组判断", "征服-点击开战"],
        )
        gate = self.nodes["征服-首次选择卡组判断"]
        self.assertEqual(
            gate["recognition"]["param"]["custom_recognition_param"]["command"],
            "should_select_deck",
        )
        self.assertEqual(next_names(gate), ["征服-打开卡组列表"])
        self.assertEqual(gate["on_error"], ["征服-点击开战"])
        opener = self.nodes["征服-打开卡组列表"]
        self.assertEqual(
            opener["action"]["param"]["target"],
            [755, 785, 65, 65],
        )
        self.assertEqual(
            next_names(opener),
            ["征服-选择指定卡组", "征服-卡组未找到回退"],
        )
        selection = self.nodes["征服-选择指定卡组"]
        self.assertEqual(selection["recognition"]["type"], "OCR")
        self.assertEqual(
            selection["recognition"]["param"]["expected"],
            ["^(?!0$)0$"],
        )
        self.assertEqual(
            selection["recognition"]["param"]["roi"],
            [80, 580, 1760, 340],
        )
        self.assertEqual(selection["action"]["type"], "Click")
        self.assertEqual(next_names(selection), ["征服-验证指定卡组"])
        self.assertEqual(selection["on_error"], ["征服-卡组未找到回退"])
        verification = self.nodes["征服-验证指定卡组"]
        self.assertEqual(verification["recognition"]["type"], "OCR")
        self.assertEqual(
            verification["recognition"]["param"]["expected"],
            ["^(?!0$)0$"],
        )
        self.assertEqual(
            verification["recognition"]["param"]["roi"],
            [700, 760, 180, 170],
        )
        self.assertEqual(
            next_names(verification),
            ["征服-卡组选择完成标记"],
        )
        self.assertEqual(
            verification["on_error"],
            ["征服-卡组选择验证失败回退"],
        )
        self.assertEqual(
            next_names(self.nodes["征服-卡组选择完成标记"]),
            ["征服-点击开战"],
        )
        self.assertEqual(
            self.nodes["征服-卡组选择完成标记"]["action"]["param"]
            ["custom_action_param"]["event"],
            "deck_selection_succeeded",
        )
        self.assertEqual(
            next_names(self.nodes["征服-卡组未找到回退"]),
            ["征服-卡组未找到返回"],
        )
        self.assertEqual(
            self.nodes["征服-卡组未找到回退"]["action"]["param"]
            ["custom_action_param"]["event"],
            "deck_selection_fallback_not_found",
        )
        self.assertEqual(
            next_names(self.nodes["征服-卡组未找到返回"]),
            ["征服-点击开战"],
        )
        self.assertEqual(
            next_names(self.nodes["征服-卡组选择验证失败回退"]),
            ["征服-点击开战"],
        )

    def test_prematch_requires_conquest_match_number_and_battle_button(self) -> None:
        recognition = self.nodes["征服-赛前页面"]["recognition"]
        self.assertEqual(recognition["type"], "And")
        self.assertEqual(
            recognition["param"]["all_of"],
            ["征服-赛前对战编号", "征服-赛前开战按钮"],
        )
        self.assertEqual(recognition["param"]["box_index"], 1)
        self.assertEqual(
            self.nodes["征服-赛前对战编号"]["recognition"]["param"]["expected"],
            ["对战[0-9]+"],
        )
        self.assertEqual(
            self.nodes["征服-赛前开战按钮"]["recognition"]["param"]["expected"],
            ["^开战$"],
        )

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
            self.assertEqual(image.size, (1920, 1080))

    def test_session_initialization_contains_all_default_fields(self) -> None:
        values = self.nodes["征服-初始化会话"]["action"]["param"][
            "custom_action_param"
        ]
        self.assertEqual(
            set(values),
            {
                "play_strategy",
                "lane_order",
                "game_mode",
                "max_tier",
                "reserve_silver_tickets",
                "reserve_gold_tickets",
                "reserve_infinite_tickets",
                "stop_on_daily_pass_limit",
                "retreat_after_turn",
                "after_retreat",
                "snap_mode",
                "snap_probability",
                "claim_task_rewards_hours",
                "matchmaking_timeout_seconds",
                "auto_restart",
                "deck_name",
            },
        )
        self.assertFalse(values["stop_on_daily_pass_limit"])
        self.assertEqual(values["claim_task_rewards_hours"], 0)
        self.assertNotIn("Config_MaxMatches", self.nodes)
        self.assertNotIn("Config_MaxMinutes", self.nodes)
        claim_config = self.nodes["Config_ClaimTaskRewardsHours"]["action"]["param"][
            "custom_action_param"
        ]
        self.assertEqual(claim_config, {"claim_task_rewards_hours": 0})

    def test_reward_claim_is_checked_only_after_full_match_and_stop_gate(self) -> None:
        self.assertEqual(
            next_names(self.nodes["征服-记录整场完成"]),
            ["征服-结束后停止判断"],
        )
        self.assertEqual(
            next_names(self.nodes["征服-结束后停止判断"]),
            [
                "日常-对局后继续处理",
                "征服-结束后停止命中",
                "征服-结束后继续",
            ],
        )
        self.assertEqual(
            next_names(self.nodes["征服-结束后继续"]),
            ["征服-任务奖励到期", "征服-选择档位候选"],
        )
        gate = self.nodes["征服-任务奖励到期"]["recognition"]["param"]
        self.assertEqual(gate["custom_recognition"], "MarvelSessionGate")
        self.assertEqual(
            gate["custom_recognition_param"]["command"],
            "task_rewards_due",
        )
        predecessors = {
            name
            for name, node in self.nodes.items()
            if "征服-任务奖励到期" in next_names(node)
        }
        self.assertEqual(predecessors, {"征服-结束后继续"})
        self.assertEqual(
            next_names(self.nodes["征服-任务奖励到期"]),
            ["公共-领取任务奖励入口"],
        )

    def test_victory_confirmation_precedes_match_completed_event(self) -> None:
        result = self.nodes["征服-整场结果"]
        self.assertEqual(result["recognition"]["type"], "Or")
        self.assertEqual(
            result["recognition"]["param"]["any_of"],
            ["征服-整场结果顶部标题", "征服-胜利结算页"],
        )
        self.assertEqual(next_names(result), ["征服-结果后状态"])

        top_title = self.nodes["征服-整场结果顶部标题"]["recognition"]["param"]
        self.assertEqual(top_title["roi"], [580, 40, 600, 160])
        self.assertTrue(top_title["only_rec"])

        confirmation = self.nodes["征服-点击胜利结算下一步"]
        self.assertEqual(confirmation["recognition"]["type"], "OCR")
        self.assertEqual(
            confirmation["recognition"]["param"]["expected"], ["^下一步$"]
        )
        self.assertTrue(confirmation["recognition"]["param"]["only_rec"])
        self.assertEqual(confirmation["action"]["type"], "Click")
        self.assertEqual(
            next_names(confirmation), ["征服-胜利结算确认后状态"]
        )

        failure_confirmation = self.nodes["征服-失败结算下一步"]
        self.assertEqual(failure_confirmation["recognition"]["type"], "OCR")
        self.assertEqual(
            failure_confirmation["recognition"]["param"]["roi"],
            [1620, 930, 280, 150],
        )
        self.assertTrue(failure_confirmation["recognition"]["param"]["only_rec"])
        self.assertEqual(
            failure_confirmation["recognition"]["param"]["expected"], ["^下一步$"]
        )
        self.assertEqual(failure_confirmation["action"]["type"], "Click")
        self.assertEqual(
            next_names(failure_confirmation), ["征服-结算后等待弹窗"]
        )

        self.assertEqual(
            next_names(self.nodes["征服-结果继续"]), ["征服-结算后等待弹窗"]
        )
        wait_for_popup = self.nodes["征服-结算后等待弹窗"]
        self.assertEqual(wait_for_popup["action"]["type"], "DoNothing")
        self.assertGreaterEqual(wait_for_popup["post_delay"], 5000)
        self.assertEqual(next_names(wait_for_popup), ["征服-结果后状态"])

        after_confirmation = self.nodes["征服-胜利结算确认后状态"]
        self.assertEqual(next_names(after_confirmation), ["征服-结果后状态"])
        self.assertNotIn("征服-记录整场完成", next_names(after_confirmation))

        season = self.nodes["征服-赛季结束转化"]
        self.assertEqual(season["recognition"]["type"], "OCR")
        self.assertTrue(season["recognition"]["param"]["only_rec"])
        self.assertEqual(next_names(season), ["征服-赛季结束转化按钮"])
        self.assertEqual(
            self.nodes["征服-赛季结束转化按钮"]["action"]["type"], "Click"
        )
        new_season = self.nodes["征服-新赛季弹窗"]
        self.assertEqual(new_season["recognition"]["type"], "OCR")
        self.assertTrue(new_season["recognition"]["param"]["only_rec"])
        self.assertEqual(next_names(new_season), ["征服-新赛季开战按钮"])
        self.assertEqual(
            self.nodes["征服-新赛季开战按钮"]["action"]["type"], "Click"
        )

        self.assertEqual(
            next_names(self.nodes["征服-返回征服大厅"]),
            ["征服-大厅结算缓冲"],
        )
        lobby = self.nodes["征服-返回征服大厅"]["recognition"]
        self.assertEqual(lobby["type"], "And")
        self.assertEqual(
            lobby["param"]["all_of"],
            ["征服-大厅档位标题"],
        )
        self.assertEqual(lobby["param"]["box_index"], 0)
        self.assertEqual(
            self.nodes["征服-返回征服大厅"]["on_error"],
            ["公共-恢复决策"],
        )
        buffer = self.nodes["征服-大厅结算缓冲"]
        self.assertGreaterEqual(buffer["post_delay"], 5000)
        self.assertEqual(
            next_names(buffer),
            ["征服-胜利结算页", "征服-大厅完成确认"],
        )
        self.assertEqual(
            next_names(self.nodes["征服-大厅完成确认"]),
            ["征服-记录整场完成"],
        )

        for name in ("征服-轮间结果", "征服-轮间继续"):
            expected = self.nodes[name]["recognition"]["param"]["expected"]
            self.assertNotIn("^下一步$", expected, name)
            self.assertNotIn("^继续$", expected, name)

        central = self.nodes["征服-结果后状态"]
        self.assertEqual(
            next_names(central),
            [
                "征服-每日经验上限",
                "征服-胜利结算页",
                "[JumpBack]征服-赛季结束转化",
                "[JumpBack]征服-新赛季弹窗",
                "征服-结果继续",
            ],
        )
        self.assertEqual(central["on_error"], ["征服-结果后状态-右下按钮"])
        right = self.nodes["征服-结果后状态-右下按钮"]
        self.assertEqual(
            next_names(right), ["征服-失败结算下一步", "征服-轮间结果"]
        )
        self.assertEqual(right["on_error"], ["征服-结果后状态-大厅"])
        lobby_stage = self.nodes["征服-结果后状态-大厅"]
        self.assertEqual(next_names(lobby_stage), ["征服-返回征服大厅"])
        self.assertEqual(lobby_stage["on_error"], ["征服-结果后状态-天梯结果"])
        ladder_result_stage = self.nodes["征服-结果后状态-天梯结果"]
        self.assertEqual(next_names(ladder_result_stage), ["天梯-整场结果"])
        self.assertEqual(
            ladder_result_stage["on_error"], ["征服-结果后状态-天梯主页"]
        )
        ladder_home_stage = self.nodes["征服-结果后状态-天梯主页"]
        self.assertEqual(next_names(ladder_home_stage), ["天梯-返回主页"])
        self.assertEqual(ladder_home_stage["on_error"], ["公共-恢复决策"])

        groups = (
            central,
            right,
            lobby_stage,
            ladder_result_stage,
            ladder_home_stage,
        )
        for group in groups:
            batch_eligible = []
            for raw_name in next_names(group):
                name = raw_name.removeprefix("[JumpBack]")
                recognition = self.nodes[name].get("recognition", {})
                if recognition.get("type") != "OCR":
                    continue
                param = recognition.get("param", {})
                if not param.get("only_rec") and not param.get("color_filter"):
                    batch_eligible.append(name)
            self.assertLessEqual(len(batch_eligible), 1)
            if batch_eligible:
                self.assertEqual(batch_eligible, ["征服-每日经验上限"])

        fixed_ocr_nodes = (
            "征服-整场结果顶部标题",
            "征服-胜利结算页",
            "征服-点击胜利结算下一步",
            "征服-失败结算下一步",
            "征服-轮间结果",
            "征服-轮间继续",
            "征服-赛季结束转化",
            "征服-赛季结束转化按钮",
            "征服-新赛季弹窗",
            "征服-新赛季开战按钮",
            "征服-结果继续",
        )
        for name in fixed_ocr_nodes:
            param = self.nodes[name]["recognition"]["param"]
            self.assertTrue(param["only_rec"], name)
            self.assertGreaterEqual(param["roi"][3], 100, name)

        self.assertNotEqual(
            self.nodes["征服-每日经验上限"]["recognition"]["param"]["roi"],
            [0, 0, 1920, 1080],
        )

    def analyze_entry(
        self,
        tier: str,
        matches: set[str],
        ticket_texts: tuple[str, ...] = (),
    ):
        STORE.configure({}, now=0.0)
        argv = SimpleNamespace(
            custom_recognition_param=f'{{"tier": "{tier}"}}',
            image=object(),
        )
        return SafeEntry().analyze(FakeRecognitionContext(matches, ticket_texts), argv)

    def test_safe_entry_accepts_free_proving_grounds(self) -> None:
        result = self.analyze_entry(
            "proving_grounds", {"征服-证据-免费进入"}
        )
        self.assertIsNotNone(result.box)
        self.assertTrue(result.detail["safe"])

    def test_safe_entry_accepts_ticket_above_reserve_without_paid_evidence(self) -> None:
        result = self.analyze_entry("silver", set(), ("2/1",))
        self.assertIsNotNone(result.box)
        self.assertTrue(result.detail["safe"])

    def test_safe_entry_rejects_gold_even_when_ticket_is_visible(self) -> None:
        result = self.analyze_entry(
            "infinite",
            {"征服-证据-金块图标"},
            ("2/1",),
        )
        self.assertIsNone(result.box)
        self.assertFalse(result.detail["safe"])

    def test_safe_entry_rejects_missing_evidence(self) -> None:
        result = self.analyze_entry("gold", set())
        self.assertIsNone(result.box)
        self.assertFalse(result.detail["safe"])


if __name__ == "__main__":
    unittest.main()
