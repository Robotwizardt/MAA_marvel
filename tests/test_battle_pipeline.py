from pathlib import Path
import re
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
    def test_detail_close_uses_landscape_close_text(self) -> None:
        recognition = self.nodes["公共-详情关闭按钮"]["recognition"]
        self.assertEqual(recognition["type"], "OCR")
        self.assertEqual(recognition["param"]["expected"], ["^关闭$"])
        self.assertEqual(recognition["param"]["roi"], [0, 900, 320, 180])

    def test_training_entries_are_single_turn_and_player_turn_guarded(self) -> None:
        training = load_jsonc(
            ROOT / "assets/resource/pipeline/common/training_battle.json"
        )
        for entry in ("训练-OCR选牌测试入口",):
            node = training[entry]
            self.assertEqual(node["recognition"]["type"], "OCR")
            self.assertEqual(node["recognition"]["param"]["roi"], [1580, 900, 340, 180])
            self.assertEqual(node["next"], ["训练-执行单回合出牌"])
        self.assertEqual(
            training["训练-执行单回合出牌"]["next"], ["训练-结束当前回合"]
        )

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
            "公共-放置中状态",
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
        self.assertNotIn("征服-结果继续", battle_wait)
        self.assertIn("征服-整场结果", battle_wait)
        self.assertIn("征服-返回征服大厅", battle_wait)

        state_wait = next_names(self.nodes["公共-等待新状态"])
        self.assertNotIn("征服-结果继续", state_wait)
        self.assertIn("征服-整场结果", state_wait)

        node = self.nodes["公共-战斗继续"]
        self.assertEqual(node["recognition"]["type"], "OCR")
        self.assertIn("^继续$", node["recognition"]["param"]["expected"])
        self.assertEqual(node["action"]["type"], "Click")
        self.assertNotIn("target", node["action"].get("param", {}))

    def test_bootstrap_waits_on_loading_screen_and_recognizes_all_tier_lobbies(self) -> None:
        bootstrap = next_names(self.nodes["公共-识别当前页面"])
        self.assertEqual(bootstrap[0], "公共-启动加载中")
        self.assertEqual(bootstrap[1], "公共-启动品牌加载中")
        self.assertEqual(bootstrap[2], "公共-启动活动弹窗")
        self.assertEqual(bootstrap[3], "公共-累计签到领取")
        self.assertEqual(bootstrap[4], "公共-累计签到奖励")
        self.assertEqual(bootstrap[5], "公共-活动中心关闭")
        self.assertIn("征服-白银标题", bootstrap)
        self.assertIn("征服-黄金标题", bootstrap)
        loading = self.nodes["公共-启动加载中"]
        self.assertEqual(
            loading["recognition"]["param"]["expected"],
            ["^prod[ -]?v?[0-9.]+[+]?[0-9]*[.]?$"],
        )
        self.assertEqual(next_names(loading), ["公共-识别当前页面"])
        brand_loading = self.nodes["公共-启动品牌加载中"]
        self.assertEqual(
            brand_loading["recognition"]["param"]["expected"],
            ["^MARVEL$", "^终极逆转$"],
        )
        self.assertEqual(
            brand_loading["recognition"]["param"]["roi"],
            [650, 400, 620, 360],
        )
        self.assertEqual(next_names(brand_loading), ["公共-识别当前页面"])
        popup = self.nodes["公共-启动活动弹窗"]
        self.assertEqual(
            popup["recognition"]["type"],
            "And",
        )
        self.assertEqual(
            popup["recognition"]["param"]["all_of"],
            ["公共-启动活动弹窗页面证据", "公共-启动活动弹窗关闭图标"],
        )
        self.assertEqual(popup["recognition"]["param"]["box_index"], 0)
        self.assertEqual(
            popup["action"],
            {"type": "Click", "param": {"target": [1135, 115, 85, 90]}},
        )

        popup_evidence = self.nodes["公共-启动活动弹窗页面证据"]["recognition"]
        self.assertEqual(popup_evidence["type"], "Or")
        self.assertEqual(
            popup_evidence["param"]["any_of"],
            ["公共-启动活动弹窗文案", "公共-启动活动弹窗前往文案"],
        )

        popup_text = self.nodes["公共-启动活动弹窗文案"]["recognition"]
        self.assertEqual(popup_text["type"], "OCR")
        self.assertEqual(popup_text["param"]["roi"], [800, 800, 400, 105])
        self.assertEqual(
            popup_text["param"]["expected"],
            ["^今日内不再弹出$"],
        )

        popup_close = self.nodes["公共-启动活动弹窗关闭图标"]["recognition"]
        self.assertEqual(popup_close["type"], "ColorMatch")
        self.assertEqual(popup_close["param"]["roi"], [1138, 135, 60, 65])
        self.assertEqual(popup_close["param"]["method"], 40)
        self.assertEqual(popup_close["param"]["lower"], [[0, 0, 180]])
        self.assertEqual(popup_close["param"]["upper"], [[180, 80, 255]])
        self.assertEqual(popup_close["param"]["count"], 120)
        self.assertTrue(popup_close["param"]["connected"])
        self.assertEqual(popup_close["param"]["order_by"], "Area")
        sign_in_title = self.nodes["公共-累计签到奖励文案"]["recognition"]
        self.assertEqual(sign_in_title["type"], "OCR")
        self.assertEqual(sign_in_title["param"]["roi"], [800, 150, 330, 110])
        self.assertEqual(sign_in_title["param"]["expected"], ["^累签大奖$"])

        sign_in_claim_text = self.nodes["公共-累计签到领取按钮"]["recognition"]
        self.assertEqual(sign_in_claim_text["type"], "OCR")
        self.assertEqual(sign_in_claim_text["param"]["roi"], [840, 900, 260, 140])
        self.assertEqual(
            sign_in_claim_text["param"]["expected"],
            ["^领取$", "^领取奖励$"],
        )
        purple = self.nodes["公共-累计签到领取按钮紫色背景"]["recognition"]
        self.assertEqual(purple["type"], "ColorMatch")
        self.assertEqual(purple["param"]["roi"], [835, 910, 252, 122])
        self.assertEqual(purple["param"]["method"], 40)
        self.assertEqual(purple["param"]["lower"], [[120, 100, 50]])
        self.assertEqual(purple["param"]["upper"], [[150, 255, 255]])
        self.assertEqual(purple["param"]["count"], 3000)
        self.assertTrue(purple["param"]["connected"])

        sign_in_close_text = self.nodes["公共-累计签到关闭按钮"]["recognition"]
        self.assertEqual(sign_in_close_text["type"], "OCR")
        self.assertEqual(sign_in_close_text["param"]["roi"], [120, 950, 160, 100])
        self.assertEqual(sign_in_close_text["param"]["expected"], ["^关闭$"])

        claim = self.nodes["公共-累计签到领取"]
        self.assertEqual(claim["recognition"]["type"], "And")
        self.assertEqual(
            claim["recognition"]["param"]["all_of"],
            ["公共-累计签到领取按钮紫色背景", "公共-累计签到领取按钮"],
        )
        self.assertEqual(claim["recognition"]["param"]["box_index"], 1)
        self.assertEqual(claim["action"], {"type": "Click"})
        self.assertEqual(next_names(claim), ["公共-识别当前页面"])

        reward = self.nodes["公共-累计签到奖励"]
        self.assertEqual(reward["recognition"]["type"], "And")
        self.assertEqual(
            reward["recognition"]["param"]["all_of"],
            ["公共-累计签到奖励文案", "公共-累计签到关闭按钮"],
        )
        self.assertEqual(reward["recognition"]["param"]["box_index"], 1)
        self.assertEqual(reward["action"], {"type": "Click"})
        self.assertEqual(next_names(reward), ["公共-识别当前页面"])

        activity_title = self.nodes["公共-活动中心文案"]["recognition"]
        self.assertEqual(activity_title["type"], "OCR")
        self.assertEqual(activity_title["param"]["roi"], [800, 0, 320, 100])
        self.assertEqual(activity_title["param"]["expected"], ["^活动中心$"])
        activity_close = self.nodes["公共-活动中心关闭"]
        self.assertEqual(activity_close["recognition"]["type"], "And")
        self.assertEqual(
            activity_close["recognition"]["param"]["all_of"],
            ["公共-活动中心文案", "公共-累计签到关闭按钮"],
        )
        self.assertEqual(activity_close["recognition"]["param"]["box_index"], 1)
        self.assertEqual(activity_close["action"], {"type": "Click"})
        self.assertEqual(next_names(activity_close), ["公共-识别当前页面"])

    def test_initial_start_and_auto_restart_share_popup_aware_router(self) -> None:
        self.assertEqual(
            next_names(self.nodes["公共-启动游戏"]),
            ["公共-识别当前页面"],
        )
        self.assertEqual(
            next_names(self.nodes["公共-恢复重启"]),
            ["公共-停止游戏"],
        )
        self.assertEqual(
            next_names(self.nodes["公共-停止游戏"]),
            ["公共-重新启动游戏"],
        )
        self.assertEqual(
            next_names(self.nodes["公共-重新启动游戏"]),
            ["公共-识别当前页面"],
        )
        self.assertEqual(
            self.nodes["公共-重新启动游戏"]["on_error"],
            ["公共-恢复等待"],
        )
        self.assertNotIn(
            "公共-安全停止",
            next_names(self.nodes["公共-恢复决策"]),
        )
        router = next_names(self.nodes["公共-识别当前页面"])
        self.assertLess(
            router.index("公共-启动活动弹窗"),
            router.index("公共-主界面"),
        )

    def test_android_launcher_restarts_the_game_from_exact_app_label(self) -> None:
        node = self.nodes["公共-模拟器桌面重启游戏"]
        recognition = node["recognition"]
        self.assertEqual(recognition["type"], "OCR")
        self.assertEqual(recognition["param"]["roi"], [1040, 130, 340, 190])
        self.assertEqual(
            recognition["param"]["expected"],
            ["^漫威终极逆转$"],
        )
        self.assertEqual(node["action"], {"type": "Click"})
        self.assertEqual(node["max_hit"], 2)
        self.assertEqual(next_names(node), ["公共-识别当前页面"])
        self.assertEqual(node["on_error"], ["公共-恢复决策"])

        for router_name in (
            "公共-识别当前页面",
            "公共-等待战斗状态",
            "公共-出牌后状态",
            "公共-等待新状态",
        ):
            with self.subTest(router=router_name):
                self.assertIn(
                    "公共-模拟器桌面重启游戏",
                    next_names(self.nodes[router_name]),
                )

    def test_exit_confirmation_cancel_uses_live_exact_ocr(self) -> None:
        node = self.nodes["公共-退出游戏取消"]
        self.assertEqual(node["recognition"]["type"], "OCR")
        self.assertEqual(node["recognition"]["param"]["roi"], [500, 500, 920, 420])
        self.assertEqual(node["recognition"]["param"]["expected"], ["^否$"])

    def test_reconnect_uses_live_bottom_button(self) -> None:
        node = self.nodes["公共-重新连接"]
        self.assertEqual(node["recognition"]["param"]["roi"], [500, 700, 920, 330])
        self.assertEqual(
            node["recognition"]["param"]["expected"],
            ["^重新连接$", "^重连$"],
        )

    def test_play_turn_is_reached_only_after_stop_gate(self) -> None:
        self.assertEqual(
            self.predecessors("公共-执行出牌"),
            {"公共-停止跳过", "公共-继续出牌门禁"},
        )
        recognition = self.nodes["公共-停止命中"]["recognition"]
        self.assertEqual(
            recognition["param"]["custom_recognition_param"]["command"],
            "should_stop",
        )

    def test_snap_click_is_reached_only_from_snap_gate(self) -> None:
        self.assertEqual(
            self.predecessors("公共-点击SNAP"),
            {"公共-SNAP命中", "公共-最终回合SNAP命中"},
        )
        recognition = self.nodes["公共-SNAP命中"]["recognition"]
        self.assertEqual(
            recognition["param"]["custom_recognition_param"]["command"],
            "should_snap_first",
        )
        final = self.nodes["公共-最终回合SNAP命中"]["recognition"]
        self.assertEqual(
            final["param"]["custom_recognition_param"]["command"],
            "should_snap_final",
        )
        marker = self.nodes["公共-最终回合标记"]
        self.assertEqual(marker["recognition"]["param"]["roi"], [1640, 960, 260, 100])
        self.assertRegex("最终回合", marker["recognition"]["param"]["expected"][0])
        self.assertEqual(
            next_names(self.nodes["公共-SNAP判断"]),
            ["公共-最终回合标记", "公共-SNAP命中", "公共-SNAP跳过"],
        )
        click_recognition = self.nodes["公共-点击SNAP"]["recognition"]
        self.assertEqual(click_recognition["param"]["roi"], [1340, 580, 240, 240])
        self.assertEqual(click_recognition["param"]["expected"], ["^[1248]$"])

    def test_snap_gates_recover_instead_of_ending_the_whole_task(self) -> None:
        for name in ("公共-SNAP命中", "公共-最终回合SNAP命中"):
            with self.subTest(name=name):
                self.assertEqual(
                    next_names(self.nodes[name]),
                    [
                        "公共-点击SNAP",
                        "公共-结束回合门禁",
                        "公共-继续出牌门禁",
                        "公共-模拟器桌面重启游戏",
                    ],
                )
                self.assertEqual(self.nodes[name]["timeout"], 3000)
                self.assertEqual(
                    self.nodes[name]["on_error"],
                    ["公共-恢复决策"],
                )

    def test_post_play_state_router_handles_end_turn_and_transitions(self) -> None:
        next_nodes = self.nodes["公共-出牌后状态"]["next"]
        self.assertEqual(next_nodes[0], "征服-每日经验上限")
        self.assertEqual(next_nodes[1], "公共-结束回合门禁")
        self.assertEqual(next_nodes[2], "公共-继续出牌门禁")
        self.assertEqual(self.nodes["公共-出牌后状态"]["rate_limit"], 200)
        self.assertIn("公共-战斗继续", next_nodes)
        self.assertIn("公共-新回合", next_nodes)
        self.assertNotIn("公共-结束回合", self.nodes["公共-等待新状态"]["next"])
        self.assertEqual(self.nodes["公共-SNAP跳过"]["next"], ["公共-出牌后状态"])
        self.assertEqual(self.nodes["公共-点击SNAP"]["next"], ["公共-出牌后状态"])

    def test_end_turn_is_reachable_only_through_custom_gate(self) -> None:
        gate = self.nodes["公共-结束回合门禁"]
        self.assertEqual(gate["recognition"]["type"], "Custom")
        self.assertEqual(
            gate["recognition"]["param"]["custom_recognition_param"]["command"],
            "can_end_turn",
        )
        self.assertEqual(next_names(gate), ["公共-结束回合"])
        continue_gate = self.nodes["公共-继续出牌门禁"]
        self.assertEqual(
            continue_gate["recognition"]["param"]["custom_recognition_param"]["command"],
            "must_continue_playing",
        )
        self.assertEqual(next_names(continue_gate), ["公共-执行出牌"])

        direct_parents = [
            name
            for name, node in self.nodes.items()
            if name != "公共-结束回合门禁"
            and "公共-结束回合" in next_names(node)
        ]
        self.assertEqual(direct_parents, [])

    def test_live_battle_routers_never_scan_whole_match_continue_button(self) -> None:
        for name in (
            "公共-等待战斗状态",
            "公共-出牌后状态",
            "公共-等待新状态",
        ):
            with self.subTest(name=name):
                self.assertNotIn("征服-结果继续", next_names(self.nodes[name]))

    def test_daily_pass_limit_can_stop_or_continue_from_result_flow(self) -> None:
        limit = self.nodes["征服-每日经验上限"]
        self.assertEqual(limit["recognition"]["type"], "OCR")
        self.assertEqual(
            limit["recognition"]["param"]["roi"], [420, 180, 1080, 720]
        )
        self.assertNotEqual(
            limit["recognition"]["param"]["roi"], [0, 0, 1920, 1080]
        )
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
        self.assertEqual(recognition["param"]["roi"], [440, 620, 170, 170])
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
        self.assertEqual(recognition["param"]["roi"], [0, 900, 340, 180])
        self.assertEqual(recognition["param"]["expected"], ["^(撤退|放弃)$"])
        confirmation = self.nodes["公共-确认撤退"]["recognition"]
        self.assertEqual(confirmation["param"]["roi"], [600, 300, 720, 600])
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
            "公共-点击撤退",
            "公共-确认撤退",
            "公共-点击整场认输",
            "公共-确认整场认输",
            "征服-轮间继续",
            "征服-点击胜利结算下一步",
            "征服-结果继续",
        ):
            with self.subTest(name=name):
                node = self.nodes[name]
                self.assertEqual(node["recognition"]["type"], "OCR")
                self.assertEqual(node["action"]["type"], "Click")
                self.assertNotIn("target", node["action"].get("param", {}))

    def test_end_turn_accepts_live_ocr_variant_in_button_area(self) -> None:
        params = self.nodes["公共-结束回合文字"]["recognition"]["param"]
        self.assertIn("^结束回[合会]$", params["expected"])
        self.assertIsNone(re.fullmatch(params["expected"][0], "取消结束回合"))
        x, y, width, height = params["roi"]
        self.assertGreaterEqual(x, 1580)
        self.assertGreaterEqual(y, 900)
        self.assertEqual(x + width, 1920)
        self.assertLessEqual(y + height, 1080)

        end_turn = self.nodes["公共-结束回合"]
        self.assertEqual(end_turn["recognition"]["type"], "And")
        self.assertEqual(
            end_turn["recognition"]["param"]["all_of"],
            ["公共-结束回合文字", "公共-激活回合按钮颜色"],
        )
        self.assertEqual(end_turn["recognition"]["param"]["box_index"], 0)
        self.assertEqual(end_turn["action"]["type"], "Click")
        self.assertNotIn("target", end_turn["action"].get("param", {}))

    def test_active_turn_color_accepts_dim_live_button(self) -> None:
        recognition = self.nodes["公共-激活回合按钮颜色"]["recognition"]
        self.assertEqual(recognition["type"], "ColorMatch")
        params = recognition["param"]
        self.assertEqual(params["roi"], [1670, 960, 180, 70])
        self.assertEqual(params["method"], 40)
        self.assertEqual(params["count"], 3500)

    def test_placing_state_uses_end_turn_button_area(self) -> None:
        params = self.nodes["公共-放置中状态"]["recognition"]["param"]
        self.assertEqual(params["expected"], ["放置中"])
        self.assertEqual(params["roi"], [1580, 900, 340, 180])

    def test_round_result_accepts_prepare_battle_button_without_waiting(self) -> None:
        for name in ("征服-轮间结果", "征服-轮间继续"):
            params = self.nodes[name]["recognition"]["param"]
            expected = params["expected"]
            self.assertEqual(expected, ["^准备战斗[？?]?$", "^下一轮$"])
            self.assertRegex("准备战斗？", expected[0])
            self.assertRegex("准备战斗", expected[0])
            self.assertNotIn("^下一步$", expected)
            self.assertNotIn("^继续$", expected)
            self.assertEqual(params["roi"], [1620, 930, 280, 150])
            self.assertTrue(params["only_rec"])
        self.assertEqual(self.nodes["公共-等待新状态"]["rate_limit"], 300)
        self.assertEqual(
            next_names(self.nodes["征服-轮间继续"]),
            ["公共-比赛开始"],
        )
        self.assertEqual(
            next_names(self.nodes["征服-整场结果"]), ["征服-结果后状态"]
        )
        self.assertIn(
            "征服-轮间结果",
            next_names(self.nodes["征服-结果后状态-右下按钮"]),
        )

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

    def test_task_reward_flow_is_recognition_guarded_and_returns_through_home(self) -> None:
        entry = self.nodes["公共-领取任务奖励入口"]
        self.assertEqual(
            next_names(entry),
            [
                "公共-领奖-首页稳定态",
                "[JumpBack]公共-领奖-征服大厅返回",
                "[JumpBack]公共-领奖-模式列表返回",
            ],
        )
        self.assertGreater(entry["timeout"], 0)

        stable_home = self.nodes["公共-领奖-首页稳定态"]
        self.assertEqual(stable_home["recognition"]["type"], "And")
        self.assertEqual(
            stable_home["recognition"]["param"]["all_of"],
            ["公共-首页开战按钮", "公共-首页标签"],
        )
        self.assertEqual(
            next_names(stable_home),
            ["公共-领奖-首页入口"],
        )
        self.assertGreater(stable_home["timeout"], 0)
        self.assertEqual(
            stable_home["on_error"],
            ["公共-领奖-首页入口重试"],
        )
        retry = self.nodes["公共-领奖-首页入口重试"]
        self.assertEqual(retry["action"]["type"], "DoNothing")
        self.assertEqual(
            next_names(retry),
            ["公共-领取任务奖励入口"],
        )

        home_text = self.nodes["公共-领奖-查看所有每日任务"]["recognition"]
        self.assertEqual(home_text["type"], "OCR")
        self.assertEqual(
            home_text["param"]["expected"],
            ["^查看所有每日任务$", "^查看所有任务$"],
        )
        self.assertEqual(home_text["param"]["roi"], [1500, 320, 420, 740])
        home_click = self.nodes["公共-领奖-首页入口"]
        self.assertEqual(home_click["recognition"]["type"], "And")
        self.assertEqual(home_click["action"]["type"], "Click")

        page = self.nodes["公共-领奖-任务页证据"]["recognition"]
        self.assertEqual(page["type"], "OCR")
        self.assertEqual(page["param"]["expected"], ["^每周挑战$"])
        completed = self.nodes["公共-领奖-点击已完成任务"]
        self.assertEqual(completed["recognition"]["type"], "Custom")
        self.assertEqual(
            completed["recognition"]["param"]["custom_recognition"],
            "MarvelDailyTaskReward",
        )
        self.assertEqual(completed["action"]["type"], "Click")
        self.assertEqual(
            self.nodes["公共-领奖-领取弹窗"]["recognition"]["param"]["expected"],
            ["^领取$"],
        )

        record = self.nodes["公共-领奖-记录检查完成"]
        self.assertEqual(
            record["action"]["param"]["custom_action_param"]["event"],
            "task_rewards_checked",
        )
        self.assertEqual(
            self.predecessors("公共-领奖-记录检查完成"),
            {"公共-领奖-首页确认"},
        )
        self.assertEqual(
            next_names(record),
            ["日常-任务奖励完成路由", "公共-主界面"],
        )

    def test_round_result_rejects_whole_match_next_step_label(self) -> None:
        for name in ("征服-轮间结果", "征服-轮间继续"):
            params = self.nodes[name]["recognition"]["param"]
            expected = params["expected"]
            self.assertNotIn("^下一步$", expected, name)
            self.assertNotIn("^继续$", expected, name)
            x, _, width, _ = params["roi"]
            self.assertGreaterEqual(x + width, 1880, name)
            self.assertLessEqual(x + width, 1920, name)
            self.assertTrue(params["only_rec"], name)

    def test_whole_match_continue_rejects_in_battle_next_turn_text(self) -> None:
        params = self.nodes["征服-结果继续"]["recognition"]["param"]
        self.assertEqual(params["roi"], [860, 900, 240, 130])
        self.assertTrue(params["only_rec"])
        self.assertEqual(params["expected"], ["^继续$", "^下一步$", "^领取$"])
        self.assertEqual(
            next_names(self.nodes["征服-结果继续"]), ["征服-结算后等待弹窗"]
        )
        wait_for_popup = self.nodes["征服-结算后等待弹窗"]
        self.assertEqual(wait_for_popup["action"]["type"], "DoNothing")
        self.assertGreaterEqual(wait_for_popup["post_delay"], 5000)
        self.assertEqual(next_names(wait_for_popup), ["征服-结果后状态"])
        for pattern in params["expected"]:
            self.assertIsNone(re.fullmatch(pattern, "下一回合：2"))
            self.assertIsNone(re.fullmatch(pattern, "下一"))

    def test_victory_result_waits_for_next_step_before_lobby_recording(self) -> None:
        result = self.nodes["征服-整场结果"]
        self.assertEqual(result["recognition"]["type"], "Or")
        self.assertEqual(
            result["recognition"]["param"]["any_of"],
            ["征服-整场结果顶部标题", "征服-胜利结算页"],
        )
        self.assertEqual(next_names(result), ["征服-结果后状态"])

        top_title = self.nodes["征服-整场结果顶部标题"]["recognition"]["param"]
        self.assertEqual(top_title["roi"], [580, 40, 600, 160])
        self.assertEqual(
            top_title["expected"], ["征服完成", "战斗胜利", "战斗失败"]
        )
        self.assertTrue(top_title["only_rec"])

        victory = self.nodes["征服-胜利结算页"]
        self.assertEqual(
            victory["recognition"]["param"]["roi"], [740, 510, 440, 120]
        )
        self.assertTrue(victory["recognition"]["param"]["only_rec"])
        self.assertEqual(next_names(victory), ["征服-点击胜利结算下一步"])
        self.assertIn("post_wait_freezes", victory)

        click = self.nodes["征服-点击胜利结算下一步"]
        self.assertEqual(click["recognition"]["param"]["roi"], [840, 840, 240, 130])
        self.assertTrue(click["recognition"]["param"]["only_rec"])
        self.assertEqual(click["recognition"]["param"]["expected"], ["^下一步$"])
        self.assertEqual(click["action"]["type"], "Click")
        self.assertIn("pre_wait_freezes", click)
        self.assertEqual(next_names(click), ["征服-胜利结算确认后状态"])

        after_click = self.nodes["征服-胜利结算确认后状态"]
        self.assertEqual(next_names(after_click), ["征服-结果后状态"])
        self.assertNotIn("征服-记录整场完成", next_names(after_click))

        central = self.nodes["征服-结果后状态"]
        self.assertIn("征服-胜利结算页", next_names(central))
        self.assertIn("征服-结果继续", next_names(central))
        self.assertEqual(
            central["on_error"], ["征服-结果后状态-右下按钮"]
        )
        right = self.nodes["征服-结果后状态-右下按钮"]
        self.assertEqual(
            next_names(right), ["征服-失败结算下一步", "征服-轮间结果"]
        )
        self.assertEqual(right["on_error"], ["征服-结果后状态-大厅"])

        lobby = self.nodes["征服-返回征服大厅"]
        self.assertEqual(next_names(lobby), ["征服-大厅结算缓冲"])
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

        season = self.nodes["征服-新赛季弹窗"]
        self.assertEqual(season["recognition"]["type"], "OCR")
        self.assertTrue(season["recognition"]["param"]["only_rec"])
        self.assertEqual(next_names(season), ["征服-新赛季开战按钮"])
        self.assertEqual(
            self.nodes["征服-新赛季开战按钮"]["action"]["type"], "Click"
        )

        conversion = self.nodes["征服-赛季结束转化"]
        self.assertEqual(conversion["recognition"]["type"], "OCR")
        self.assertTrue(conversion["recognition"]["param"]["only_rec"])
        self.assertEqual(next_names(conversion), ["征服-赛季结束转化按钮"])
        self.assertEqual(
            self.nodes["征服-赛季结束转化按钮"]["action"]["type"], "Click"
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
