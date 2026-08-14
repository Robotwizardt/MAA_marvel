import random
import time

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.recognitions.card_selection import (
    BattleHand,
    DetectedCard,
    HandFrameTracker,
    MINIMUM_CONFIDENCE,
    scan_battle_hand,
)
from agent.recognitions.lane_state import scan_lane_states
from agent.runtime.store import STORE
from agent.session.config import LaneOrder, SnapMode
from agent.strategies.ocr import CardCandidate, choose_card
from agent.strategies.model import Point


# Maa 原生 1920×1080 横屏坐标。三个落点位于玩家侧场地区域，
# 避开场地说明文字和右下角结束回合按钮。
LANE_TARGETS = (Point(710, 730), Point(960, 730), Point(1210, 730))
# 单回合安全上限。正常一回合不可能成功打出 12 张牌；此限制用于防止异常循环。
MAX_SUCCESSFUL_PLAYS = 6
# 500ms 会被游戏识别成长按并打开卡牌详情；快速拖动才会进入放牌状态。
SWIPE_DURATION_MS = 280
# 同一场地至少连续证明两次无法放牌，才在本回合将其屏蔽。一次失败可能只是
# 动画、起点偏差或场上卡牌详情误触，不能据此永久放弃该场地。
LANE_FAILURES_BEFORE_BLOCK = 2
# “结束回合”按钮在拖牌动画开始前可能短暂保留。若没有观察到“放置中”，
# 必须连续多帧识别到结束回合才认为拖牌没有触发或画面已经稳定。
PLAY_READY_CONFIRMATIONS = 4
PLAY_RESOLUTION_TIMEOUT_SECONDS = 8.0
POST_PLAY_VERIFY_RETRIES = 6
READY_POST_PLAY_VERIFY_RETRIES = 3
# 未找到可支付牌时复查一帧。每帧内部已有整条快扫和重叠窗口兜底，
# 两帧足以覆盖动画，同时限制空过和满场地时的最坏等待时间。
SCAN_RETRIES = 2
SCAN_RETRY_DELAY_SECONDS = 0.12
DETAIL_CLOSE_POINT = Point(358, 1201)
DETAIL_CLOSE_DELAY_SECONDS = 0.5
DIRECT_END_TURN_DELAY_SECONDS = 0.2


def lane_targets_for_order(order: LaneOrder) -> tuple[Point, ...]:
    """按界面配置返回本次放牌尝试顺序；随机模式每张牌重新洗牌。"""
    if order is LaneOrder.RIGHT_TO_LEFT:
        return tuple(reversed(LANE_TARGETS))
    if order is LaneOrder.RANDOM:
        return tuple(random.sample(LANE_TARGETS, k=len(LANE_TARGETS)))
    return LANE_TARGETS


def _is_detail_overlay(image: object) -> bool:
    """横屏详情页使用 OCR“关闭”，此函数仅保留基础画面尺寸校验。"""
    pixels = np.asarray(image)
    return (
        pixels.ndim == 3
        and pixels.shape[2] >= 3
        and pixels.shape[0] == 1080
        and pixels.shape[1] == 1920
    )


def _adaptive_wait() -> AdaptiveFrameWait:
    return AdaptiveFrameWait(
        PLAY_ACTIVITY_ROI,
        initial_seconds=DYNAMIC_WAIT_INITIAL_SECONDS,
        maximum_seconds=DYNAMIC_WAIT_MAXIMUM_SECONDS,
        sleep=time.sleep,
    )


def _capture_frame(
    controller: object,
    performance: PerformanceTrace | None,
    phase: str,
) -> object:
    started = time.perf_counter()
    image = controller.post_screencap().get(wait=True)
    if performance is not None:
        performance.record(f"{phase}.screencap", time.perf_counter() - started)
    return image


def _wait_for_visual_change(
    controller: object,
    before: object,
    performance: PerformanceTrace | None,
    phase: str,
) -> bool:
    waiter = _adaptive_wait()
    waiter.prime(before)
    deadline = time.monotonic() + DETAIL_CHANGE_TIMEOUT_SECONDS
    for _ in range(DETAIL_CHANGE_MAX_POLLS):
        image = _capture_frame(controller, performance, phase)
        delay = waiter.wait_if_static(image, trace=performance, phase=phase)
        if delay == 0.0:
            return True
        if time.monotonic() >= deadline:
            break
    return False


def _close_detail_overlay(
    context: Context,
    controller: object,
    image: object,
    performance: PerformanceTrace | None = None,
) -> bool:
    """横屏详情页左下角固定显示“关闭”，OCR 命中后点击其文本框。"""
    if not _is_detail_overlay(image):
        return False
    recognition_started = time.perf_counter()
    matched = context.run_recognition("公共-详情关闭按钮", image)
    if performance is not None:
        performance.record(
            "detail.recognition",
            time.perf_counter() - recognition_started,
        )
    if not _recognition_hit(matched):
        return False
    close_x, close_y = _box_center(matched.box)
    click_started = time.perf_counter()
    job = controller.post_click(close_x, close_y).wait()
    if performance is not None:
        performance.record("detail.click", time.perf_counter() - click_started)
    if not job.succeeded:
        return False
    changed = _wait_for_visual_change(
        controller,
        image,
        performance,
        "detail.close_wait",
    )
    if performance is not None:
        performance.event("detail_closed", visual_change=changed)
    return True


def _box_center(box: object) -> tuple[int, int]:
    """兼容 tuple 和 MaaFramework Rect，返回识别框中心作为拖牌起点。"""
    if isinstance(box, (list, tuple)):
        x, y, width, height = box
    else:
        x, y = getattr(box, "x"), getattr(box, "y")
        width, height = getattr(box, "w"), getattr(box, "h")
    return int(x + width // 2), int(y + height // 2)


def _recognition_hit(detail: object | None) -> bool:
    """Maa 的 run_recognition 未命中时也会返回 RecognitionDetail。

    因此不能只判断对象或 box 是否存在，必须读取 hit；否则“放置中”等
    未命中画面会被误判成“结束回合”已经恢复。
    """
    return bool(
        detail is not None
        and getattr(detail, "hit", False)
        and getattr(detail, "box", None) is not None
    )


def _choose_highest(
    hand: BattleHand,
    excluded_slots: set[int] | None = None,
) -> DetectedCard | None:
    """选择当前能量能够支付的最高费用牌。"""
    excluded = excluded_slots or set()
    decision = choose_card(
        energy=hand.energy or 0,
        cards=(
            CardCandidate(c.slot, c.cost, c.confidence)
            for c in hand.cards
            if c.slot not in excluded
        ),
        minimum_confidence=MINIMUM_CONFIDENCE,
    )
    if decision.card is None:
        return None
    return next(card for card in hand.cards if card.slot == decision.card.slot)


def _can_end_on_stable_no_highlighted_badges(
    state: SessionState,
    hand: BattleHand,
) -> bool:
    """Only use the no-highlight shortcut before the high-value final turns."""
    return (
        1 <= state.current_turn < STRICT_NO_BADGE_GATE_TURN
        and hand.energy is not None
        and hand.energy > 0
        and hand.no_badge_frames >= NO_BADGE_END_TURN_CONFIRMATIONS
    )


def _scan_with_retry(
    context: Context,
    controller: object,
    performance: PerformanceTrace | None = None,
    tracker: HandFrameTracker | None = None,
) -> BattleHand:
    """只对不确定结果复查，明确无可支付牌时不重复整套 OCR。"""
    last = BattleHand(None, (), "not_scanned")
    waiter = _adaptive_wait()
    no_badge_streak = 0
    for attempt in range(INACTIVE_SCAN_RETRIES):
        attempt_started = time.perf_counter()
        image = _capture_frame(controller, performance, "scan")
        # 满场地时，拖牌落点可能打开场上卡牌或场地详情。必须先关闭弹层，
        # 再重新识别战斗画面，禁止在弹层上继续拖动。
        if _close_detail_overlay(context, controller, image, performance):
            continue
        last = scan_battle_hand(
            context,
            image,
            None if performance is None else performance.record,
            tracker,
        )
        if last.reason == "no_blue_badges":
            no_badge_streak += 1
            last = BattleHand(
                last.energy,
                last.cards,
                last.reason,
                no_badge_streak,
                last.unresolved_badges,
            )
        else:
            no_badge_streak = 0
        DIAGNOSTICS.update_hand(last)
        if performance is not None:
            performance.event(
                "scan",
                attempt=attempt + 1,
                duration_ms=round(
                    (time.perf_counter() - attempt_started) * 1000.0,
                    2,
                ),
                energy=last.energy,
                cards=len(last.cards),
                reason=last.reason,
                no_badge_frames=last.no_badge_frames,
                unresolved_badges=last.unresolved_badges,
            )
        if last.energy is not None and any(
            card.cost <= last.energy for card in last.cards
        ):
            return last
        # 零能量可以立即结束；只要仍有能量，就必须再看一帧。细字体 1 费牌在
        # 单帧 OCR 中最容易漏掉，不能因为同时识别到了高费牌就提前判定无牌可出。
        if last.energy == 0:
            return last
        if last.reason == "inactive_turn":
            # “放置中”、新回合按钮刚恢复以及最终回合动画都会令紫色按钮面积
            # 短暂低于活动阈值。首次进入出牌阶段也必须有限复查，不能直接把
            # 过渡帧当成对手回合并跳到结束回合。
            if attempt + 1 < INACTIVE_SCAN_RETRIES:
                waiter.wait_if_static(
                    image,
                    trace=performance,
                    phase="scan.retry",
                )
                continue
            return last
        # 征服流程会在结束回合按钮刚由灰转紫时进入这里。此时能量可能已经出现，
        # 但手牌费用徽章仍要约一秒才会亮；必须连续复查，不能把首帧 no_blue_badges
        # 当成“无牌可出”。非我方回合则由 inactive_turn 快速安全退出。
        badge_is_settling = last.reason == "no_blue_badges"
        final_turn = False
        if badge_is_settling and last.energy is not None and last.energy > 0:
            final_turn = _recognition_hit(
                context.run_recognition("公共-最终回合标记", image)
            )
        required_scans = (
            FINAL_NO_BADGE_SCAN_RETRIES
            if final_turn
            else NO_BADGE_END_TURN_CONFIRMATIONS
            if badge_is_settling
            else SCAN_RETRIES
            if last.reason == "blue_badge_ocr_failed"
            else 2
        )
        if last.energy is not None and attempt + 1 >= required_scans:
            return last
        if attempt + 1 < INACTIVE_SCAN_RETRIES:
            waiter.wait_if_static(
                image,
                trace=performance,
                phase="scan.retry",
            )
    return last


def _click_end_turn(
    context: Context,
    controller: object,
    performance: PerformanceTrace | None = None,
) -> str:
    """等待拖牌稳定，避免在“放置中”时扫描手牌或点击结束回合。"""
    deadline = time.monotonic() + PLAY_RESOLUTION_TIMEOUT_SECONDS
    saw_placing = False
    ready_confirmations = 0
    waiter = _adaptive_wait()
    while time.monotonic() < deadline:
        image = _capture_frame(controller, performance, "resolution")
        if _close_detail_overlay(context, controller, image, performance):
            return "detail"

        placing_started = time.perf_counter()
        placing = context.run_recognition("公共-放置中状态", image)
        if performance is not None:
            performance.record(
                "resolution.placing_recognition",
                time.perf_counter() - placing_started,
            )
        if _recognition_hit(placing):
            saw_placing = True
            ready_confirmations = 0
            waiter.wait_if_static(
                image,
                trace=performance,
                phase="resolution.poll",
            )
            continue

        end_turn_started = time.perf_counter()
        end_turn = context.run_recognition("公共-结束回合", image)
        if performance is not None:
            performance.record(
                "resolution.end_turn_recognition",
                time.perf_counter() - end_turn_started,
            )
        if _recognition_hit(end_turn):
            ready_confirmations += 1
            if saw_placing or ready_confirmations >= PLAY_READY_CONFIRMATIONS:
                print(
                    f"[MarvelPlayTurn] play_resolution=ready "
                    f"saw_placing={saw_placing}",
                    flush=True,
                )
                return "placed" if saw_placing else "ready"
        else:
            ready_confirmations = 0
        waiter.wait_if_static(
            image,
            trace=performance,
            phase="resolution.poll",
        )
    # “放置中”只会在游戏已经接受拖牌后出现。部分卡牌/场地连锁动画会持续
    # 超过等待上限；此时若退回 timeout，调用方会把已接受的牌当作失败并再次
    # 拖动，造成同牌跨场地重复重试。即使按钮尚未恢复，也应保留这条强成功证据。
    if saw_placing:
        print(
            "[MarvelPlayTurn] play_resolution=placed "
            "saw_placing=True settled=False",
            flush=True,
        )
        return "placed"
    print("[MarvelPlayTurn] play_resolution=timeout", flush=True)
    return "timeout"


def _play_succeeded(
    before: BattleHand,
    after: BattleHand,
    played_cost: int,
) -> bool:
    """通过画面变化确认出牌，而不是把“执行过 swipe”当作成功。"""
    if (
        before.energy is not None
        and after.energy is not None
        and after.energy < before.energy
    ):
        return True
    if played_cost != 0:
        return False
    # 0 费牌不会改变能量。旧逻辑只比较“亮牌候选数量”，当打出的 0 费牌
    # 被另一张先前漏检的亮牌接替时，数量不变，会误用旧坐标再次拖牌。
    # 候选减少，或原位置附近的同费用牌已经消失，都可证明画面已经更新。
    if len(after.cards) < len(before.cards):
        return True
    before_zeroes = tuple(card for card in before.cards if card.cost == 0)
    after_zeroes = tuple(card for card in after.cards if card.cost == 0)
    return any(
        not any(
            abs(saved.box[0] - played.box[0]) <= 55
            for saved in after_zeroes
        )
        for played in before_zeroes
    )


def _confirm_play_from_fresh_frames(
    context: Context,
    controller: object,
    before: BattleHand,
    played_cost: int,
    resolution: str,
    performance: PerformanceTrace | None = None,
    tracker: HandFrameTracker | None = None,
) -> bool:
    """拖牌后等待画面真正更新，禁止用拖牌前的旧帧继续操作。

    征服实测中结束回合按钮可能先恢复，能量和手牌稍后才刷新。原逻辑只扫
    一帧，会把已成功打出的牌当失败并继续用旧手牌坐标拖动场上区域。
    """
    if resolution == "placed":
        verify_retries = 1
    elif resolution == "ready":
        verify_retries = READY_POST_PLAY_VERIFY_RETRIES
    else:
        verify_retries = POST_PLAY_VERIFY_RETRIES
    waiter = _adaptive_wait()
    for attempt in range(verify_retries):
        image = _capture_frame(controller, performance, "verify")
        if _close_detail_overlay(context, controller, image, performance):
            return False
        # “放置中”只会在游戏接受拖牌后出现，是最可靠的成功信号。此处只取
        # 一张新截图刷新画面，不再为已经确定成功的动作执行无意义费用 OCR。
        if resolution == "placed":
            if tracker is not None:
                tracker.observe_frame(image)
            return True
        after = scan_battle_hand(
            context,
            image,
            None if performance is None else performance.record,
            tracker,
        )
        if _play_succeeded(before, after, played_cost):
            return True
        if after.reason == "inactive_turn":
            return False
        if attempt + 1 < verify_retries:
            waiter.wait_if_static(
                image,
                trace=performance,
                phase="verify.retry",
            )
    return False


@AgentServer.custom_action("MarvelPlayTurn")
class PlayTurn(CustomAction):
    """执行一个我方出牌阶段；结束回合按钮由后续 Pipeline 节点点击。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        state = STORE.require_state()
        performance = PerformanceTrace(
            "play_turn",
            run_id=state.run_id,
            turn=state.current_turn,
            strategy=state.config.play_strategy.value,
        )
        status = "exception"
        try:
            result = self._run(context, argv, state, performance)
            status = "succeeded" if result else "failed"
            return result
        finally:
            performance.finish(status=status)

    def _run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
        state: SessionState,
        performance: PerformanceTrace,
    ) -> bool:
        state.deny_end_turn()
        # 达到局数/时间限制时不再操作画面，让 Pipeline 进入停止流程。
        if state.should_stop(time.monotonic()):
            return True
        controller = context.tasker.controller

        successful_plays = 0
        no_more_playable_cards = False
        tracker = HandFrameTracker()
        # 失败计数只在本次 CustomAction（即当前回合）内有效。下一回合重新清空，
        # 因为场地容量和可放置条件都可能变化。
        lane_failures: dict[tuple[int, int], int] = {}

        while successful_plays < MAX_SUCCESSFUL_PLAYS:
            if state.should_stop(time.monotonic()):
                break
            # 每次重新进入这里都先比较徽章位置和卡牌轮廓；布局未变时只读取
            # 能量并复用费用，出牌引发重新排列后才回退完整费用 OCR。
            hand = _scan_with_retry(context, controller, performance, tracker)
            if hand.energy == 0:
                # 明确零能量是最高置信度快速路径，无需扫描场地或继续等待。
                state.allow_end_turn("energy_zero")
                no_more_playable_cards = True
                break
            if _can_end_on_stable_no_highlighted_badges(state, hand):
                state.allow_end_turn("stable_no_highlighted_badges")
                no_more_playable_cards = True
                break
            rejected_slots: set[int] = set()
            explicit_failed_slots: set[int] = set()
            card_lane_reasons: dict[int, dict[tuple[int, int], str]] = {}
            played = False

            lane_scan_started = time.perf_counter()
            lane_states = scan_lane_states(tracker.latest_image)
            performance.record(
                "lane_scan.total",
                time.perf_counter() - lane_scan_started,
            )
            full_lane_keys = {
                (lane.center_x, 730)
                for lane in lane_states
                if lane.is_full
            }
            performance.event(
                "lane_state",
                lanes=[
                    {
                        "x": lane.center_x,
                        "occupied": lane.occupied_count,
                        "full": lane.is_full,
                        "scores": lane.slot_scores,
                    }
                    for lane in lane_states
                ],
            )

            while True:
                card = _choose_highest(hand, rejected_slots)
                print(
                    f"[MarvelPlayTurn] energy={hand.energy} "
                    f"cards={[(item.cost, item.box[0]) for item in hand.cards]} "
                    f"rejected={sorted(rejected_slots)} "
                    f"selected={None if card is None else card.cost} "
                    f"reason={hand.reason}",
                    flush=True,
                )
                # 零能量、没有可信手牌或所有候选牌均尝试失败。
                if card is None:
                    if hand.reason == "inactive_turn":
                        # 游戏闪退/离开前台时，手牌扫描同样会表现为 inactive_turn。
                        # 此时不能再进入只面向战斗画面的 SNAP 门禁；交给统一状态
                        # 路由识别桌面并重启，或处理正常的回合切换/结算。
                        print(
                            "[MarvelPlayTurn] inactive_turn route=公共-出牌后状态",
                            flush=True,
                        )
                        DIAGNOSTICS.capture(
                            controller,
                            state,
                            source="play_turn",
                            reason="inactive_turn_after_retries",
                            node=getattr(argv, "node_name", None),
                            throttle_seconds=20.0,
                        )
                        return context.override_next(
                            getattr(argv, "node_name", "公共-执行出牌"),
                            ["公共-出牌后状态"],
                        )
                    if hand.energy == 0:
                        state.allow_end_turn("energy_zero")
                    elif _can_end_on_stable_no_highlighted_badges(state, hand):
                        state.allow_end_turn("stable_no_highlighted_badges")
                    else:
                        payable_slots = {
                            item.slot
                            for item in hand.cards
                            if hand.energy is not None and item.cost <= hand.energy
                        }
                        if (
                            payable_slots
                            and hand.unresolved_badges == 0
                            and payable_slots.issubset(explicit_failed_slots)
                        ):
                            reason = (
                                "all_lanes_full"
                                if len(full_lane_keys) == len(LANE_TARGETS)
                                else "all_payable_cards_failed"
                            )
                            state.allow_end_turn(reason)
                    no_more_playable_cards = True
                    break

                # 按用户配置尝试三个玩家侧场地；某个场地已满时继续尝试下一个。
                lane_reasons = card_lane_reasons.setdefault(card.slot, {})
                for lane_key in full_lane_keys:
                    lane_reasons[lane_key] = "lane_full"
                for lane_key, failures in lane_failures.items():
                    if failures >= LANE_FAILURES_BEFORE_BLOCK:
                        lane_reasons[lane_key] = "lane_retry_limit"
                ordered_lanes = tuple(
                    lane
                    for lane in lane_targets_for_order(state.config.lane_order)
                    if (lane.x, lane.y) not in full_lane_keys
                    if lane_failures.get((lane.x, lane.y), 0)
                    < LANE_FAILURES_BEFORE_BLOCK
                )
                blocked_lanes = sorted(
                    lane
                    for lane, failures in lane_failures.items()
                    if failures >= LANE_FAILURES_BEFORE_BLOCK
                )
                print(
                    f"[MarvelPlayTurn] lane_order={state.config.lane_order.value} "
                    f"lanes={[(lane.x, lane.y) for lane in ordered_lanes]} "
                    f"failures={dict(sorted(lane_failures.items()))} "
                    f"blocked={blocked_lanes} full={sorted(full_lane_keys)}",
                    flush=True,
                )
                for lane in ordered_lanes:
                    lane_key = (lane.x, lane.y)
                    start_x, start_y = _box_center(card.box)
                    placement_started = time.perf_counter()
                    print(
                        f"[MarvelPlayTurn] try slot={card.slot} cost={card.cost} "
                        f"lane={lane_key} prior_failures="
                        f"{lane_failures.get(lane_key, 0)}",
                        flush=True,
                    )
                    swipe_started = time.perf_counter()
                    job = controller.post_swipe(
                        start_x,
                        start_y,
                        lane.x,
                        lane.y,
                        SWIPE_DURATION_MS,
                    ).wait()
                    swipe_seconds = time.perf_counter() - swipe_started
                    performance.record("placement.swipe", swipe_seconds)
                    if not job.succeeded:
                        # 控制器本身执行失败属于真正错误，交给 Pipeline on_error 恢复。
                        performance.event(
                            "placement",
                            slot=card.slot,
                            cost=card.cost,
                            lane=lane_key,
                            result="controller_failed",
                            swipe_ms=round(swipe_seconds * 1000.0, 2),
                        )
                        return False
                    resolution_started = time.perf_counter()
                    resolution = _wait_for_play_resolution(
                        context,
                        controller,
                        performance,
                    )
                    resolution_seconds = time.perf_counter() - resolution_started
                    performance.record(
                        "placement.resolution",
                        resolution_seconds,
                    )
                    succeeded = False
                    verify_seconds = 0.0
                    if resolution != "detail":
                        # 即使“放置中/结束回合”状态等待超时，也必须根据新鲜
                        # 能量/手牌帧确认结果；不能直接复用旧坐标尝试下一场地。
                        verify_started = time.perf_counter()
                        succeeded = _confirm_play_from_fresh_frames(
                            context,
                            controller,
                            hand,
                            card.cost,
                            resolution,
                            performance,
                            tracker,
                        )
                        verify_seconds = time.perf_counter() - verify_started
                        performance.record(
                            "placement.verify",
                            verify_seconds,
                        )
                    performance.event(
                        "placement",
                        slot=card.slot,
                        cost=card.cost,
                        lane=lane_key,
                        resolution=resolution,
                        result="placed" if succeeded else "rejected",
                        swipe_ms=round(swipe_seconds * 1000.0, 2),
                        resolution_ms=round(resolution_seconds * 1000.0, 2),
                        verify_ms=round(verify_seconds * 1000.0, 2),
                        total_ms=round(
                            (time.perf_counter() - placement_started) * 1000.0,
                            2,
                        ),
                    )
                    if succeeded:
                        # 成功证明该场地当前仍可用，清除此前由动画或误触造成的失败。
                        lane_failures.pop(lane_key, None)
                        successful_plays += 1
                        played = True
                        # 成功出牌必然可能触发手牌重排或费用变化。下一轮必须先
                        # 建立新基线，不能继续使用拖牌前的跨帧缓存。
                        tracker.invalidate()
                        print(
                            f"[MarvelPlayTurn] placed slot={card.slot} "
                            f"cost={card.cost} lane={lane_key}",
                            flush=True,
                        )
                        break

                    # 打开卡牌详情说明拖拽起点被识别成长按，不能据此判定场地不可放。
                    counts_toward_block = resolution != "detail"
                    failures = lane_failures.get(lane_key, 0)
                    if counts_toward_block:
                        lane_reasons[lane_key] = (
                            f"placement_rejected:{resolution}"
                        )
                        failures += 1
                        lane_failures[lane_key] = failures
                        if failures == LANE_FAILURES_BEFORE_BLOCK:
                            DIAGNOSTICS.capture(
                                controller,
                                state,
                                source="play_turn",
                                reason="consecutive_placement_failure",
                                node=getattr(argv, "node_name", None),
                                detail={
                                    "slot": card.slot,
                                    "cost": card.cost,
                                    "lane": lane_key,
                                    "resolution": resolution,
                                },
                                throttle_seconds=10.0,
                            )
                    print(
                        f"[MarvelPlayTurn] placement_failed slot={card.slot} "
                        f"cost={card.cost} lane={lane_key} "
                        f"resolution={resolution} failures={failures} "
                        f"blocked={failures >= LANE_FAILURES_BEFORE_BLOCK} "
                        f"counts_toward_block={counts_toward_block}",
                        flush=True,
                    )
                    # 同一张牌在一个场地失败后直接换下一个场地，不在原地连拖；
                    # 后续其他牌再次失败，累计到阈值后才屏蔽该场地。
                    if played:
                        break
                if played:
                    break
                all_lane_keys = {(lane.x, lane.y) for lane in LANE_TARGETS}
                if all_lane_keys.issubset(lane_reasons):
                    explicit_failed_slots.add(card.slot)
                    performance.event(
                        "card_rejected",
                        slot=card.slot,
                        cost=card.cost,
                        reasons={
                            f"{lane[0]},{lane[1]}": reason
                            for lane, reason in sorted(lane_reasons.items())
                        },
                    )
                # 当前最高费用牌无法放置；保留同一帧手牌，降级尝试下一张。
                rejected_slots.add(card.slot)

            if no_more_playable_cards:
                break
        if state.end_turn_allowed:
            performance.event(
                "end_turn_gate",
                allowed=True,
                reason=state.end_turn_reason,
            )
            return True

        performance.event(
            "end_turn_gate",
            allowed=False,
            reason="insufficient_evidence",
            successful_plays=successful_plays,
        )
        # 当前仍是可操作回合但证据不足时，跳过 SNAP/结束回合，交给状态路由
        # 的“继续出牌门禁”重新观察。动画、OCR 冲突和详情误触都不会导致空过。
        return context.override_next(
            getattr(argv, "node_name", "公共-执行出牌"),
            ["公共-出牌后状态"],
        )
