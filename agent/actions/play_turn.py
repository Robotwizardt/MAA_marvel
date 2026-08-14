import random
import time

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.recognitions.card_selection import (
    BattleHand,
    DetectedCard,
    MINIMUM_CONFIDENCE,
    scan_battle_hand,
)
from agent.runtime.store import STORE
from agent.session.config import LaneOrder, SnapMode
from agent.strategies.ocr import CardCandidate, choose_card
from agent.strategies.model import Point


LANE_TARGETS = (Point(145, 840), Point(360, 840), Point(575, 840))
# 单回合安全上限。正常一回合不可能成功打出 12 张牌；此限制用于防止异常循环。
MAX_SUCCESSFUL_PLAYS = 6
# 500ms 会被游戏识别成长按并打开卡牌详情；快速拖动才会进入放牌状态。
SWIPE_DURATION_MS = 180
# 第一次手势偶尔会被游戏当成点击卡牌。关闭详情后用更快手势在同一场地
# 重试一次；只有连续两次打开详情才把该场地判为本回合不可用。
DETAIL_SWIPE_RETRIES = 2
DETAIL_RETRY_SWIPE_DURATION_MS = 120
DETAIL_RETRY_START_Y_OFFSET = 18
# 拖牌后等待动画结束，再截图判断是否真的出牌成功。
POST_SWIPE_DELAY_SECONDS = 0.7
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
    """检测卡牌/场地详情弹层底部的红色六边形白 X 按钮。"""
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return False

    # MaaFramework 截图为 BGR；正常战斗画面此处是蓝色能量球，
    # 详情弹层则固定为红色关闭按钮，因此颜色组合区分度很高。
    region = pixels[1135:1255, 285:435, :3].astype(np.int16)
    blue, green, red = region[..., 0], region[..., 1], region[..., 2]
    red_pixels = np.count_nonzero(
        (red >= 120) & (red * 2 >= green * 3) & (red * 10 >= blue * 13)
    )
    channel_max = np.maximum(np.maximum(red, green), blue)
    channel_min = np.minimum(np.minimum(red, green), blue)
    white_pixels = np.count_nonzero(
        (red >= 190)
        & (green >= 190)
        & (blue >= 190)
        & (channel_max - channel_min <= 40)
    )
    return red_pixels >= 800 and white_pixels >= 40


def _close_detail_overlay(context: Context, controller: object, image: object) -> bool:
    """颜色预筛选后再用关闭按钮模板确认，避免把能量球误认为 X。"""
    if not _is_detail_overlay(image):
        return False
    matched = context.run_recognition("公共-详情关闭按钮", image)
    if matched is None or matched.box is None:
        return False
    close_x, close_y = _box_center(matched.box)
    job = controller.post_click(close_x, close_y).wait()
    if not job.succeeded:
        return False
    time.sleep(DETAIL_CLOSE_DELAY_SECONDS)
    return True


def _box_center(box: object) -> tuple[int, int]:
    """兼容 tuple 和 MaaFramework Rect，返回识别框中心作为拖牌起点。"""
    if isinstance(box, (list, tuple)):
        x, y, width, height = box
    else:
        x, y = getattr(box, "x"), getattr(box, "y")
        width, height = getattr(box, "w"), getattr(box, "h")
    return int(x + width // 2), int(y + height // 2)


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


def _scan_with_retry(context: Context, controller: object) -> BattleHand:
    """只对不确定结果复查，明确无可支付牌时不重复整套 OCR。"""
    last = BattleHand(None, (), "not_scanned")
    for attempt in range(SCAN_RETRIES):
        image = controller.post_screencap().get(wait=True)
        # 满场地时，拖牌落点可能打开场上卡牌或场地详情。必须先关闭弹层，
        # 再重新识别战斗画面，禁止在弹层上继续拖动。
        if _close_detail_overlay(context, controller, image):
            continue
        last = scan_battle_hand(context, image)
        if last.energy is not None and any(
            card.cost <= last.energy for card in last.cards
        ):
            return last
        # 已识别到能量和手牌，且所有费用都更高，属于确定的“无可支付牌”。
        # 不再进行第二次完整窗口扫描。
        if last.energy is not None and last.cards:
            return last
        if attempt + 1 < SCAN_RETRIES:
            time.sleep(SCAN_RETRY_DELAY_SECONDS)
    return last


def _click_end_turn(
    context: Context,
    controller: object,
) -> bool:
    """直接识别并点击结束回合，省去 Pipeline 的整屏批量 OCR。"""
    image = controller.post_screencap().get(wait=True)
    matched = context.run_recognition("公共-结束回合", image)
    if matched is None or matched.box is None:
        return False
    end_x, end_y = _box_center(matched.box)
    job = controller.post_click(end_x, end_y).wait()
    if not job.succeeded:
        return False
    time.sleep(DIRECT_END_TURN_DELAY_SECONDS)
    return True


def _can_end_turn_directly(state: object) -> bool:
    """不会跳过本回合 SNAP 决策时，才允许在 CustomAction 内结束回合。"""
    return (
        state.config.snap_mode is SnapMode.OFF
        or state.snap_decision_made
        or state.snapped_this_match
    )


def _play_succeeded(before: BattleHand, after: BattleHand) -> bool:
    """通过画面变化确认出牌，而不是把“执行过 swipe”当作成功。"""
    if (
        before.energy is not None
        and after.energy is not None
        and after.energy < before.energy
    ):
        return True
    # 0 费牌不消耗能量，使用手牌重排作为第二个成功信号。
    before_signature = tuple((card.cost, card.box[0]) for card in before.cards)
    after_signature = tuple((card.cost, card.box[0]) for card in after.cards)
    return before_signature != after_signature


@AgentServer.custom_action("MarvelPlayTurn")
class PlayTurn(CustomAction):
    """执行一个我方出牌阶段；结束回合按钮由后续 Pipeline 节点点击。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        state = STORE.require_state()
        # 达到局数/时间限制时不再操作画面，让 Pipeline 进入停止流程。
        if state.should_stop(time.monotonic()):
            return True
        controller = context.tasker.controller

        successful_plays = 0
        no_more_playable_cards = False

        while successful_plays < MAX_SUCCESSFUL_PLAYS:
            if state.should_stop(time.monotonic()):
                break
            # 每打出一张牌，剩余手牌会重新排布，所以每轮必须重新截图识别。
            hand = _scan_with_retry(context, controller)
            rejected_slots: set[int] = set()
            played = False

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
                    no_more_playable_cards = True
                    break

                # 按用户配置尝试三个玩家侧场地；某个场地已满时继续尝试下一个。
                ordered_lanes = lane_targets_for_order(state.config.lane_order)
                for lane in ordered_lanes:
                    lane_index = LANE_TARGETS.index(lane)
                    if lane_index in state.blocked_lanes:
                        continue
                    for drag_attempt in range(DETAIL_SWIPE_RETRIES):
                        start_x, start_y = _box_center(card.box)
                        duration = SWIPE_DURATION_MS
                        if drag_attempt > 0:
                            # 第二次从卡牌更下方快速划出，降低被识别为点击的概率。
                            start_y += DETAIL_RETRY_START_Y_OFFSET
                            duration = DETAIL_RETRY_SWIPE_DURATION_MS
                        job = controller.post_swipe(
                            start_x,
                            start_y,
                            lane.x,
                            lane.y,
                            duration,
                        ).wait()
                        if not job.succeeded:
                            # 控制器本身执行失败属于真正错误，交给 Pipeline on_error 恢复。
                            return False
                        time.sleep(POST_SWIPE_DELAY_SECONDS)
                        image = controller.post_screencap().get(wait=True)
                        if _close_detail_overlay(context, controller, image):
                            if drag_attempt + 1 < DETAIL_SWIPE_RETRIES:
                                # 可能只是误点卡牌，先在原场地重试。
                                continue
                            state.blocked_lanes.add(lane_index)
                            break
                        after = _scan_with_retry(context, controller)
                        if _play_succeeded(hand, after):
                            successful_plays += 1
                            played = True
                            break
                        # 手势完成但画面不变，换场尝试。
                        break
                    if played:
                        break
                if played:
                    break
                # 当前最高费用牌无法放置；保留同一帧手牌，降级尝试下一张。
                rejected_slots.add(card.slot)

            if no_more_playable_cards:
                break
        if no_more_playable_cards and _can_end_turn_directly(state):
            _click_end_turn(context, controller)
        return True
