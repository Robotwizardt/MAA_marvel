from collections.abc import Iterable
from dataclasses import dataclass
import re
import time
from typing import Any, Callable

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context, JRecognitionType
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JOCR

from agent.strategies.ocr import CardCandidate, choose_card

# 坐标均以 Maa 原生 1920×1080 横屏为基准。
# 玩家可用费用位于左侧蓝橙圆球；右侧紫色晶体是 SNAP 点数。
# 只覆盖能量球中央数字。旧 ROI (465, 630, 120, 130) 包含大量蓝橙圆环，
# 征服实测会把 1 识别成 C、把已经归零的圆环误识别成 6。
ENERGY_DIGIT_ROI = (495, 665, 50, 75)
HAND_COST_LEFT = 500
HAND_COST_RIGHT = 1430
HAND_COST_ROI = (HAND_COST_LEFT, 800, HAND_COST_RIGHT - HAND_COST_LEFT, 140)
# 可出牌的费用徽章在 850~900 的窄带中呈高亮蓝色。先在这个窄带定位徽章，
# 再逐个做 OCR，避免整片手牌 OCR 把右上角橙色战力或暗色费用当成候选。
BLUE_BADGE_SCAN_ROI = (HAND_COST_LEFT, 850, HAND_COST_RIGHT - HAND_COST_LEFT, 50)
BLUE_BADGE_OCR_TOP = 840
BLUE_BADGE_OCR_WIDTH = 44
BLUE_BADGE_OCR_HEIGHT = 60
BLUE_BADGE_TIGHT_OCR_TOP = 845
BLUE_BADGE_TIGHT_OCR_WIDTH = 32
BLUE_BADGE_TIGHT_OCR_HEIGHT = 50
BLUE_BADGE_WINDOW_WIDTH = 41
BLUE_BADGE_MIN_PIXELS = 240
BLUE_BADGE_MIN_SPACING = 80
BLUE_BADGE_MAX_COUNT = 7
ACTIVE_TURN_BUTTON_ROI = (1670, 960, 180, 70)
# 颜色只作为缺少 Pipeline 上下文时的兼容兜底。实机中真正的“结束回合”
# 在暗色背景/动画帧上可能只有约 4.2k 个紫色像素，原 7k 阈值会漏判。
# “准备战斗？”同样是紫色按钮，因此业务门禁必须再用文字证据确认。
ACTIVE_TURN_PURPLE_MIN_PIXELS = 3500
# 七个宽窗口互相重叠：文本检测器在整条手牌中漏掉细数字时，
# 数字会在至少一个较小窗口中重新参与检测。
HAND_COST_WINDOWS = tuple(
    (left, 800, min(260, HAND_COST_RIGHT - left), 140)
    for left in range(HAND_COST_LEFT, HAND_COST_RIGHT, 100)
)
MINIMUM_CONFIDENCE = 0.45
HAND_DIGIT_MIN_Y = 815
HAND_DIGIT_MAX_Y = 925
# 费用在卡牌左上、战力在右上。常规 OCR 漏掉细小的 1/2 时，往往仍能
# 识别右侧战力；从战力左边截取窄区域再做单行识别，可以找回费用。
POWER_TO_COST_MIN_OFFSET = 30
POWER_TO_COST_MAX_OFFSET = 180
# 横屏重叠手牌中，上一张牌右上战力与下一张牌左上费用常只相隔约 30px；
# 近邻冲突应保留更靠右的“下一张费用”。相邻卡牌费用中心仍有约 90px 以上间距。
SAME_CARD_DIGIT_DISTANCE = 55

TimingSink = Callable[[str, float], None]


@dataclass(frozen=True, slots=True)
class ParsedDigit:
    """单个数字的解析结果，reason 用于日志和测试诊断。"""
    value: int | None
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class DetectedCard:
    """从当前截图恢复出的动态手牌位置与费用。"""
    slot: int
    cost: int
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class BlueBadge:
    """由颜色先验定位出的可出牌费用徽章。"""
    center_x: int
    color_score: int
    roi: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class BattleHand:
    """一次截图中的完整战斗手牌快照。"""
    energy: int | None
    cards: tuple[DetectedCard, ...]
    reason: str
    no_badge_frames: int = 0
    unresolved_badges: int = 0


@dataclass(slots=True)
class _TrackedHand:
    cards: tuple[DetectedCard, ...]
    badge_centers: tuple[int, ...]
    cost_signatures: tuple[np.ndarray, ...]
    outline_signatures: tuple[np.ndarray, ...]


class HandFrameTracker:
    """跨帧关联费用徽章，仅在手牌布局不变时复用费用 OCR。"""

    CENTER_TOLERANCE = 12
    PIXEL_DELTA = 20
    COST_CHANGED_RATIO = 0.02
    OUTLINE_CHANGED_RATIO = 0.08

    def __init__(self) -> None:
        self._tracked: _TrackedHand | None = None
        self.latest_image: Any | None = None

    def observe_frame(self, image: Any) -> None:
        self.latest_image = image

    def invalidate(self) -> None:
        self._tracked = None

    def try_reuse(
        self,
        image: Any,
        badges: tuple[BlueBadge, ...],
        energy: int,
        timing: TimingSink | None = None,
    ) -> tuple[DetectedCard, ...] | None:
        started = time.perf_counter()
        tracked = self._tracked
        reason = "empty"
        if tracked is not None and len(tracked.cards) == len(badges):
            centers = tuple(badge.center_x for badge in badges)
            if all(
                abs(current - saved) <= self.CENTER_TOLERANCE
                for current, saved in zip(centers, tracked.badge_centers)
            ):
                signatures = tuple(_hand_track_signatures(image, center) for center in centers)
                if all(cost is not None and outline is not None for cost, outline in signatures):
                    cost_signatures = tuple(item[0] for item in signatures)
                    outline_signatures = tuple(item[1] for item in signatures)
                    cost_stable = all(
                        _changed_ratio(current, saved, self.PIXEL_DELTA)
                        <= self.COST_CHANGED_RATIO
                        for current, saved in zip(
                            cost_signatures,
                            tracked.cost_signatures,
                        )
                    )
                    outline_stable = all(
                        _changed_ratio(current, saved, self.PIXEL_DELTA)
                        <= self.OUTLINE_CHANGED_RATIO
                        for current, saved in zip(
                            outline_signatures,
                            tracked.outline_signatures,
                        )
                    )
                    affordable = all(card.cost <= energy for card in tracked.cards)
                    if cost_stable and outline_stable and affordable:
                        cards = tuple(
                            DetectedCard(
                                slot=slot,
                                cost=saved.cost,
                                confidence=saved.confidence,
                                box=_card_drag_box(badge.center_x),
                            )
                            for slot, (badge, saved) in enumerate(
                                zip(badges, tracked.cards)
                            )
                        )
                        if timing is not None:
                            timing(
                                "hand_scan.cache_hit",
                                time.perf_counter() - started,
                            )
                        return cards
                    reason = "signature_changed"
                else:
                    reason = "invalid_signature"
            else:
                reason = "centers_changed"
        elif tracked is not None:
            reason = "count_changed"

        if timing is not None:
            timing(
                f"hand_scan.cache_miss.{reason}",
                time.perf_counter() - started,
            )
        return None

    def remember(
        self,
        image: Any,
        badges: tuple[BlueBadge, ...],
        cards: tuple[DetectedCard, ...],
    ) -> None:
        if not cards or len(cards) != len(badges):
            self.invalidate()
            return
        centers = tuple(badge.center_x for badge in badges)
        signatures = tuple(_hand_track_signatures(image, center) for center in centers)
        if any(cost is None or outline is None for cost, outline in signatures):
            self.invalidate()
            return
        self._tracked = _TrackedHand(
            cards=cards,
            badge_centers=centers,
            cost_signatures=tuple(item[0] for item in signatures),
            outline_signatures=tuple(item[1] for item in signatures),
        )


def _card_drag_box(center_x: int) -> tuple[int, int, int, int]:
    card_center_x = max(530, min(center_x + 28, 1440))
    return card_center_x - 45, 885, 90, 90


def _hand_track_signatures(
    image: Any,
    center_x: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return None, None

    cost = _fixed_patch(pixels, center_x - 18, 842, 36, 58, sample_step=1)
    # 费用徽章右下方覆盖卡牌上沿和少量牌面。即使徽章数量与位置巧合不变，
    # 抽牌、变形或重新排列也会改变这块轮廓签名并强制重新 OCR。
    outline = _fixed_patch(pixels, center_x - 20, 900, 92, 44, sample_step=2)
    return cost, outline


def _fixed_patch(
    pixels: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
    *,
    sample_step: int,
) -> np.ndarray | None:
    right = left + width
    bottom = top + height
    if left < 0 or top < 0 or right > pixels.shape[1] or bottom > pixels.shape[0]:
        return None
    return np.ascontiguousarray(
        pixels[top:bottom:sample_step, left:right:sample_step, :3],
        dtype=np.uint8,
    )


def _changed_ratio(current: np.ndarray, saved: np.ndarray, delta: int) -> float:
    if current.shape != saved.shape or current.size == 0:
        return 1.0
    difference = np.max(
        np.abs(current.astype(np.int16) - saved.astype(np.int16)),
        axis=2,
    )
    return float(np.count_nonzero(difference >= delta) / difference.size)


def is_active_turn_frame(image: Any) -> bool:
    """区分紫色可点击的结束回合按钮与灰色等待对手按钮。"""
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return False
    left, top, width, height = ACTIVE_TURN_BUTTON_ROI
    region = pixels[top : top + height, left : left + width, :3].astype(np.float32)
    if region.shape[:2] != (height, width):
        return False

    blue, green, red = region[..., 0], region[..., 1], region[..., 2]
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 0
    red_max = nonzero & (maximum == red)
    green_max = nonzero & (maximum == green)
    blue_max = nonzero & (maximum == blue)
    hue[red_max] = (60.0 * ((green[red_max] - blue[red_max]) / delta[red_max])) % 360.0
    hue[green_max] = (
        60.0 * ((blue[green_max] - red[green_max]) / delta[green_max]) + 120.0
    )
    hue[blue_max] = (
        60.0 * ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 240.0
    )
    saturation = np.zeros_like(maximum)
    positive = maximum > 0
    saturation[positive] = delta[positive] / maximum[positive]
    purple = (
        (hue >= 240.0)
        & (hue <= 330.0)
        & (saturation >= 80.0 / 255.0)
        & (maximum >= 100.0)
    )
    # numpy 的比较结果是 numpy.bool_。CustomRecognition 会把该值写入
    # detail 并交给 json.dumps；若不转换为原生 bool，实机回合门禁会抛出
    # “Object of type bool is not JSON serializable”，导致门禁识别随机失败。
    return bool(
        np.count_nonzero(purple) >= ACTIVE_TURN_PURPLE_MIN_PIXELS
    )


def _recognition_hit(detail: object | None) -> bool:
    return bool(
        detail is not None
        and getattr(detail, "hit", False)
        and getattr(detail, "box", None) is not None
    )


def is_active_turn(context: Context, image: Any) -> bool:
    """用“结束回合”文字确认可操作回合，颜色仅作无上下文兜底。

    征服轮间的“准备战斗？”与结束回合按钮共用紫色外观，不能只靠颜色。
    Maa 的 OCR 节点能区分两者，也能覆盖亮度较低的真实结束回合按钮。
    """
    run_recognition = getattr(context, "run_recognition", None)
    if callable(run_recognition):
        return _recognition_hit(
            run_recognition("公共-结束回合文字", image)
        )
    return is_active_turn_frame(image)


def _find_blue_cost_badges(image: Any) -> tuple[BlueBadge, ...]:
    """只从高亮蓝色费用圆形区域产生 OCR 候选。

    Maa 截图是 BGR。阈值等价于 HSV 中约 197°~268° 的高饱和、高亮蓝色，
    可以排除橙色战力、灰黑色不可用费用，以及手牌后方偏青的场景背景。
    """
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return ()

    left, top, width, height = BLUE_BADGE_SCAN_ROI
    bottom = min(pixels.shape[0], top + height)
    right = min(pixels.shape[1], left + width)
    if bottom <= top or right <= left:
        return ()

    region = pixels[top:bottom, left:right, :3].astype(np.float32)
    blue, green, red = region[..., 0], region[..., 1], region[..., 2]
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    delta = maximum - minimum

    hue = np.zeros_like(maximum)
    nonzero = delta > 0
    red_max = nonzero & (maximum == red)
    green_max = nonzero & (maximum == green)
    blue_max = nonzero & (maximum == blue)
    hue[red_max] = (60.0 * ((green[red_max] - blue[red_max]) / delta[red_max])) % 360.0
    hue[green_max] = (
        60.0 * ((blue[green_max] - red[green_max]) / delta[green_max]) + 120.0
    )
    hue[blue_max] = (
        60.0 * ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 240.0
    )
    saturation = np.zeros_like(maximum)
    positive_value = maximum > 0
    saturation[positive_value] = delta[positive_value] / maximum[positive_value]

    blue_mask = (
        (hue >= 197.0)
        & (hue <= 268.0)
        & (saturation >= 0.47)
        & (maximum >= 170.0)
    )
    column_counts = np.count_nonzero(blue_mask, axis=0)
    half_window = BLUE_BADGE_WINDOW_WIDTH // 2
    window_scores = np.convolve(
        column_counts,
        np.ones(BLUE_BADGE_WINDOW_WIDTH, dtype=np.int32),
        mode="same",
    )

    ranked = sorted(
        (
            (int(window_scores[index]), left + index)
            for index in range(half_window, len(window_scores) - half_window)
            if window_scores[index] >= BLUE_BADGE_MIN_PIXELS
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    selected: list[tuple[int, int]] = []
    for score, peak_x in ranked:
        local_left = max(0, peak_x - left - half_window)
        local_right = min(blue_mask.shape[1], peak_x - left + half_window + 1)
        local_x = np.nonzero(blue_mask[:, local_left:local_right])[1]
        if local_x.size == 0:
            continue
        center_x = left + local_left + int(np.median(local_x))
        if any(
            abs(center_x - saved_x) < BLUE_BADGE_MIN_SPACING
            for _, saved_x in selected
        ):
            continue
        selected.append((score, center_x))
        if len(selected) >= BLUE_BADGE_MAX_COUNT:
            break

    badges = [
        BlueBadge(
            center_x=center_x,
            color_score=score,
            roi=(
                max(HAND_COST_LEFT, center_x - BLUE_BADGE_OCR_WIDTH // 2),
                BLUE_BADGE_OCR_TOP,
                BLUE_BADGE_OCR_WIDTH,
                BLUE_BADGE_OCR_HEIGHT,
            ),
        )
        for score, center_x in selected
    ]
    badges.sort(key=lambda badge: badge.center_x)
    return tuple(badges)


def _card_from_blue_badge(
    badge: BlueBadge,
    results: Iterable[Any],
    slot: int,
) -> DetectedCard | None:
    """从单个蓝色候选区中读取费用，拖拽坐标始终锚定在该蓝色徽章所属卡牌。"""
    candidates: list[tuple[int, float, int, int]] = []
    for result in results:
        text = str(getattr(result, "text", "")).strip()
        score = float(getattr(result, "score", 0.0))
        box = getattr(result, "box", None)
        if (
            not re.fullmatch(r"\d{1,2}", text)
            or score < MINIMUM_CONFIDENCE
            or box is None
        ):
            continue
        value = int(text)
        if not 0 <= value <= 20:
            continue
        x, y, width, height = _absolute_box(box, badge.roi)
        if (
            value == 7
            and height > 0
            and width * 10 <= height * 7
            and score >= 0.55
        ):
            value = 1
        distance = abs((x + width // 2) - badge.center_x)
        candidates.append((value, score, distance, width))

    if not candidates:
        return None
    value, score, _, _ = min(candidates, key=lambda item: (item[2], -item[1]))
    return DetectedCard(
        slot=slot,
        cost=value,
        confidence=score,
        box=_card_drag_box(badge.center_x),
    )


def _tight_blue_badge(badge: BlueBadge) -> BlueBadge:
    """缩小到数字主体；用于修正徽章边框被并入数字形成的 12/92 等结果。"""
    return BlueBadge(
        center_x=badge.center_x,
        color_score=badge.color_score,
        roi=(
            badge.center_x - BLUE_BADGE_TIGHT_OCR_WIDTH // 2,
            BLUE_BADGE_TIGHT_OCR_TOP,
            BLUE_BADGE_TIGHT_OCR_WIDTH,
            BLUE_BADGE_TIGHT_OCR_HEIGHT,
        ),
    )


def parse_digit_results(results: Iterable[Any]) -> ParsedDigit:
    """解析唯一的 0~20 能量值；超出游戏业务范围时拒绝猜测。"""
    valid: list[tuple[int, float]] = []
    for result in results:
        text = str(getattr(result, "text", "")).strip()
        if re.fullmatch(r"\d{1,2}", text) and 0 <= int(text) <= 20:
            valid.append((int(text), float(getattr(result, "score", 0.0))))

    if not valid:
        return ParsedDigit(None, 0.0, "no_single_digit")
    values = {value for value, _ in valid}
    if len(values) != 1:
        return ParsedDigit(None, max(score for _, score in valid), "conflicting_digits")
    value = valid[0][0]
    return ParsedDigit(value, max(score for _, score in valid), "recognized")


def _results(detail: Any | None) -> list[Any]:
    """读取 OCR 结果；expected 未命中时保留已经正确识别的原始数字。

    MaaFramework 的 expected 默认按普通文本匹配，形如 ``^[0-9]$`` 的内容
    若未显式开启正则会造成 filtered_results 为空，但 all_results 中仍可能有
    高置信度的正确数字。数字范围和格式会继续由 Python 严格校验。
    """
    if detail is None:
        return []
    filtered = list(getattr(detail, "filtered_results", []))
    if filtered:
        return filtered
    return list(getattr(detail, "all_results", []))


def _box_tuple(box: Any) -> tuple[int, int, int, int]:
    """把 MaaFramework Rect 或普通 tuple 统一为 (x, y, w, h)。"""
    if isinstance(box, (list, tuple)):
        x, y, width, height = box
    else:
        x, y = getattr(box, "x"), getattr(box, "y")
        width, height = getattr(box, "w"), getattr(box, "h")
    return int(x), int(y), int(width), int(height)


def _absolute_box(box: Any, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """兼容 Maa 返回 ROI 内相对坐标或画面绝对坐标这两种情况。"""
    x, y, width, height = _box_tuple(box)
    roi_x, roi_y, roi_width, roi_height = roi
    if 0 <= x < roi_width and 0 <= y < roi_height:
        return x + roi_x, y + roi_y, width, height
    return x, y, width, height


def _is_blue_cost_badge(
    image: Any,
    box: tuple[int, int, int, int],
) -> bool:
    """确认数字周围存在费用角标的蓝色底，不再用亮度判断能否出牌。

    卡牌效果可能把数字字体改为红色或绿色，但费用角标底色仍保留蓝色成分。
    阈值特意放宽以兼容暗牌；可否打出只由 OCR 能量与费用比较决定。
    """
    try:
        pixels = np.asarray(image)
    except Exception:
        return True
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        # 测试代码和第三方调用有时只传占位对象；此时保持旧的纯 OCR 行为。
        # MaaFramework 实际运行传入 OpenCV BGR 图像，会继续执行颜色判断。
        return True

    x, y, width, height = box
    height_limit, width_limit = pixels.shape[:2]
    # 六七张手牌会高度重叠：一张牌右上角的橙色战力数字，可能只比下一张牌
    # 左上角的蓝色费用徽标早约 30px。横向扩张过大会“借到”相邻费用的蓝色，
    # 将战力误判成费用；这里只检查数字自身及紧邻的徽标底色。
    x1, y1 = max(0, x - 5), max(0, y - 8)
    # 只取数字左半侧。相邻下一张牌的费用徽标总在右侧；即使 OCR 给战力数字
    # 一个偏宽文本框，也不能让取样范围一路延伸到下一张牌。
    x2 = min(width_limit, x + min(width, 14) + 4)
    y2 = min(height_limit, y + height + 8)
    region = pixels[y1:y2, x1:x2, :3].astype(np.int16)
    if region.size == 0:
        return False

    blue, green, red = region[..., 0], region[..., 1], region[..., 2]
    blue_pixels = (
        (blue >= 55)
        & (blue * 10 >= green * 9)
        & (blue * 10 >= red * 11)
        & (blue - red >= 10)
    )
    return np.count_nonzero(blue_pixels) >= 8


def _detect_cards(
    located_results: Iterable[tuple[Any, tuple[int, int, int, int]]],
    image: Any | None = None,
) -> tuple[DetectedCard, ...]:
    """从费用数字的横向位置重建当前手牌，适应出牌后的重新排布。"""
    badges: list[tuple[int, int, float]] = []
    for result, roi in located_results:
        text = str(getattr(result, "text", "")).strip()
        score = float(getattr(result, "score", 0.0))
        box = getattr(result, "box", None)
        if (
            not re.fullmatch(r"\d{1,2}", text)
            or score < MINIMUM_CONFIDENCE
            or box is None
        ):
            continue
        parsed_cost = int(text)
        # 游戏的细字体“1”偶尔被 OCR 读成“7”。只有数字框非常窄时才纠正，
        # 正常宽度的真正 7 费牌不会进入这个分支。
        raw_x, raw_y, raw_width, raw_height = _absolute_box(box, roi)
        if (
            parsed_cost == 7
            and raw_height > 0
            and raw_width * 10 <= raw_height * 7
            and score >= 0.55
        ):
            parsed_cost = 1
        if not 0 <= parsed_cost <= 20:
            continue
        x, y, width, height = raw_x, raw_y, raw_width, raw_height
        # 手牌左上角蓝色数字是费用，右上角橙色数字是战力。仅凭文本和 y 坐标
        # 无法区分二者，必须结合当前截图的徽标颜色。
        if image is not None and not _is_blue_cost_badge(
            image, (x, y, width, height)
        ):
            continue
        center_x = x + width // 2
        center_y = y + height // 2
        if not HAND_DIGIT_MIN_Y <= center_y <= HAND_DIGIT_MAX_Y:
            continue
        badges.append((center_x, parsed_cost, score))

    # 相邻窗口会重复识别同一徽标。横屏卡牌互相覆盖时，上一张的右上战力会
    # 紧贴下一张的左上费用；按 x 分组并保留更靠右的下一张费用。
    badges.sort(key=lambda item: item[0])
    unique_badges: list[tuple[int, int, float]] = []
    for badge in badges:
        if unique_badges and badge[0] - unique_badges[-1][0] <= SAME_CARD_DIGIT_DISTANCE:
            unique_badges[-1] = badge
            continue
        unique_badges.append(badge)

    # 横坐标排序后，slot=0/1/2... 就代表从左到右的动态手牌顺序。
    unique_badges.sort(key=lambda item: item[0])
    cards: list[DetectedCard] = []
    for slot, (badge_x, cost, score) in enumerate(unique_badges):
        # 横屏手牌从左到右相互覆盖。费用徽标始终位于本牌暴露的左上角，
        # 只向右偏移 28px 再向下拖，避免 +58px 落入下一张牌的覆盖区域。
        card_center_x = max(530, min(badge_x + 28, 1440))
        cards.append(
            DetectedCard(
                slot=slot,
                cost=cost,
                confidence=score,
                # 横屏手牌在底部中央横向重叠；从卡牌上半段主体开始拖，
                # 避免费用角标并保证六张牌时仍落在暴露区域。
                box=(card_center_x - 45, 885, 90, 90),
            )
        )
    return tuple(cards)


def _merge_cards(
    first: Iterable[DetectedCard],
    second: Iterable[DetectedCard],
) -> tuple[DetectedCard, ...]:
    """合并快扫和小区域复核结果，同一横坐标保留置信度更高的一项。"""
    ordered = sorted(
        (*first, *second),
        key=lambda card: (-card.confidence, card.box[0]),
    )
    unique: list[DetectedCard] = []
    for card in ordered:
        if any(
            abs(card.box[0] - saved.box[0]) <= SAME_CARD_DIGIT_DISTANCE
            for saved in unique
        ):
            continue
        unique.append(card)
    unique.sort(key=lambda card: card.box[0])
    return tuple(
        DetectedCard(index, card.cost, card.confidence, card.box)
        for index, card in enumerate(unique)
    )


def _build_cost_probe_rois(
    located_results: Iterable[tuple[Any, tuple[int, int, int, int]]],
) -> tuple[tuple[int, int, int, int], ...]:
    """根据所有手牌数字生成其左侧费用角标复核区。

    这里故意使用未经过蓝色筛选的 OCR 结果，因为右侧橙色战力正是定位
    左侧费用的可靠参照。每个窄区最多覆盖同一张牌的一对角标，不会把
    相邻卡牌的大块图案交给 only_rec。
    """
    rois: list[tuple[int, int, int, int]] = []
    centers: list[int] = []
    for result, source_roi in located_results:
        text = str(getattr(result, "text", "")).strip()
        box = getattr(result, "box", None)
        if not re.fullmatch(r"\d{1,3}", text) or box is None:
            continue
        x, y, width, height = _absolute_box(box, source_roi)
        center_y = y + height // 2
        if not HAND_DIGIT_MIN_Y <= center_y <= HAND_DIGIT_MAX_Y:
            continue
        center_x = x + width // 2
        if any(abs(center_x - saved) <= 12 for saved in centers):
            continue
        centers.append(center_x)
        left = max(HAND_COST_LEFT, center_x - POWER_TO_COST_MAX_OFFSET)
        right = max(left + 1, center_x - POWER_TO_COST_MIN_OFFSET)
        rois.append((left, max(800, center_y - 45), right - left, 90))
    return tuple(rois)


def scan_battle_hand(
    context: Context,
    image: Any,
    timing: TimingSink | None = None,
    tracker: HandFrameTracker | None = None,
) -> BattleHand:
    """识别能量后，先定位亮蓝费用徽章，再逐个读取其中的数字。"""
    total_started = time.perf_counter()
    if tracker is not None:
        tracker.observe_frame(image)

    def finish(hand: BattleHand) -> BattleHand:
        if timing is not None:
            timing("hand_scan.total", time.perf_counter() - total_started)
        return hand

    active_started = time.perf_counter()
    active_turn = is_active_turn(context, image)
    if timing is not None:
        timing("hand_scan.active_turn", time.perf_counter() - active_started)
    if not active_turn:
        if tracker is not None:
            tracker.invalidate()
        return finish(BattleHand(None, (), "inactive_turn"))

    energy_started = time.perf_counter()
    energy_detail = context.run_recognition_direct(
        JRecognitionType.OCR,
        JOCR(
            roi=ENERGY_DIGIT_ROI,
            threshold=0.12,
            only_rec=True,
        ),
        image,
    )
    energy = parse_digit_results(_results(energy_detail))
    if timing is not None:
        timing("hand_scan.energy_ocr", time.perf_counter() - energy_started)
    if energy.value is None:
        return finish(BattleHand(None, (), f"energy_{energy.reason}"))
    if energy.value == 0:
        if tracker is not None:
            tracker.invalidate()
        return finish(BattleHand(0, (), "energy_zero"))

    badges_started = time.perf_counter()
    badges = _find_blue_cost_badges(image)
    if timing is not None:
        timing("hand_scan.badge_color", time.perf_counter() - badges_started)
    if not badges:
        if tracker is not None:
            tracker.invalidate()
        return finish(BattleHand(energy.value, (), "no_blue_badges"))

    if tracker is not None:
        tracked_cards = tracker.try_reuse(image, badges, energy.value, timing)
        if tracked_cards is not None:
            return finish(
                BattleHand(
                    energy.value,
                    tracked_cards,
                    "tracked_blue_badge_path",
                    unresolved_badges=0,
                )
            )

    cards: list[DetectedCard] = []
    for badge in badges:
        cost_started = time.perf_counter()
        detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(
                roi=badge.roi,
                threshold=0.08,
                order_by="Horizontal",
            ),
            image,
        )
        if timing is not None:
            timing("hand_scan.cost_ocr", time.perf_counter() - cost_started)
        detected = _card_from_blue_badge(badge, _results(detail), len(cards))
        # 蓝色徽章只会出现在当前可支付的牌上。因此“费用大于当前能量”不是
        # 真正的高费牌，而是圆环/相邻笔画与数字粘连（实机观察到 2→12）。
        # 此时缩到数字主体再识别一次；仍不可信则丢弃，绝不盲拖。
        if detected is None or detected.cost > energy.value:
            tight = _tight_blue_badge(badge)
            retry_started = time.perf_counter()
            retry_detail = context.run_recognition_direct(
                JRecognitionType.OCR,
                JOCR(
                    roi=tight.roi,
                    threshold=0.08,
                    order_by="Horizontal",
                ),
                image,
            )
            if timing is not None:
                timing(
                    "hand_scan.cost_ocr_retry",
                    time.perf_counter() - retry_started,
                )
            detected = _card_from_blue_badge(
                tight,
                _results(retry_detail),
                len(cards),
            )
        if detected is not None and detected.cost > energy.value:
            detected = None
        if detected is not None:
            cards.append(detected)

    hand = BattleHand(
        energy.value,
        tuple(cards),
        "blue_badge_path" if cards else "blue_badge_ocr_failed",
        unresolved_badges=max(0, len(badges) - len(cards)),
    )
    if tracker is not None:
        tracker.remember(image, badges, hand.cards)
    return finish(hand)


@AgentServer.custom_recognition("MarvelCardSelection")
class CardSelection(CustomRecognition):
    """供 Pipeline 调试使用：返回当前最高费用可支付手牌的 box。"""
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        hand = scan_battle_hand(context, argv.image)
        # 识别层的 DetectedCard 转换为纯决策层 CardCandidate，降低模块耦合。
        decision = choose_card(
            energy=hand.energy or 0,
            cards=(
                CardCandidate(card.slot, card.cost, card.confidence)
                for card in hand.cards
            ),
            minimum_confidence=MINIMUM_CONFIDENCE,
        )
        selected = None
        if decision.card is not None:
            selected = next(
                card for card in hand.cards if card.slot == decision.card.slot
            )
        # CustomRecognition 以 box 是否存在表示命中；detail 保留完整诊断信息。
        return CustomRecognition.AnalyzeResult(
            box=None if selected is None else selected.box,
            detail={
                "energy": hand.energy,
                "candidates": [
                    {
                        "slot": card.slot,
                        "cost": card.cost,
                        "confidence": card.confidence,
                        "box": card.box,
                    }
                    for card in hand.cards
                ],
                "selected_slot": None if selected is None else selected.slot,
                "reason": hand.reason,
            },
        )
