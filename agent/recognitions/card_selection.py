from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context, JRecognitionType
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JOCR

from agent.strategies.ocr import CardCandidate, choose_card

# 坐标均以 720×1280 竖屏为基准。
ENERGY_DIGIT_ROI = (315, 1140, 90, 100)
HAND_COST_ROI = (0, 925, 720, 125)
# 七个宽窗口互相重叠：文本检测器在整条手牌中漏掉细数字时，
# 数字会在至少一个较小窗口中重新参与检测。
HAND_COST_WINDOWS = tuple(
    (left, 925, min(180, 720 - left), 125)
    for left in range(0, 631, 90)
)
MINIMUM_CONFIDENCE = 0.45
HAND_DIGIT_MIN_Y = 940
HAND_DIGIT_MAX_Y = 1045
# 费用在卡牌左上、战力在右上。常规 OCR 漏掉细小的 1/2 时，往往仍能
# 识别右侧战力；从战力左边截取窄区域再做单行识别，可以找回费用。
POWER_TO_COST_MIN_OFFSET = 22
POWER_TO_COST_MAX_OFFSET = 120


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
class BattleHand:
    """一次截图中的完整战斗手牌快照。"""
    energy: int | None
    cards: tuple[DetectedCard, ...]
    reason: str


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
    x1, y1 = max(0, x - 12), max(0, y - 12)
    x2 = min(width_limit, x + width + 12)
    y2 = min(height_limit, y + height + 12)
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
    return np.count_nonzero(blue_pixels) >= 10


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

    # 相邻重叠窗口会识别到同一个徽标；保留同一位置置信度最高的结果。
    badges.sort(key=lambda item: (-item[2], item[0]))
    unique_badges: list[tuple[int, int, float]] = []
    for badge in badges:
        if any(abs(badge[0] - existing[0]) <= 28 for existing in unique_badges):
            continue
        unique_badges.append(badge)

    # 横坐标排序后，slot=0/1/2... 就代表从左到右的动态手牌顺序。
    unique_badges.sort(key=lambda item: item[0])
    cards: list[DetectedCard] = []
    for slot, (badge_x, cost, score) in enumerate(unique_badges):
        # 费用徽标位于卡牌左上角，向右偏移约 42px 得到卡牌主体中心。
        card_center_x = max(55, min(badge_x + 42, 665))
        cards.append(
            DetectedCard(
                slot=slot,
                cost=cost,
                confidence=score,
                # 识别框只用来计算拖牌起点。把中心下移到卡牌主体中下部，
                # 避免费用/战力角标附近的手势被游戏当成“查看详情”。
                box=(card_center_x - 35, 1040, 70, 100),
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
        if any(abs(card.box[0] - saved.box[0]) <= 28 for saved in unique):
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
        left = max(0, center_x - POWER_TO_COST_MAX_OFFSET)
        right = max(left + 1, center_x - POWER_TO_COST_MIN_OFFSET)
        rois.append((left, max(925, center_y - 35), right - left, 70))
    return tuple(rois)


def scan_battle_hand(context: Context, image: Any) -> BattleHand:
    """两阶段 OCR：能量 + 整条手牌快扫，必要时才做小区域复核。"""
    energy_detail = context.run_recognition_direct(
        JRecognitionType.OCR,
        JOCR(
            expected=["^(?:[0-9]|1[0-9]|20)$"],
            roi=ENERGY_DIGIT_ROI,
            threshold=0.12,
            only_rec=True,
            replace=[
                ["O", "0"],
                ["Q", "0"],
                ["D", "0"],
                ["I", "1"],
                ["l", "1"],
                ["|", "1"],
            ],
        ),
        image,
    )
    energy = parse_digit_results(_results(energy_detail))
    if energy.value is None:
        return BattleHand(None, (), f"energy_{energy.reason}")

    hand_detail = context.run_recognition_direct(
        JRecognitionType.OCR,
        JOCR(roi=HAND_COST_ROI, threshold=0.20, order_by="Horizontal"),
        image,
    )
    located_results = [(result, HAND_COST_ROI) for result in _results(hand_detail)]
    cards = _detect_cards(located_results, image)

    # 手牌较多时卡牌紧密重叠，费用与战力距离固定，蓝色徽标筛选足够可靠。
    # 手牌只有 1～2 张时卡面会横向展开，右侧战力离费用更远，必须继续做
    # 稀疏手牌复核，避免把蚁人等卡牌的战力当成费用和拖牌起点。
    image_pixels = np.asarray(image)
    sparse_hand = image_pixels.ndim == 3 and len(cards) <= 2
    if not sparse_hand and any(card.cost <= energy.value for card in cards):
        return BattleHand(energy.value, cards, "fast_path")

    # 即将空过时改用重叠窗口重新运行“检测 + 识别”。这比把整块卡图交给
    # only_rec 更可靠：只有 OCR 真正检测到的小数字框才会成为候选。
    window_results: list[tuple[Any, tuple[int, int, int, int]]] = []
    for roi in HAND_COST_WINDOWS:
        detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=roi, threshold=0.20, order_by="Horizontal"),
            image,
        )
        window_results.extend((result, roi) for result in _results(detail))
    merged = _merge_cards(cards, _detect_cards(window_results, image))
    if not sparse_hand and any(card.cost <= energy.value for card in merged):
        return BattleHand(energy.value, merged, "window_path")

    # 最后只探测已知数字左侧的费用角标。天梯卡组中细字体 1/2 经常漏检，
    # 但同一张卡右侧较大的战力数字仍会被识别，因此无需再次扫描整条手牌。
    probe_results: list[tuple[Any, tuple[int, int, int, int]]] = []
    for roi in _build_cost_probe_rois((*located_results, *window_results)):
        detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(
                roi=roi,
                threshold=0.08,
                only_rec=True,
                replace=[["I", "1"], ["l", "1"], ["|", "1"]],
            ),
            image,
        )
        probe_results.extend((result, roi) for result in _results(detail))
    probe_cards = _detect_cards(probe_results, image)
    if probe_cards:
        # 同一张展开卡牌的费用一定在战力左侧。窄区成功找回费用后，删除其
        # 右侧 20～130px 内的旧候选，防止决策层再次选择战力数字。
        merged = tuple(
            card
            for card in merged
            if not any(
                20 <= card.box[0] - probe.box[0] <= 130
                for probe in probe_cards
            )
        )
    merged = _merge_cards(merged, probe_cards)
    return BattleHand(
        energy.value,
        merged,
        "cost_probe_path" if merged else "no_cards",
    )


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
