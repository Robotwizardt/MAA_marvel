from collections.abc import Iterable
import re
from typing import Any

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context, JRecognitionType
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JOCR


# “查看所有每日任务”打开后的任务页采用居中的竖版布局。已完成任务会排在
# 每日任务列表最上方；每个圆环中上、下两个数字分别是当前进度和目标进度。
DAILY_TASK_PROGRESS_ROIS = (
    ((740, 455, 60, 45), (745, 495, 55, 40)),
    ((740, 550, 60, 45), (745, 590, 55, 40)),
    ((740, 645, 60, 45), (745, 685, 55, 40)),
)
DAILY_TASK_ROW_BOXES = (
    (730, 455, 450, 80),
    (730, 550, 450, 80),
    (730, 645, 450, 80),
)
TASK_REWARD_BADGE_ROI = (735, 50, 75, 60)
TASK_REWARD_BADGE_MIN_PIXELS = 120


def _ocr_results(detail: Any | None) -> list[Any]:
    if detail is None:
        return []
    filtered = list(getattr(detail, "filtered_results", []))
    if filtered:
        return filtered
    return list(getattr(detail, "all_results", []))


def completed_task_progress(texts: Iterable[str]) -> tuple[int, int] | None:
    """解析任务圆环中的 ``当前/目标``，只接受明确相等的正数进度。

    OCR 可能把斜线保留在同一个结果中，也可能把上下两个数字拆开，甚至把
    ``5/5`` 合并成 ``55``。任何冲突或多余数字都按未完成处理，避免误点。
    """

    normalized = [
        str(text)
        .strip()
        .replace("／", "/")
        .replace("\\", "/")
        .replace(" ", "")
        for text in texts
        if str(text).strip()
    ]
    if not normalized:
        return None

    joined = "".join(normalized)
    explicit = re.fullmatch(r"\D*(\d{1,3})/(\d{1,3})\D*", joined)
    if explicit:
        current, target = (int(explicit.group(1)), int(explicit.group(2)))
        return (current, target) if current == target > 0 else None

    digit_texts = [
        text
        for text in normalized
        if re.fullmatch(r"\d{1,6}", text)
    ]
    if len(digit_texts) == 2 and all(len(text) <= 3 for text in digit_texts):
        current, target = (int(text) for text in digit_texts)
        return (current, target) if current == target > 0 else None
    if len(digit_texts) != 1:
        return None

    # 部分 OCR 会丢掉斜线并把两个相同数字连在一起，例如 5/5 -> 55。
    merged = digit_texts[0]
    if len(merged) % 2:
        return None
    half = len(merged) // 2
    current, target = int(merged[:half]), int(merged[half:])
    return (current, target) if current == target > 0 else None


def has_pending_task_reward(image: Any) -> bool:
    """识别任务页日历标签上的红色待领奖计数。"""
    try:
        pixels = np.asarray(image)
    except Exception:
        return False
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return False
    left, top, width, height = TASK_REWARD_BADGE_ROI
    region = pixels[top : top + height, left : left + width, :3]
    if region.shape[:2] != (height, width):
        return False
    # MaaFramework 图像为 BGR。红点边框和填充均满足高红、明显高于蓝绿。
    blue = region[..., 0].astype(np.int16)
    green = region[..., 1].astype(np.int16)
    red = region[..., 2].astype(np.int16)
    red_pixels = (
        (red >= 160)
        & (red - green >= 40)
        & (red - blue >= 40)
    )
    return bool(np.count_nonzero(red_pixels) >= TASK_REWARD_BADGE_MIN_PIXELS)


def _single_progress_digit(detail: Any | None) -> int | None:
    values: list[int] = []
    for result in _ocr_results(detail):
        text = str(getattr(result, "text", "")).strip()
        if re.fullmatch(r"\d{1,3}", text):
            values.append(int(text))
    unique = set(values)
    if len(unique) != 1:
        return None
    return values[0]


@AgentServer.custom_recognition("MarvelDailyTaskReward")
class DailyTaskReward(CustomRecognition):
    """在任务页中定位第一个进度已满的每日任务，供 Pipeline 安全点击。"""

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        page = context.run_recognition("公共-领奖-任务页证据", argv.image)
        if not (
            page is not None
            and getattr(page, "hit", False)
            and getattr(page, "box", None) is not None
        ):
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail={"reason": "not_daily_task_page"},
            )

        pending_badge = has_pending_task_reward(argv.image)
        if not pending_badge:
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail={"reason": "no_pending_task_badge"},
            )

        inspected: list[dict[str, object]] = []
        for index, (digit_rois, row_box) in enumerate(
            zip(DAILY_TASK_PROGRESS_ROIS, DAILY_TASK_ROW_BOXES, strict=True)
        ):
            current_detail = context.run_recognition_direct(
                JRecognitionType.OCR,
                JOCR(roi=digit_rois[0], threshold=0.08, only_rec=True),
                argv.image,
            )
            target_detail = context.run_recognition_direct(
                JRecognitionType.OCR,
                JOCR(roi=digit_rois[1], threshold=0.08, only_rec=True),
                argv.image,
            )
            current = _single_progress_digit(current_detail)
            target = _single_progress_digit(target_detail)
            progress = (
                (current, target)
                if current is not None and current == target and current > 0
                else None
            )
            inspected.append(
                {
                    "row": index,
                    "current": current,
                    "target": target,
                    "progress": progress,
                }
            )
            if progress is not None:
                return CustomRecognition.AnalyzeResult(
                    box=row_box,
                    detail={
                        "reason": "completed_daily_task",
                        "row": index,
                        "progress": progress,
                        "inspected": inspected,
                    },
                )

        # 红色计数是游戏自身给出的“仍有奖励未领取”强证据，且已完成每日任务
        # 固定排在任务列表顶部。数字字体被斜线粘连时仍点击第一行，不能把
        # OCR 模糊误报成“没有奖励”。
        return CustomRecognition.AnalyzeResult(
            box=DAILY_TASK_ROW_BOXES[0],
            detail={
                "reason": "pending_badge_first_row_fallback",
                "inspected": inspected,
            },
        )


@AgentServer.custom_recognition("MarvelDailyTaskClear")
class DailyTaskClear(CustomRecognition):
    """只有任务页存在且红色待领奖计数消失时才允许结束检查。"""

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        page = context.run_recognition("公共-领奖-任务页证据", argv.image)
        on_page = bool(
            page is not None
            and getattr(page, "hit", False)
            and getattr(page, "box", None) is not None
        )
        clear = on_page and not has_pending_task_reward(argv.image)
        return CustomRecognition.AnalyzeResult(
            box=(480, 0, 960, 420) if clear else None,
            detail={
                "reason": (
                    "task_rewards_clear"
                    if clear
                    else ("pending_task_badge" if on_page else "not_daily_task_page")
                )
            },
        )
