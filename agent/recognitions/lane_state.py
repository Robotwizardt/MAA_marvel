from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# 玩家侧三个场地均使用固定 2x2 卡位。中心与落牌目标保持一致；卡位中心
# 来自 1920x1080 横屏实机截图，刻意避开场地卡牌与底部手牌区域。
LANE_CENTERS = (710, 960, 1210)
SLOT_X_OFFSETS = (-55, 45)
SLOT_Y_CENTERS = (645, 755)
SLOT_HALF_WIDTH = 35
SLOT_HALF_HEIGHT = 45
SLOT_EDGE_DELTA = 28.0
SLOT_EDGE_DENSITY_MIN = 0.08
SLOT_CONTRAST_MIN = 32.0


@dataclass(frozen=True, slots=True)
class LaneState:
    """一次截图中单个玩家侧场地的保守占用判断。"""

    center_x: int
    occupied_slots: tuple[bool, bool, bool, bool]
    slot_scores: tuple[float, float, float, float]

    @property
    def occupied_count(self) -> int:
        return sum(self.occupied_slots)

    @property
    def is_full(self) -> bool:
        # 只允许四个卡位全部具有强卡牌纹理证据时跳过该场地。漏检只会回到
        # 原有拖牌验证，误把可用场地判满则可能漏出牌，因此这里保持保守。
        return all(self.occupied_slots)


def scan_lane_states(image: Any) -> tuple[LaneState, LaneState, LaneState]:
    """通过固定卡位的边缘密度和局部对比度判断三个场地是否已满。"""

    try:
        pixels = np.asarray(image)
    except Exception:
        pixels = np.empty((0, 0, 0), dtype=np.uint8)

    valid = pixels.ndim == 3 and pixels.shape[2] >= 3
    states: list[LaneState] = []
    for center_x in LANE_CENTERS:
        occupied: list[bool] = []
        scores: list[float] = []
        for center_y in SLOT_Y_CENTERS:
            for slot_x in (center_x + offset for offset in SLOT_X_OFFSETS):
                present, score = (
                    _slot_has_card(pixels, slot_x, center_y)
                    if valid
                    else (False, 0.0)
                )
                occupied.append(present)
                scores.append(score)
        states.append(
            LaneState(
                center_x=center_x,
                occupied_slots=tuple(occupied),
                slot_scores=tuple(scores),
            )
        )
    return tuple(states)


def _slot_has_card(
    pixels: np.ndarray,
    center_x: int,
    center_y: int,
) -> tuple[bool, float]:
    left = center_x - SLOT_HALF_WIDTH
    right = center_x + SLOT_HALF_WIDTH
    top = center_y - SLOT_HALF_HEIGHT
    bottom = center_y + SLOT_HALF_HEIGHT
    if (
        left < 0
        or top < 0
        or right > pixels.shape[1]
        or bottom > pixels.shape[0]
    ):
        return False, 0.0

    region = pixels[top:bottom, left:right, :3].astype(np.float32)
    gray = np.mean(region, axis=2)
    contrast = float(np.std(gray))
    horizontal = np.abs(np.diff(gray, axis=1)) >= SLOT_EDGE_DELTA
    vertical = np.abs(np.diff(gray, axis=0)) >= SLOT_EDGE_DELTA
    edge_density = (float(np.mean(horizontal)) + float(np.mean(vertical))) / 2.0
    present = (
        contrast >= SLOT_CONTRAST_MIN
        and edge_density >= SLOT_EDGE_DENSITY_MIN
    )
    score = min(
        contrast / SLOT_CONTRAST_MIN,
        edge_density / SLOT_EDGE_DENSITY_MIN,
    )
    return present, round(score, 3)
