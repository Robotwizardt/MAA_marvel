from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    """1920×1080 横屏基准画面中的一个坐标点。"""
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Swipe:
    """一次拖动动作的数据描述；当前新版出牌主要直接调用 controller。"""
    start: Point
    end: Point
    duration_ms: int = 350


@dataclass(frozen=True, slots=True)
class TurnPlan:
    """历史策略计划模型，当前仅由阿加莎策略测试使用。"""
    swipes: tuple[Swipe, ...]
    end_turn: bool = True
