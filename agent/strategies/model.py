from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    """1920×1080 横屏基准画面中的一个坐标点。"""
    x: int
    y: int
