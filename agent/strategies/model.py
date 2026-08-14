from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    """720×1280 基准画面中的一个坐标点。"""
    x: int
    y: int
