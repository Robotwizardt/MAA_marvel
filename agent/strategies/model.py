from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Swipe:
    start: Point
    end: Point
    duration_ms: int = 350


@dataclass(frozen=True, slots=True)
class TurnPlan:
    swipes: tuple[Swipe, ...]
    end_turn: bool = True
