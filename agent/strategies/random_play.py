from typing import Protocol, TypeVar

from agent.strategies.model import Point, Swipe, TurnPlan


T = TypeVar("T")


class RandomSource(Protocol):
    def shuffle(self, values: list[Point]) -> None: ...

    def choice(self, values: tuple[T, ...]) -> T: ...

    def randint(self, start: int, stop: int) -> int: ...


HAND_SLOTS = (
    Point(90, 1050),
    Point(270, 1050),
    Point(450, 1050),
    Point(630, 1050),
)
LANE_TARGETS = (Point(120, 650), Point(360, 650), Point(600, 650))


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _jitter_hand(point: Point, rng: RandomSource) -> Point:
    return Point(
        _clamp(point.x + rng.randint(-12, 12), 55, 665),
        _clamp(point.y + rng.randint(-12, 12), 1020, 1090),
    )


def _jitter_lane(point: Point, rng: RandomSource) -> Point:
    return Point(
        _clamp(point.x + rng.randint(-12, 12), 85, 635),
        _clamp(point.y + rng.randint(-12, 12), 600, 720),
    )


def build_random_plan(rng: RandomSource, rounds: int = 2) -> TurnPlan:
    if rounds < 1:
        raise ValueError("rounds must be positive")

    swipes: list[Swipe] = []
    for _ in range(rounds):
        slots = list(HAND_SLOTS)
        rng.shuffle(slots)
        for slot in slots:
            lane = rng.choice(LANE_TARGETS)
            swipes.append(Swipe(_jitter_hand(slot, rng), _jitter_lane(lane, rng)))
    return TurnPlan(tuple(swipes))
