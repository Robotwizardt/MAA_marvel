from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Callable

import numpy as np


@dataclass(slots=True)
class _TimingBucket:
    count: int = 0
    total_seconds: float = 0.0
    maximum_seconds: float = 0.0

    def add(self, elapsed_seconds: float) -> None:
        elapsed = max(0.0, float(elapsed_seconds))
        self.count += 1
        self.total_seconds += elapsed
        self.maximum_seconds = max(self.maximum_seconds, elapsed)


class PerformanceTrace:
    """Aggregate high-frequency timings and emit compact structured log events."""

    def __init__(self, category: str, **context: Any) -> None:
        self.category = category
        self.context = context
        self.started_at = time.perf_counter()
        self._steps: dict[str, _TimingBucket] = {}
        self._finished = False

    def record(self, step: str, elapsed_seconds: float) -> None:
        self._steps.setdefault(step, _TimingBucket()).add(elapsed_seconds)

    def increment(self, step: str) -> None:
        self.record(step, 0.0)

    def event(self, event: str, **detail: Any) -> None:
        payload = {
            "category": self.category,
            "event": event,
            "elapsed_ms": _milliseconds(time.perf_counter() - self.started_at),
            **self.context,
            **detail,
        }
        print(
            "[MarvelPlayPerf] "
            + json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ),
            flush=True,
        )

    def finish(self, **detail: Any) -> None:
        if self._finished:
            return
        self._finished = True
        steps = {
            name: {
                "count": bucket.count,
                "total_ms": _milliseconds(bucket.total_seconds),
                "max_ms": _milliseconds(bucket.maximum_seconds),
            }
            for name, bucket in sorted(self._steps.items())
        }
        self.event("summary", steps=steps, **detail)


class AdaptiveFrameWait:
    """Skip waiting while a relevant screen region changes; back off when static."""

    def __init__(
        self,
        roi: tuple[int, int, int, int],
        *,
        initial_seconds: float = 0.04,
        maximum_seconds: float = 0.28,
        sample_step: int = 8,
        pixel_delta: int = 12,
        changed_ratio: float = 0.003,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.roi = roi
        self.initial_seconds = initial_seconds
        self.maximum_seconds = maximum_seconds
        self.sample_step = sample_step
        self.pixel_delta = pixel_delta
        self.changed_ratio = changed_ratio
        self._sleep = sleep
        self._previous: np.ndarray | None = None
        self._stable_observations = 0

    def prime(self, image: Any) -> None:
        signature = self._signature(image)
        if signature is not None:
            self._previous = signature
            self._stable_observations = 0

    def wait_if_static(
        self,
        image: Any,
        *,
        trace: PerformanceTrace | None = None,
        phase: str = "poll",
    ) -> float:
        signature = self._signature(image)
        changed = False
        if signature is not None:
            if self._previous is None:
                changed = True
            elif self._previous.shape != signature.shape:
                changed = True
            else:
                difference = np.max(
                    np.abs(signature.astype(np.int16) - self._previous.astype(np.int16)),
                    axis=2,
                )
                changed = (
                    np.count_nonzero(difference >= self.pixel_delta)
                    / difference.size
                    >= self.changed_ratio
                )
            self._previous = signature

        if changed:
            self._stable_observations = 0
            if trace is not None:
                trace.increment(f"{phase}.changed_no_wait")
            return 0.0

        self._stable_observations += 1
        delay = min(
            self.maximum_seconds,
            self.initial_seconds * (2 ** min(self._stable_observations - 1, 4)),
        )
        started = time.perf_counter()
        self._sleep(delay)
        elapsed = time.perf_counter() - started
        if trace is not None:
            trace.record(f"{phase}.static_wait", elapsed)
        return delay

    def _signature(self, image: Any) -> np.ndarray | None:
        pixels = np.asarray(image)
        if pixels.ndim != 3 or pixels.shape[2] < 3:
            return None
        left, top, width, height = self.roi
        right = min(pixels.shape[1], left + width)
        bottom = min(pixels.shape[0], top + height)
        if left < 0 or top < 0 or right <= left or bottom <= top:
            return None
        region = pixels[top:bottom:self.sample_step, left:right:self.sample_step, :3]
        if region.size == 0:
            return None
        return np.ascontiguousarray(region, dtype=np.uint8)


def _milliseconds(seconds: float) -> float:
    return round(max(0.0, seconds) * 1000.0, 2)
