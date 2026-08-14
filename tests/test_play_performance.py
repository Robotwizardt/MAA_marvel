from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest

import numpy as np

from agent.runtime.performance import AdaptiveFrameWait, PerformanceTrace


ROOT = Path(__file__).resolve().parents[1]


class PlayPerformanceTests(unittest.TestCase):
    def test_changed_frame_skips_wait_and_resets_backoff(self) -> None:
        sleeps: list[float] = []
        waiter = AdaptiveFrameWait(
            (0, 0, 200, 200),
            initial_seconds=0.04,
            maximum_seconds=0.16,
            sample_step=4,
            sleep=sleeps.append,
        )
        first = np.zeros((200, 200, 3), dtype=np.uint8)
        changed = first.copy()
        changed[40:120, 40:120] = 255

        self.assertEqual(waiter.wait_if_static(first), 0.0)
        self.assertEqual(waiter.wait_if_static(first), 0.04)
        self.assertEqual(waiter.wait_if_static(first), 0.08)
        self.assertEqual(waiter.wait_if_static(changed), 0.0)
        self.assertEqual(waiter.wait_if_static(changed), 0.04)
        self.assertEqual(sleeps, [0.04, 0.08, 0.04])

    def test_static_frame_uses_bounded_exponential_backoff(self) -> None:
        sleeps: list[float] = []
        waiter = AdaptiveFrameWait(
            (0, 0, 200, 200),
            initial_seconds=0.04,
            maximum_seconds=0.16,
            sample_step=4,
            sleep=sleeps.append,
        )
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        self.assertEqual(waiter.wait_if_static(frame), 0.0)
        observed = [waiter.wait_if_static(frame) for _ in range(5)]

        self.assertEqual(observed, [0.04, 0.08, 0.16, 0.16, 0.16])
        self.assertEqual(sleeps, observed)

    def test_changes_outside_activity_roi_do_not_force_busy_polling(self) -> None:
        sleeps: list[float] = []
        waiter = AdaptiveFrameWait(
            (0, 100, 200, 100),
            initial_seconds=0.04,
            sleep=sleeps.append,
        )
        first = np.zeros((200, 200, 3), dtype=np.uint8)
        changed_only_above_roi = first.copy()
        changed_only_above_roi[:80] = 255

        self.assertEqual(waiter.wait_if_static(first), 0.0)
        self.assertEqual(waiter.wait_if_static(changed_only_above_roi), 0.04)
        self.assertEqual(sleeps, [0.04])

    def test_performance_trace_emits_step_summary_as_json(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            trace = PerformanceTrace("play_turn", run_id="run-1", turn=3)
            trace.record("hand_scan.energy_ocr", 0.012)
            trace.record("hand_scan.energy_ocr", 0.008)
            trace.event("placement", result="placed", total_ms=35.0)
            trace.finish(status="succeeded")

        payloads = [
            json.loads(line.removeprefix("[MarvelPlayPerf] "))
            for line in output.getvalue().splitlines()
        ]
        self.assertEqual([item["event"] for item in payloads], ["placement", "summary"])
        summary = payloads[-1]
        self.assertEqual(summary["run_id"], "run-1")
        self.assertEqual(summary["status"], "succeeded")
        self.assertEqual(
            summary["steps"]["hand_scan.energy_ocr"],
            {"count": 2, "total_ms": 20.0, "max_ms": 12.0},
        )

    def test_play_turn_contains_no_direct_fixed_sleep(self) -> None:
        source = (ROOT / "agent/actions/play_turn.py").read_text(encoding="utf-8")
        self.assertNotIn("time.sleep(", source)


if __name__ == "__main__":
    unittest.main()
