from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from agent.recognitions.lane_state import (
    LANE_CENTERS,
    SLOT_X_OFFSETS,
    SLOT_Y_CENTERS,
    scan_lane_states,
)


ROOT = Path(__file__).resolve().parents[1]


def marked_board(counts: tuple[int, int, int]) -> np.ndarray:
    image = np.full((1080, 1920, 3), 90, dtype=np.uint8)
    for lane_index, count in enumerate(counts):
        centers = [
            (LANE_CENTERS[lane_index] + x_offset, center_y)
            for center_y in SLOT_Y_CENTERS
            for x_offset in SLOT_X_OFFSETS
        ]
        for center_x, center_y in centers[:count]:
            yy, xx = np.indices((90, 70))
            checker = ((xx // 5 + yy // 5) % 2).astype(bool)
            patch = np.where(checker[..., None], 235, 20).astype(np.uint8)
            image[
                center_y - 45 : center_y + 45,
                center_x - 35 : center_x + 35,
            ] = patch
    return image


class LaneStateTests(unittest.TestCase):
    def test_only_four_confirmed_slots_mark_lane_full(self) -> None:
        states = scan_lane_states(marked_board((4, 3, 2)))

        self.assertEqual([item.occupied_count for item in states], [4, 3, 2])
        self.assertEqual([item.is_full for item in states], [True, False, False])

    def test_invalid_frame_never_marks_lane_full(self) -> None:
        states = scan_lane_states(object())

        self.assertTrue(all(not item.is_full for item in states))
        self.assertEqual([item.occupied_count for item in states], [0, 0, 0])

    def test_reference_empty_board_has_no_false_full_lane(self) -> None:
        image = np.asarray(
            Image.open(
                ROOT / "tests/fixtures/screens/battle/normal_turn.png"
            ).convert("RGB")
        )

        states = scan_lane_states(image)

        self.assertEqual([item.occupied_count for item in states], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
