import unittest

from agent.session.config import (
    AfterRetreat,
    ConquestTier,
    LaneOrder,
    SessionConfig,
    SnapMode,
)


class SessionConfigTests(unittest.TestCase):
    def test_defaults_match_approved_design(self) -> None:
        config = SessionConfig.from_mapping({})
        self.assertEqual(config.lane_order, LaneOrder.LEFT_TO_RIGHT)
        self.assertEqual(config.max_tier, ConquestTier.PROVING_GROUNDS)
        self.assertEqual(config.reserve_silver_tickets, 1)
        self.assertEqual(config.reserve_gold_tickets, 1)
        self.assertEqual(config.reserve_infinite_tickets, 1)
        self.assertTrue(config.stop_on_daily_pass_limit)
        self.assertEqual(config.after_retreat, AfterRetreat.CONTINUE)
        self.assertEqual(config.snap_mode, SnapMode.ALWAYS)
        self.assertEqual(config.snap_probability, 46)
        self.assertEqual(config.max_matches, 1)
        self.assertEqual(config.max_minutes, 30)
        self.assertEqual(config.matchmaking_timeout_seconds, 600)
        self.assertTrue(config.auto_restart)

    def test_converts_interface_values(self) -> None:
        config = SessionConfig.from_mapping(
            {
                "lane_order": "right_to_left",
                "max_tier": "silver",
                "reserve_silver_tickets": 2,
                "reserve_gold_tickets": 3,
                "reserve_infinite_tickets": 4,
                "stop_on_daily_pass_limit": False,
                "retreat_after_turn": 3,
                "after_retreat": "concede",
                "snap_mode": "probability",
                "snap_probability": 75,
                "max_matches": 8,
                "max_minutes": 90,
                "matchmaking_timeout_seconds": 300,
                "auto_restart": False,
            }
        )
        self.assertEqual(config.lane_order, LaneOrder.RIGHT_TO_LEFT)
        self.assertEqual(config.max_tier, ConquestTier.SILVER)
        self.assertEqual(config.reserve_silver_tickets, 2)
        self.assertEqual(config.reserve_gold_tickets, 3)
        self.assertEqual(config.reserve_infinite_tickets, 4)
        self.assertFalse(config.stop_on_daily_pass_limit)
        self.assertEqual(config.retreat_after_turn, 3)
        self.assertEqual(config.after_retreat, AfterRetreat.CONCEDE)
        self.assertEqual(config.snap_probability, 75)
        self.assertEqual(config.max_matches, 8)
        self.assertEqual(config.max_minutes, 90)
        self.assertEqual(config.matchmaking_timeout_seconds, 300)
        self.assertFalse(config.auto_restart)

    def test_rejects_out_of_range_values(self) -> None:
        invalid_values = (
            {"retreat_after_turn": -1},
            {"retreat_after_turn": 7},
            {"snap_probability": -1},
            {"snap_probability": 101},
            {"max_matches": -1},
            {"max_minutes": -1},
            {"matchmaking_timeout_seconds": 0},
            {"reserve_silver_tickets": -1},
            {"reserve_gold_tickets": -1},
            {"reserve_infinite_tickets": -1},
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    SessionConfig.from_mapping(values)

    def test_rejects_unknown_enum_values(self) -> None:
        with self.assertRaises(ValueError):
            SessionConfig.from_mapping({"lane_order": "center_first"})
