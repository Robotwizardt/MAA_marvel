import unittest

from agent.session.config import (
    AfterRetreat,
    ConquestTier,
    NoTicketBehavior,
    PlayStrategy,
    SessionConfig,
    SnapMode,
)


class SessionConfigTests(unittest.TestCase):
    def test_defaults_match_approved_design(self) -> None:
        config = SessionConfig.from_mapping({})
        self.assertEqual(config.play_strategy, PlayStrategy.RANDOM)
        self.assertEqual(config.max_tier, ConquestTier.PROVING_GROUNDS)
        self.assertEqual(config.no_ticket, NoTicketBehavior.STOP)
        self.assertEqual(config.after_retreat, AfterRetreat.CONTINUE)
        self.assertEqual(config.snap_mode, SnapMode.OFF)
        self.assertEqual(config.snap_probability, 46)
        self.assertEqual(config.max_matches, 1)
        self.assertEqual(config.max_minutes, 30)
        self.assertEqual(config.matchmaking_timeout_seconds, 600)
        self.assertTrue(config.auto_restart)

    def test_converts_interface_values(self) -> None:
        config = SessionConfig.from_mapping(
            {
                "play_strategy": "agatha",
                "max_tier": "silver",
                "no_ticket": "stop",
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
        self.assertEqual(config.play_strategy, PlayStrategy.AGATHA)
        self.assertEqual(config.max_tier, ConquestTier.SILVER)
        self.assertEqual(config.no_ticket, NoTicketBehavior.STOP)
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
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    SessionConfig.from_mapping(values)

    def test_rejects_unknown_enum_values(self) -> None:
        with self.assertRaises(ValueError):
            SessionConfig.from_mapping({"play_strategy": "smart"})
