import unittest

from agent.session.config import SessionConfig, SnapMode
from agent.session.state import RecoveryAction, SessionState, StopReason


class FixedRng:
    def __init__(self, *values: int) -> None:
        self.values = list(values)

    def randrange(self, stop: int) -> int:
        value = self.values.pop(0)
        if not 0 <= value < stop:
            raise AssertionError(f"{value} is outside randrange({stop})")
        return value


class SessionStateTests(unittest.TestCase):
    def test_stops_after_configured_match_count(self) -> None:
        state = SessionState(SessionConfig(max_matches=2), started_at=100.0)
        state.complete_match()
        self.assertFalse(state.should_stop(101.0))
        state.complete_match()
        self.assertTrue(state.should_stop(102.0))
        self.assertEqual(state.stop_reason, StopReason.MAX_MATCHES)

    def test_stops_after_configured_runtime(self) -> None:
        state = SessionState(SessionConfig(max_minutes=10), started_at=100.0)
        self.assertFalse(state.should_stop(699.9))
        self.assertTrue(state.should_stop(700.0))
        self.assertEqual(state.stop_reason, StopReason.MAX_RUNTIME)

    def test_zero_limits_do_not_stop_the_session(self) -> None:
        state = SessionState(
            SessionConfig(max_matches=0, max_minutes=0),
            started_at=0.0,
        )
        state.completed_matches = 500
        self.assertFalse(state.should_stop(100_000.0))

    def test_retreat_happens_on_the_following_turn(self) -> None:
        state = SessionState(SessionConfig(retreat_after_turn=3), started_at=0.0)
        state.begin_match()
        state.begin_turn(3)
        self.assertFalse(state.should_retreat())
        state.begin_turn(4)
        self.assertTrue(state.should_retreat())

    def test_snap_off_never_snaps(self) -> None:
        state = SessionState(SessionConfig(snap_mode=SnapMode.OFF), started_at=0.0)
        state.begin_match()
        self.assertFalse(state.decide_snap(FixedRng(0)))

    def test_snap_always_happens_at_most_once_per_match(self) -> None:
        state = SessionState(
            SessionConfig(snap_mode=SnapMode.ALWAYS), started_at=0.0
        )
        state.begin_match()
        self.assertTrue(state.decide_snap(FixedRng()))
        self.assertFalse(state.decide_snap(FixedRng()))

    def test_probability_snap_uses_one_roll_per_match(self) -> None:
        state = SessionState(
            SessionConfig(snap_mode=SnapMode.PROBABILITY, snap_probability=46),
            started_at=0.0,
        )
        state.begin_match()
        self.assertTrue(state.decide_snap(FixedRng(45)))
        self.assertFalse(state.decide_snap(FixedRng()))

        state.begin_match()
        self.assertFalse(state.decide_snap(FixedRng(46)))
        self.assertFalse(state.decide_snap(FixedRng(0)))

    def test_recovery_retries_backs_waits_and_restarts(self) -> None:
        state = SessionState(SessionConfig(), started_at=0.0)
        self.assertEqual(
            [state.next_recovery_action(10.0) for _ in range(3)],
            [RecoveryAction.RETRY] * 3,
        )
        self.assertEqual(
            [state.next_recovery_action(10.0) for _ in range(3)],
            [RecoveryAction.ANDROID_BACK] * 3,
        )
        self.assertEqual(state.next_recovery_action(129.9), RecoveryAction.WAIT)
        self.assertEqual(state.next_recovery_action(130.0), RecoveryAction.RESTART)
        self.assertEqual(state.restart_count, 1)

    def test_recovery_stops_after_restart_limit(self) -> None:
        state = SessionState(
            SessionConfig(unknown_timeout_seconds=120, max_restarts=3),
            started_at=0.0,
        )
        now = 10.0
        for expected_restart_count in range(1, 4):
            for _ in range(3):
                self.assertEqual(state.next_recovery_action(now), RecoveryAction.RETRY)
            for _ in range(3):
                self.assertEqual(
                    state.next_recovery_action(now), RecoveryAction.ANDROID_BACK
                )
            self.assertEqual(
                state.next_recovery_action(now + 120.0), RecoveryAction.RESTART
            )
            self.assertEqual(state.restart_count, expected_restart_count)
            now += 121.0

        for _ in range(3):
            state.next_recovery_action(now)
        for _ in range(3):
            state.next_recovery_action(now)
        self.assertEqual(state.next_recovery_action(now + 120.0), RecoveryAction.STOP)
        self.assertEqual(state.stop_reason, StopReason.RECOVERY_EXHAUSTED)

    def test_known_state_resets_recovery_without_resetting_matches(self) -> None:
        state = SessionState(SessionConfig(), started_at=0.0)
        state.completed_matches = 2
        state.next_recovery_action(10.0)
        state.mark_known("conquest_lobby")
        self.assertEqual(state.retry_count, 0)
        self.assertEqual(state.back_count, 0)
        self.assertIsNone(state.unknown_since)
        self.assertEqual(state.completed_matches, 2)
        self.assertEqual(state.last_known_state, "conquest_lobby")

    def test_disabled_restart_stops_after_unknown_timeout(self) -> None:
        state = SessionState(SessionConfig(auto_restart=False), started_at=0.0)
        for _ in range(3):
            state.next_recovery_action(10.0)
        for _ in range(3):
            state.next_recovery_action(10.0)
        self.assertEqual(state.next_recovery_action(130.0), RecoveryAction.STOP)
