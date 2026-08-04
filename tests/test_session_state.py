from datetime import date
import unittest

from agent.session.config import SessionConfig, SnapMode
from agent.session.state import (
    RecoveryAction,
    SessionState,
    SnapStage,
    StopReason,
)


class FixedRng:
    def __init__(self, *values: int) -> None:
        self.values = list(values)

    def randrange(self, stop: int) -> int:
        value = self.values.pop(0)
        if not 0 <= value < stop:
            raise AssertionError(f"{value} is outside randrange({stop})")
        return value


class SessionStateTests(unittest.TestCase):
    def test_daily_routine_is_pending_once_per_calendar_day(self) -> None:
        state = SessionState(
            SessionConfig(daily_routine=True),
            started_at=0.0,
        )
        first_day = date(2026, 8, 2)
        next_day = date(2026, 8, 3)

        self.assertTrue(state.daily_routine_pending(first_day))
        state.mark_daily_routine_completed(first_day)
        self.assertFalse(state.daily_routine_pending(first_day))
        self.assertTrue(state.daily_routine_pending(next_day))

    def test_normal_battle_session_never_enters_daily_routine(self) -> None:
        state = SessionState(SessionConfig(), started_at=0.0)
        self.assertFalse(state.daily_routine_pending(date(2026, 8, 2)))

    def test_end_turn_permission_is_scoped_to_current_turn(self) -> None:
        state = SessionState(SessionConfig(), started_at=0.0)
        state.begin_turn(1)
        state.allow_end_turn("energy_zero")

        self.assertTrue(state.end_turn_allowed)
        self.assertEqual(state.end_turn_reason, "energy_zero")

        state.begin_turn(2)
        self.assertFalse(state.end_turn_allowed)
        self.assertIsNone(state.end_turn_reason)

    def test_session_runs_indefinitely_without_an_explicit_stop(self) -> None:
        state = SessionState(SessionConfig(), started_at=0.0)
        state.completed_matches = 500
        self.assertFalse(state.should_stop(100_000.0))

    def test_explicit_stop_reason_still_stops_the_session(self) -> None:
        state = SessionState(SessionConfig(), started_at=0.0)
        state.request_stop(StopReason.USER_STOPPED)
        self.assertTrue(state.should_stop(100_000.0))

    def test_task_rewards_are_due_periodically_only_between_matches(self) -> None:
        state = SessionState(
            SessionConfig(claim_task_rewards_hours=2),
            started_at=100.0,
        )
        self.assertFalse(state.task_rewards_due(7_299.9))
        self.assertTrue(state.task_rewards_due(7_300.0))

        state.begin_match()
        self.assertFalse(state.task_rewards_due(8_000.0))
        state.complete_match()
        self.assertTrue(state.task_rewards_due(8_000.0))

        state.mark_task_rewards_checked(8_000.0)
        self.assertFalse(state.task_rewards_due(15_199.9))
        self.assertTrue(state.task_rewards_due(15_200.0))

    def test_task_rewards_timer_can_restart_from_root_task_start(self) -> None:
        state = SessionState(
            SessionConfig(claim_task_rewards_hours=1),
            started_at=0.0,
            last_task_rewards_check_at=10.0,
        )

        state.start_task_rewards_timer(1_000.0, 20_000.0)

        self.assertIsNone(state.last_task_rewards_check_at)
        self.assertIsNone(state.last_task_rewards_check_wall_time)
        self.assertFalse(state.task_rewards_due(4_599.9))
        self.assertTrue(state.task_rewards_due(4_600.0))

    def test_zero_reward_interval_disables_checks(self) -> None:
        state = SessionState(SessionConfig(), started_at=0.0)
        self.assertFalse(state.task_rewards_due(1_000_000.0))

    def test_match_events_maintain_in_progress_state(self) -> None:
        state = SessionState(SessionConfig(), started_at=0.0)
        self.assertFalse(state.match_in_progress)

    def test_deck_selection_runs_once_per_session(self) -> None:
        state = SessionState(SessionConfig(deck_name="动物园"), started_at=0.0)
        self.assertTrue(state.should_select_deck())
        state.mark_deck_selection_completed()
        self.assertFalse(state.should_select_deck())
        self.assertEqual(state.deck_selection_result, "succeeded")

    def test_zero_or_empty_deck_name_disables_selection(self) -> None:
        for value in ("0", "", "   "):
            with self.subTest(value=value):
                state = SessionState(SessionConfig(deck_name=value), started_at=0.0)
                self.assertFalse(state.should_select_deck())
        state.begin_match()
        self.assertTrue(state.match_in_progress)
        state.complete_match()
        self.assertFalse(state.match_in_progress)

    def test_retreat_happens_on_the_following_turn(self) -> None:
        state = SessionState(SessionConfig(retreat_after_turn=3), started_at=0.0)
        state.begin_match()
        state.begin_turn(3)
        self.assertFalse(state.should_retreat())
        state.begin_turn(4)
        self.assertTrue(state.should_retreat())

    def test_default_retreat_setting_never_retreats(self) -> None:
        state = SessionState(SessionConfig(), started_at=0.0)
        state.begin_match()
        for turn in range(1, 7):
            state.begin_turn(turn)
            self.assertFalse(state.should_retreat())

    def test_snap_off_never_snaps(self) -> None:
        state = SessionState(SessionConfig(snap_mode=SnapMode.OFF), started_at=0.0)
        state.begin_match()
        state.begin_turn(1)
        self.assertFalse(state.decide_snap(SnapStage.FIRST, FixedRng(0)))
        self.assertFalse(state.decide_snap(SnapStage.FINAL, FixedRng(0)))

    def test_snap_always_happens_once_on_first_and_final_turn(self) -> None:
        state = SessionState(
            SessionConfig(snap_mode=SnapMode.ALWAYS), started_at=0.0
        )
        state.begin_match()
        state.begin_turn(1)
        self.assertTrue(state.decide_snap(SnapStage.FIRST, FixedRng()))
        self.assertFalse(state.decide_snap(SnapStage.FIRST, FixedRng()))
        self.assertTrue(state.decide_snap(SnapStage.FINAL, FixedRng()))
        self.assertFalse(state.decide_snap(SnapStage.FINAL, FixedRng()))
        self.assertTrue(state.first_snap_committed)
        self.assertTrue(state.final_snap_committed)

    def test_probability_snap_uses_one_roll_per_stage(self) -> None:
        state = SessionState(
            SessionConfig(snap_mode=SnapMode.PROBABILITY, snap_probability=46),
            started_at=0.0,
        )
        state.begin_match()
        state.begin_turn(1)
        self.assertTrue(state.decide_snap(SnapStage.FIRST, FixedRng(45)))
        self.assertFalse(state.decide_snap(SnapStage.FIRST, FixedRng()))
        self.assertFalse(state.decide_snap(SnapStage.FINAL, FixedRng(46)))
        self.assertFalse(state.decide_snap(SnapStage.FINAL, FixedRng(0)))

        state.begin_match()
        state.begin_turn(1)
        self.assertFalse(state.decide_snap(SnapStage.FIRST, FixedRng(46)))
        self.assertTrue(state.decide_snap(SnapStage.FINAL, FixedRng(0)))

    def test_first_turn_snap_gate_cannot_click_in_later_turns(self) -> None:
        state = SessionState(
            SessionConfig(snap_mode=SnapMode.ALWAYS), started_at=0.0
        )
        state.begin_match()
        state.begin_turn(2)
        self.assertFalse(state.decide_snap(SnapStage.FIRST, FixedRng()))

    def test_new_match_resets_both_snap_stages(self) -> None:
        state = SessionState(
            SessionConfig(snap_mode=SnapMode.ALWAYS), started_at=0.0
        )
        for _ in range(2):
            state.begin_match()
            state.begin_turn(1)
            self.assertTrue(state.decide_snap(SnapStage.FIRST, FixedRng()))
            self.assertTrue(state.decide_snap(SnapStage.FINAL, FixedRng()))

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

    def test_recovery_never_sends_android_back_during_match(self) -> None:
        state = SessionState(SessionConfig(), started_at=0.0)
        state.begin_match()
        self.assertEqual(
            [state.next_recovery_action(10.0) for _ in range(3)],
            [RecoveryAction.RETRY] * 3,
        )
        self.assertEqual(state.next_recovery_action(10.0), RecoveryAction.WAIT)
        self.assertEqual(state.back_count, 0)
        self.assertEqual(state.next_recovery_action(130.0), RecoveryAction.RESTART)

    def test_recovery_restarts_after_restart_limit_without_stopping(self) -> None:
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
        self.assertEqual(
            state.next_recovery_action(now + 120.0), RecoveryAction.RESTART
        )
        self.assertEqual(state.restart_count, 4)

    def test_known_state_resets_recovery_without_resetting_matches(self) -> None:
        state = SessionState(SessionConfig(), started_at=0.0)
        state.completed_matches = 2
        state.restart_count = 2
        state.next_recovery_action(10.0)
        state.mark_known("conquest_lobby")
        self.assertEqual(state.retry_count, 0)
        self.assertEqual(state.back_count, 0)
        self.assertEqual(state.restart_count, 0)
        self.assertIsNone(state.unknown_since)
        self.assertEqual(state.completed_matches, 2)
        self.assertEqual(state.last_known_state, "conquest_lobby")

    def test_disabled_restart_keeps_waiting_after_unknown_timeout(self) -> None:
        state = SessionState(SessionConfig(auto_restart=False), started_at=0.0)
        for _ in range(3):
            state.next_recovery_action(10.0)
        for _ in range(3):
            state.next_recovery_action(10.0)
        self.assertEqual(state.next_recovery_action(130.0), RecoveryAction.WAIT)
