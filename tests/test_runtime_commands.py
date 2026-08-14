from datetime import date
import unittest
from types import SimpleNamespace

from agent.actions.record_event import RecordEvent
from agent.recognitions.session_gate import SessionGate
from agent.runtime.commands import apply_event, parse_json_object
from agent.runtime.store import RuntimeStore
from agent.session.config import ConquestTier
from agent.session.state import StopReason


class RuntimeCommandTests(unittest.TestCase):
    def test_parse_json_object_accepts_only_objects(self) -> None:
        self.assertEqual(
            parse_json_object('{"claim_task_rewards_hours": 3}'),
            {"claim_task_rewards_hours": 3},
        )
        for raw in ("[]", "null", '"random"'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_json_object(raw)

    def test_apply_event_updates_session_state(self) -> None:
        store = RuntimeStore()
        store.configure({}, now=100.0)
        state = store.require_state()

        apply_event(state, "match_started")
        apply_event(state, "turn_started", 1)
        apply_event(state, "match_completed")
        apply_event(state, "task_rewards_checked", now=250.0)
        apply_event(state, "known_state", "conquest_lobby")
        apply_event(state, "deck_selection_succeeded")
        apply_event(state, "daily_routine_completed")

        self.assertEqual(state.current_turn, 1)
        self.assertEqual(state.completed_matches, 1)
        self.assertFalse(state.match_in_progress)
        self.assertEqual(state.last_task_rewards_check_at, 250.0)
        self.assertEqual(state.last_known_state, "conquest_lobby")
        self.assertTrue(state.deck_selection_completed)
        self.assertEqual(state.deck_selection_result, "succeeded")
        self.assertEqual(
            state.daily_routine_completed_date,
            date.today().isoformat(),
        )

    def test_deck_selection_fallback_is_not_logged_as_success(self) -> None:
        store = RuntimeStore()
        state = store.configure({"deck_name": "动物园"}, now=100.0)

        apply_event(state, "deck_selection_fallback_not_found")

        self.assertTrue(state.deck_selection_completed)
        self.assertEqual(state.deck_selection_result, "fallback_not_found")
        self.assertFalse(state.should_select_deck())

    def test_apply_event_rejects_unknown_events(self) -> None:
        store = RuntimeStore()
        store.configure({}, now=0.0)
        with self.assertRaises(ValueError):
            apply_event(store.require_state(), "purchase_ticket")

    def test_record_event_marks_recovered_progress_as_known(self) -> None:
        STORE.configure({}, now=0.0)
        state = STORE.require_state()
        state.next_recovery_action(10.0)

        result = RecordEvent().run(
            None,
            SimpleNamespace(
                custom_action_param='{"event": "match_started"}'
            ),
        )

        self.assertTrue(result)
        self.assertEqual(state.last_known_state, "match_started")
        self.assertEqual(state.retry_count, 0)
        self.assertIsNone(state.unknown_since)

    def test_match_in_progress_gate_tracks_match_lifecycle(self) -> None:
        STORE.configure({}, now=0.0)
        argv = SimpleNamespace(
            custom_recognition_param='{"command": "match_in_progress"}'
        )
        gate = SessionGate()

        self.assertIsNone(gate.analyze(None, argv).box)
        STORE.require_state().begin_match()
        self.assertEqual(gate.analyze(None, argv).box, (0, 0, 1920, 1080))
        STORE.require_state().complete_match()
        self.assertIsNone(gate.analyze(None, argv).box)

    def test_store_requires_configuration(self) -> None:
        with self.assertRaises(RuntimeError):
            RuntimeStore().require_state()

    def test_store_builds_tier_route_without_stopping_when_exhausted(self) -> None:
        store = RuntimeStore()
        store.configure({"max_tier": "gold"}, now=0.0)
        self.assertEqual(store.next_tier_candidate(), ConquestTier.GOLD)
        self.assertEqual(store.next_tier_candidate(), ConquestTier.SILVER)
        self.assertEqual(
            store.next_tier_candidate(), ConquestTier.PROVING_GROUNDS
        )
        self.assertIsNone(store.next_tier_candidate())
        self.assertIsNone(store.require_state().stop_reason)

    def test_silver_route_always_ends_at_free_tier(self) -> None:
        store = RuntimeStore()
        store.configure({"max_tier": "silver"}, now=0.0)
        self.assertEqual(store.next_tier_candidate(), ConquestTier.SILVER)
        self.assertEqual(store.next_tier_candidate(), ConquestTier.PROVING_GROUNDS)
        self.assertIsNone(store.next_tier_candidate())
        self.assertIsNone(store.require_state().stop_reason)

    def test_reconfigure_replaces_session_and_tier_route(self) -> None:
        store = RuntimeStore()
        store.configure({}, now=10.0)
        first = store.require_state()
        first.completed_matches = 7

        store.configure({}, now=20.0)
        second = store.require_state()
        self.assertIsNot(first, second)
        self.assertEqual(second.completed_matches, 0)
        self.assertEqual(second.started_at, 20.0)


if __name__ == "__main__":
    unittest.main()
