import unittest

from agent.runtime.commands import apply_event, parse_json_object
from agent.runtime.store import RuntimeStore
from agent.session.config import ConquestTier, PlayStrategy
from agent.session.state import StopReason


class RuntimeCommandTests(unittest.TestCase):
    def test_parse_json_object_accepts_only_objects(self) -> None:
        self.assertEqual(parse_json_object('{"max_matches": 3}'), {"max_matches": 3})
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
        apply_event(state, "known_state", "conquest_lobby")

        self.assertEqual(state.current_turn, 1)
        self.assertEqual(state.completed_matches, 1)
        self.assertEqual(state.last_known_state, "conquest_lobby")

    def test_apply_event_rejects_unknown_events(self) -> None:
        store = RuntimeStore()
        store.configure({}, now=0.0)
        with self.assertRaises(ValueError):
            apply_event(store.require_state(), "purchase_ticket")

    def test_store_requires_configuration(self) -> None:
        with self.assertRaises(RuntimeError):
            RuntimeStore().require_state()

    def test_store_builds_tier_route_with_fallback(self) -> None:
        store = RuntimeStore()
        store.configure(
            {"max_tier": "gold", "no_ticket": "fallback"}, now=0.0
        )
        self.assertEqual(store.next_tier_candidate(), ConquestTier.GOLD)
        self.assertEqual(store.next_tier_candidate(), ConquestTier.SILVER)
        self.assertEqual(
            store.next_tier_candidate(), ConquestTier.PROVING_GROUNDS
        )
        self.assertIsNone(store.next_tier_candidate())
        self.assertEqual(store.require_state().stop_reason, StopReason.NO_TICKET)

    def test_store_stop_policy_does_not_route_to_free_tier(self) -> None:
        store = RuntimeStore()
        store.configure(
            {"max_tier": "silver", "no_ticket": "stop"}, now=0.0
        )
        self.assertEqual(store.next_tier_candidate(), ConquestTier.SILVER)
        self.assertIsNone(store.next_tier_candidate())
        self.assertEqual(store.require_state().stop_reason, StopReason.NO_TICKET)

    def test_reconfigure_replaces_session_and_tier_route(self) -> None:
        store = RuntimeStore()
        store.configure({"play_strategy": "agatha"}, now=10.0)
        first = store.require_state()
        first.completed_matches = 7

        store.configure({"play_strategy": "ocr"}, now=20.0)
        second = store.require_state()
        self.assertIsNot(first, second)
        self.assertEqual(second.config.play_strategy, PlayStrategy.OCR)
        self.assertEqual(second.completed_matches, 0)
        self.assertEqual(second.started_at, 20.0)


if __name__ == "__main__":
    unittest.main()
