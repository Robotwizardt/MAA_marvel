import json
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.actions import trace_runtime
from agent.runtime.checkpoint import CHECKPOINT_VERSION, SessionCheckpointStore
from agent.runtime.commands import apply_event
from agent.runtime.store import RuntimeStore
from agent.session.config import ConquestTier
from tools.validate_schema import load_jsonc


ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def make_store(self, root: str) -> RuntimeStore:
        checkpoint = SessionCheckpointStore(Path(root) / "checkpoint.json")
        return RuntimeStore(checkpoint)

    def test_round_trip_restores_minimum_session_state(self) -> None:
        with TemporaryDirectory() as root:
            first = self.make_store(root)
            state = first.configure(
                {"max_tier": "gold", "deck_name": "测试卡组"},
                now=100.0,
                wall_time=1_000.0,
            )
            state.completed_matches = 7
            state.begin_match()
            state.begin_turn(4)
            state.first_snap_decision_made = True
            state.first_snap_committed = True
            state.mark_deck_selection_completed()
            state.mark_known("battle_turn:4")
            run_id = state.run_id
            first.persist_checkpoint(now=120.0, wall_time=1_020.0)

            second = self.make_store(root)
            restored = second.configure(
                {"max_tier": "gold", "deck_name": "测试卡组"},
                now=10.0,
                wall_time=1_030.0,
            )

            self.assertTrue(second.last_configure_restored())
            self.assertEqual(restored.run_id, run_id)
            self.assertEqual(restored.completed_matches, 7)
            self.assertEqual(restored.current_turn, 4)
            self.assertTrue(restored.match_in_progress)
            self.assertTrue(restored.first_snap_decision_made)
            self.assertTrue(restored.first_snap_committed)
            self.assertTrue(restored.deck_selection_completed)
            self.assertEqual(restored.deck_selection_result, "succeeded")
            self.assertEqual(restored.last_known_state, "battle_turn:4")

    def test_reward_epoch_survives_process_restart(self) -> None:
        with TemporaryDirectory() as root:
            first = self.make_store(root)
            state = first.configure(
                {"claim_task_rewards_hours": 2},
                now=100.0,
                wall_time=10_000.0,
            )
            state.mark_task_rewards_checked(500.0, 20_000.0)
            first.persist_checkpoint(now=500.0, wall_time=20_000.0)

            second = self.make_store(root)
            restored = second.configure(
                {"claim_task_rewards_hours": 2},
                now=10.0,
                wall_time=27_199.0,
            )

            self.assertFalse(restored.task_rewards_due(10.0))
            self.assertTrue(restored.task_rewards_due(11.0))

    def test_reward_timer_epoch_survives_process_restart(self) -> None:
        with TemporaryDirectory() as root:
            first = self.make_store(root)
            state = first.configure(
                {"claim_task_rewards_hours": 1},
                now=100.0,
                wall_time=10_000.0,
            )
            state.start_task_rewards_timer(500.0, 10_400.0)
            first.persist_checkpoint(now=500.0, wall_time=10_400.0)

            second = self.make_store(root)
            restored = second.configure(
                {"claim_task_rewards_hours": 1},
                now=20.0,
                wall_time=13_999.0,
            )

            self.assertFalse(restored.task_rewards_due(20.0))
            self.assertTrue(restored.task_rewards_due(21.0))

    def test_legacy_deck_completion_without_result_retries_selection(self) -> None:
        with TemporaryDirectory() as root:
            first = self.make_store(root)
            state = first.configure(
                {"deck_name": "动物园"},
                now=100.0,
                wall_time=1_000.0,
            )
            state.mark_deck_selection_completed()
            first.persist_checkpoint(now=101.0, wall_time=1_001.0)

            path = Path(root) / "checkpoint.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["sessions"]["conquest"]["state"].pop(
                "deck_selection_result"
            )
            path.write_text(
                json.dumps(document, ensure_ascii=False),
                encoding="utf-8",
            )

            second = self.make_store(root)
            restored = second.configure(
                {"deck_name": "动物园"},
                now=10.0,
                wall_time=1_010.0,
            )

            self.assertTrue(second.last_configure_restored())
            self.assertFalse(restored.deck_selection_completed)
            self.assertIsNone(restored.deck_selection_result)
            self.assertTrue(restored.should_select_deck())

    def test_daily_completion_date_survives_process_restart(self) -> None:
        with TemporaryDirectory() as root:
            first = self.make_store(root)
            state = first.configure(
                {"daily_routine": True},
                now=100.0,
                wall_time=10_000.0,
            )
            completed_day = date(2026, 8, 2)
            state.mark_daily_routine_completed(completed_day)
            first.persist_checkpoint(now=101.0, wall_time=10_001.0)

            second = self.make_store(root)
            restored = second.configure(
                {"daily_routine": True},
                now=10.0,
                wall_time=10_100.0,
            )

            self.assertTrue(second.last_configure_restored())
            self.assertFalse(restored.daily_routine_pending(completed_day))
            self.assertTrue(restored.daily_routine_pending(date(2026, 8, 3)))

    def test_current_tier_and_remaining_queue_are_restored(self) -> None:
        with TemporaryDirectory() as root:
            first = self.make_store(root)
            first.configure({"max_tier": "gold"}, now=0.0, wall_time=100.0)
            self.assertEqual(first.next_tier_candidate(), ConquestTier.GOLD)
            self.assertEqual(
                first.tier_candidates(),
                (ConquestTier.SILVER, ConquestTier.PROVING_GROUNDS),
            )

            second = self.make_store(root)
            second.configure({"max_tier": "gold"}, now=1.0, wall_time=101.0)

            self.assertEqual(second.current_tier(), ConquestTier.GOLD)
            self.assertEqual(
                second.tier_candidates(),
                (ConquestTier.SILVER, ConquestTier.PROVING_GROUNDS),
            )
            self.assertEqual(second.next_tier_candidate(), ConquestTier.GOLD)
            self.assertEqual(second.next_tier_candidate(), ConquestTier.SILVER)

    def test_changed_configuration_starts_a_new_session(self) -> None:
        with TemporaryDirectory() as root:
            first = self.make_store(root)
            old = first.configure(
                {"play_strategy": "ocr"},
                now=0.0,
                wall_time=100.0,
            )
            old.completed_matches = 9
            first.persist_checkpoint(now=1.0, wall_time=101.0)

            second = self.make_store(root)
            new = second.configure(
                {"play_strategy": "agatha"},
                now=2.0,
                wall_time=102.0,
            )

            self.assertFalse(second.last_configure_restored())
            self.assertNotEqual(new.run_id, old.run_id)
            self.assertEqual(new.completed_matches, 0)

    def test_corrupt_checkpoint_does_not_break_configuration(self) -> None:
        with TemporaryDirectory() as root:
            path = Path(root) / "checkpoint.json"
            path.write_text("{broken", encoding="utf-8")
            store = RuntimeStore(SessionCheckpointStore(path))

            state = store.configure({}, now=10.0, wall_time=100.0)

            self.assertFalse(store.last_configure_restored())
            self.assertEqual(state.completed_matches, 0)
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["version"], CHECKPOINT_VERSION)

    def test_atomic_write_leaves_no_temporary_file(self) -> None:
        with TemporaryDirectory() as root:
            checkpoint = SessionCheckpointStore(Path(root) / "checkpoint.json")
            store = RuntimeStore(checkpoint)
            store.configure({}, now=0.0, wall_time=100.0)
            store.persist_checkpoint(now=1.0, wall_time=101.0)

            self.assertTrue(checkpoint.path.is_file())
            self.assertFalse(checkpoint.temporary_path.exists())

    def test_root_task_memory_clear_preserves_checkpoint(self) -> None:
        with TemporaryDirectory() as root:
            checkpoint = SessionCheckpointStore(Path(root) / "checkpoint.json")
            store = RuntimeStore(checkpoint)
            store.configure({}, now=0.0, wall_time=100.0)

            store.clear_state()

            self.assertTrue(checkpoint.path.is_file())

    def test_training_session_cannot_overwrite_conquest_checkpoint(self) -> None:
        with TemporaryDirectory() as root:
            checkpoint = SessionCheckpointStore(Path(root) / "checkpoint.json")
            battle = RuntimeStore(checkpoint)
            battle_state = battle.configure({}, now=0.0, wall_time=100.0)
            battle_state.completed_matches = 6
            battle.persist_checkpoint(now=1.0, wall_time=101.0)

            training = RuntimeStore(checkpoint)
            training_state = training.configure(
                {"auto_restart": False},
                now=2.0,
                wall_time=102.0,
                checkpoint_enabled=False,
            )
            training_state.completed_matches = 99
            self.assertFalse(training.persist_checkpoint())

            resumed = RuntimeStore(checkpoint)
            restored = resumed.configure({}, now=3.0, wall_time=103.0)
            self.assertTrue(resumed.last_configure_restored())
            self.assertEqual(restored.completed_matches, 6)

    def test_fresh_daily_session_skips_restore_but_replaces_checkpoint(self) -> None:
        with TemporaryDirectory() as root:
            checkpoint = SessionCheckpointStore(Path(root) / "checkpoint.json")
            previous = RuntimeStore(checkpoint)
            stale = previous.configure(
                {"daily_routine": True},
                now=0.0,
                wall_time=100.0,
            )
            stale.begin_match()
            stale.begin_turn(5)
            stale.mark_known("turn_started:5")
            previous.persist_checkpoint(now=1.0, wall_time=101.0)

            daily = RuntimeStore(checkpoint)
            fresh = daily.configure(
                {"daily_routine": True},
                now=2.0,
                wall_time=102.0,
                checkpoint_enabled=True,
                restore_checkpoint=False,
            )

            self.assertFalse(daily.last_configure_restored())
            self.assertFalse(fresh.match_in_progress)
            self.assertEqual(fresh.current_turn, 0)
            self.assertEqual(fresh.last_known_state, "task_started")
            self.assertTrue(daily.persist_checkpoint(now=3.0, wall_time=103.0))

            resumed = RuntimeStore(checkpoint)
            restored = resumed.configure(
                {"daily_routine": True},
                now=4.0,
                wall_time=104.0,
            )
            self.assertTrue(resumed.last_configure_restored())
            self.assertFalse(restored.match_in_progress)
            self.assertEqual(restored.current_turn, 0)

    def test_pipeline_safe_stop_clears_checkpoint(self) -> None:
        with TemporaryDirectory() as root:
            checkpoint = SessionCheckpointStore(Path(root) / "checkpoint.json")
            store = RuntimeStore(checkpoint)
            store.configure({}, now=0.0, wall_time=100.0)
            context = SimpleNamespace(tasker=SimpleNamespace(controller=None))
            argv = SimpleNamespace(
                node_name="公共-安全停止",
                custom_action_param=(
                    '{"source":"pipeline","reason":"test_safe_stop",'
                    '"stop":true}'
                ),
            )

            with (
                patch.object(trace_runtime, "STORE", store),
                patch.object(trace_runtime.DIAGNOSTICS, "capture"),
            ):
                result = trace_runtime.TraceRuntime().run(context, argv)

            self.assertTrue(result)
            self.assertFalse(checkpoint.path.exists())

    def test_match_resumed_preserves_turn_and_snap_decisions(self) -> None:
        with TemporaryDirectory() as root:
            store = self.make_store(root)
            state = store.configure({}, now=0.0, wall_time=100.0)
            state.current_turn = 5
            state.first_snap_decision_made = True
            state.first_snap_committed = True
            state.final_snap_decision_made = True

            apply_event(state, "match_resumed")

            self.assertTrue(state.match_in_progress)
            self.assertEqual(state.current_turn, 5)
            self.assertTrue(state.first_snap_decision_made)
            self.assertTrue(state.first_snap_committed)
            self.assertTrue(state.final_snap_decision_made)

    def test_bootstrap_prefers_resume_wrappers_over_normal_battle_nodes(self) -> None:
        nodes = load_jsonc(
            ROOT / "assets/resource/pipeline/common/bootstrap.json"
        )
        route = nodes["公共-识别当前页面"]["next"]
        pairs = (
            ("公共-恢复对局继续", "公共-战斗继续"),
            ("公共-恢复可出牌回合", "公共-首回合"),
            ("公共-恢复等待对手", "公共-等待对手"),
            ("公共-恢复主界面", "公共-主界面"),
        )
        for resume, normal in pairs:
            with self.subTest(resume=resume):
                self.assertLess(route.index(resume), route.index(normal))
                event = nodes[resume]["action"]["param"][
                    "custom_action_param"
                ]["event"]
                self.assertIn(event, {"match_resumed", "page_home"})
        self.assertEqual(
            nodes["公共-恢复可出牌回合"]["next"],
            ["公共-撤退判断"],
        )


if __name__ == "__main__":
    unittest.main()
