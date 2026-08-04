import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.event_sink import NotificationType

from agent.runtime.diagnostics import RuntimeDiagnostics
from agent.runtime.event_listener import MarvelContextEventSink, _capture_async
from agent.session.config import SessionConfig
from agent.session.state import SessionState


class AgentImportTests(unittest.TestCase):
    def test_async_incident_capture_detaches_cached_frame_from_controller(self) -> None:
        frame = np.zeros((8, 12, 3), dtype=np.uint8)
        controller = MagicMock()
        controller.cached_image = frame

        class ImmediateThread:
            def __init__(self, *, target, args, kwargs, **_options) -> None:
                self.target = target
                self.args = args
                self.kwargs = kwargs

            def start(self) -> None:
                self.target(*self.args, **self.kwargs)

        with (
            patch("agent.runtime.event_listener.Thread", ImmediateThread),
            patch("agent.runtime.event_listener.DIAGNOSTICS.capture") as capture,
        ):
            _capture_async(
                controller,
                None,
                source="pipeline",
                reason="pipeline_next_list_timeout",
            )

        capture.assert_called_once()
        args, kwargs = capture.call_args
        self.assertIsNone(args[0])
        self.assertIsNone(args[1])
        self.assertIsNot(kwargs["image"], frame)
        np.testing.assert_array_equal(kwargs["image"], frame)

    def test_context_sink_captures_only_final_pipeline_failure(self) -> None:
        sink = MarvelContextEventSink()
        context = MagicMock()
        detail = SimpleNamespace(task_id=7, name="测试节点")

        with (
            patch("agent.runtime.event_listener.DIAGNOSTICS.record_node") as record,
            patch("agent.runtime.event_listener._capture_async") as capture,
            patch("agent.runtime.event_listener._state", return_value=None),
        ):
            sink.on_node_pipeline_node(
                context,
                NotificationType.Starting,
                detail,
            )
            capture.assert_not_called()

            sink.on_node_pipeline_node(
                context,
                NotificationType.Failed,
                detail,
            )

        self.assertEqual(record.call_count, 2)
        capture.assert_called_once_with(
            context.tasker.controller,
            None,
            source="pipeline",
            reason="pipeline_node_failed",
            node="测试节点",
            detail={"task_id": 7},
            throttle_seconds=30.0,
        )

    def test_main_registers_expected_adapters_without_starting_socket(self) -> None:
        importlib.import_module("agent.main")

        self.assertTrue(
            {
                "MarvelConfigureSession",
                "MarvelPlayTurn",
                "MarvelRecordEvent",
                "MarvelRouteConquestTier",
                "MarvelRecoveryAction",
                "MarvelTraceRuntime",
                "MarvelWarmupScreencap",
            }.issubset(AgentServer._custom_action_holder)
        )
        self.assertTrue(
            {
                "MarvelSessionGate",
                "MarvelCardSelection",
                "MarvelDailyTaskReward",
                "MarvelSafeEntry",
            }.issubset(AgentServer._custom_recognition_holder)
        )

    def test_diagnostics_reuses_root_task_run_id_for_session(self) -> None:
        diagnostics = RuntimeDiagnostics()
        diagnostics.begin_task(7, "征服-任务入口", "uuid", "hash")
        started = diagnostics._payload(
            None,
            event="task_started",
            source="framework",
            reason="task_starting",
        )
        state = SessionState(SessionConfig(), started_at=0.0)

        diagnostics.begin_run(state)

        self.assertIsNotNone(started["run_id"])
        self.assertEqual(state.run_id, started["run_id"])


if __name__ == "__main__":
    unittest.main()
