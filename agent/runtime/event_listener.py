from __future__ import annotations

from threading import Thread

from maa.agent.agent_server import AgentServer
from maa.context import Context, ContextEventSink
from maa.event_sink import NotificationType
from maa.tasker import Tasker, TaskerEventSink

from agent.runtime.diagnostics import DIAGNOSTICS
from agent.runtime.store import STORE


ROOT_ENTRIES = {
    "日常-任务入口",
    "征服-任务入口",
    "天梯-任务入口",
    "训练-OCR选牌测试入口",
    "邮箱-任务入口",
}


def _state():
    return STORE.state_or_none()


def _capture_async(controller, state, **kwargs) -> None:
    # Context/Tasker event callbacks expose a temporary controller proxy.  The
    # proxy identifier becomes invalid as soon as the callback returns, so it
    # must never be carried into the background writer thread.  Copy MaaFW's
    # latest cached frame while the proxy is still valid; the thread only
    # persists the detached numpy image and metadata.
    image = None
    try:
        image = controller.cached_image.copy()
    except Exception as error:
        print(
            f"[MarvelRuntimeEvent] cached_image_failed error={error}",
            flush=True,
        )
    Thread(
        target=DIAGNOSTICS.capture,
        args=(None, state),
        kwargs={**kwargs, "image": image},
        daemon=True,
        name="marvel-incident-capture",
    ).start()


@AgentServer.tasker_sink()
class MarvelTaskerEventSink(TaskerEventSink):
    """记录根任务的开始/结束；未登记的停止统一标记为外部或前端来源。"""

    def on_tasker_task(self, tasker: Tasker, noti_type, detail) -> None:
        if detail.entry not in ROOT_ENTRIES:
            return
        if noti_type is NotificationType.Starting:
            STORE.clear_state()
            DIAGNOSTICS.begin_task(
                detail.task_id,
                detail.entry,
                detail.uuid,
                detail.hash,
            )
            DIAGNOSTICS.emit(
                None,
                event="task_started",
                source="framework",
                reason="task_starting",
                detail={"entry": detail.entry, "task_id": detail.task_id},
            )
            return
        DIAGNOSTICS.set_task(detail.task_id, detail.entry, detail.uuid, detail.hash)
        state = _state()
        if noti_type not in {NotificationType.Succeeded, NotificationType.Failed}:
            return

        stop_intent = DIAGNOSTICS.stop_intent()
        if stop_intent is not None:
            reason = str(stop_intent["reason"])
            source = str(stop_intent["source"])
        elif state is not None and state.stop_reason is not None:
            reason = state.stop_reason.value
            source = state.last_stop_source or "pipeline"
        elif noti_type is NotificationType.Succeeded:
            reason = "task_succeeded_without_stop_request"
            source = "framework"
        else:
            # Tasker.Task.Failed 只说明 Pipeline 失败，不能据此推断用户点击了
            # 停止。前端主动停止若没有显式 stop_intent，也应保留为未知来源，
            # 避免把截图失败、节点超时等真实故障错误归因给用户。
            reason = "task_failed_without_stop_request"
            source = "framework"
        detail_data = {
            "entry": detail.entry,
            "task_id": detail.task_id,
            "notification": noti_type.name,
            "stop_intent": stop_intent,
        }
        DIAGNOSTICS.emit(
            state,
            event="task_finished",
            source=source,
            reason=reason,
            detail=detail_data,
        )
        if stop_intent is None:
            STORE.persist_checkpoint()
        _capture_async(
            tasker.controller,
            state,
            source=source,
            reason=reason,
            detail=detail_data,
            throttle_seconds=0.0,
        )


@AgentServer.context_sink()
class MarvelContextEventSink(ContextEventSink):
    """维护最近节点，并只在 Pipeline 节点最终失败时保存现场。"""

    def on_node_pipeline_node(
        self,
        context: Context,
        noti_type,
        detail,
    ) -> None:
        DIAGNOSTICS.record_node(detail.task_id, detail.name, noti_type.name)
        if noti_type is NotificationType.Failed:
            _capture_async(
                context.tasker.controller,
                _state(),
                source="pipeline",
                reason="pipeline_node_failed",
                node=detail.name,
                detail={"task_id": detail.task_id},
                throttle_seconds=30.0,
            )
