import time
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.commands import parse_json_object
from agent.runtime.diagnostics import DIAGNOSTICS
from agent.runtime.store import STORE


def _positive_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _image_shape(image: Any) -> tuple[int, int] | None:
    shape = getattr(image, "shape", None)
    if not shape or len(shape) < 2:
        return None
    height = int(shape[0])
    width = int(shape[1])
    if height <= 0 or width <= 0:
        return None
    return width, height


@AgentServer.custom_action("MarvelWarmupScreencap")
class WarmupScreencap(CustomAction):
    """Wait until the controller can return a real frame before root routing."""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        values = parse_json_object(argv.custom_action_param)
        timeout_ms = _positive_int(
            values.get("timeout_ms"),
            60000,
            minimum=1000,
            maximum=180000,
        )
        interval_ms = _positive_int(
            values.get("interval_ms"),
            1000,
            minimum=100,
            maximum=5000,
        )
        started = time.monotonic()
        deadline = started + timeout_ms / 1000.0
        attempts = 0
        last_error = "not attempted"

        while True:
            attempts += 1
            try:
                image = context.tasker.controller.post_screencap().get(wait=True)
                shape = _image_shape(image)
                if shape is not None:
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    print(
                        "[MarvelWarmupScreencap] success "
                        f"attempts={attempts} elapsed_ms={elapsed_ms} "
                        f"size={shape[0]}x{shape[1]}",
                        flush=True,
                    )
                    return True
                last_error = "empty image"
            except Exception as error:
                last_error = str(error)

            now = time.monotonic()
            if now >= deadline:
                break
            time.sleep(min(interval_ms / 1000.0, max(0.0, deadline - now)))

        elapsed_ms = int((time.monotonic() - started) * 1000)
        detail = {
            "attempts": attempts,
            "elapsed_ms": elapsed_ms,
            "last_error": last_error,
        }
        print(
            "[MarvelWarmupScreencap] failed "
            f"attempts={attempts} elapsed_ms={elapsed_ms} error={last_error}",
            flush=True,
        )
        DIAGNOSTICS.emit(
            STORE.state_or_none(),
            event="incident",
            source="warmup",
            reason="screencap_warmup_failed",
            node=str(getattr(argv, "node_name", "MarvelWarmupScreencap")),
            detail=detail,
        )
        return False
