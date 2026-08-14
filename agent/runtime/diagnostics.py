from __future__ import annotations

from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from threading import RLock
import time
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from agent.session.state import SessionState


INCIDENT_ROOT = Path("debug") / "incidents"
EVENT_LOG = Path("debug") / "runtime-events.jsonl"


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_json_value(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class RuntimeDiagnostics:
    """集中保存运行事件、最近节点、手牌快照和异常现场。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._nodes: deque[dict[str, Any]] = deque(maxlen=20)
        self._task: dict[str, Any] = {}
        self._last_hand: dict[str, Any] | None = None
        self._last_capture_at: dict[str, float] = {}
        self._stop_intent: dict[str, Any] | None = None
        self._run_id: str | None = None

    def begin_task(self, task_id: int, entry: str, uuid: str, task_hash: str) -> None:
        with self._lock:
            self._nodes.clear()
            self._last_hand = None
            self._last_capture_at.clear()
            self._stop_intent = None
            self._run_id = uuid4().hex
        self.set_task(task_id, entry, uuid, task_hash)

    def begin_run(self, state: SessionState, *, restored: bool = False) -> None:
        with self._lock:
            self._nodes.clear()
            self._last_hand = None
            self._last_capture_at.clear()
            if restored:
                self._run_id = state.run_id
            elif self._run_id is None:
                self._run_id = state.run_id
            else:
                state.run_id = self._run_id
        self.emit(
            state,
            event="run_started",
            source="pipeline",
            reason="session_configured",
        )

    def set_task(self, task_id: int, entry: str, uuid: str, task_hash: str) -> None:
        with self._lock:
            self._task = {
                "task_id": task_id,
                "entry": entry,
                "uuid": uuid,
                "hash": task_hash,
            }

    def register_stop_intent(
        self,
        *,
        source: str,
        reason: str,
        node: str | None,
    ) -> None:
        with self._lock:
            self._stop_intent = {
                "source": source,
                "reason": reason,
                "node": node,
            }

    def stop_intent(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._stop_intent is None else dict(self._stop_intent)

    def record_node(self, task_id: int, name: str, event: str) -> None:
        with self._lock:
            self._nodes.append(
                {
                    "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    "task_id": task_id,
                    "name": name,
                    "event": event,
                }
            )

    def update_hand(self, hand: Any) -> None:
        cards = []
        for card in getattr(hand, "cards", ()):
            cards.append(
                {
                    "slot": getattr(card, "slot", None),
                    "cost": getattr(card, "cost", None),
                    "confidence": getattr(card, "confidence", None),
                    "box": list(getattr(card, "box", ())),
                }
            )
        with self._lock:
            self._last_hand = {
                "energy": getattr(hand, "energy", None),
                "reason": getattr(hand, "reason", None),
                "cards": cards,
            }

    def _payload(
        self,
        state: SessionState | None,
        *,
        event: str,
        source: str,
        reason: str,
        node: str | None = None,
        detail: Any = None,
    ) -> dict[str, Any]:
        with self._lock:
            task = dict(self._task)
            nodes = list(self._nodes)
            hand = None if self._last_hand is None else dict(self._last_hand)
            run_id = self._run_id
        state_data = None if state is None else _json_value(state)
        return {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "run_id": state.run_id if state is not None else run_id,
            "event": event,
            "source": source,
            "reason": reason,
            "node": node or (nodes[-1]["name"] if nodes else None),
            "page": None if state is None else state.last_known_state,
            "task": task,
            "session": state_data,
            "last_hand": hand,
            "recent_nodes": nodes,
            "detail": _json_value(detail),
        }

    def emit(
        self,
        state: SessionState | None,
        *,
        event: str,
        source: str,
        reason: str,
        node: str | None = None,
        detail: Any = None,
    ) -> dict[str, Any]:
        payload = self._payload(
            state,
            event=event,
            source=source,
            reason=reason,
            node=node,
            detail=detail,
        )
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        print(f"[MarvelRuntimeEvent] {encoded}", flush=True)
        try:
            EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
            with EVENT_LOG.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
        except OSError as error:
            print(f"[MarvelRuntimeEvent] write_failed error={error}", flush=True)
        return payload

    def capture(
        self,
        controller: Any,
        state: SessionState | None,
        *,
        source: str,
        reason: str,
        node: str | None = None,
        detail: Any = None,
        throttle_seconds: float = 10.0,
        image: Any = None,
    ) -> Path | None:
        throttle_key = f"{source}:{reason}:{node or ''}"
        now = time.monotonic()
        with self._lock:
            previous = self._last_capture_at.get(throttle_key)
            if previous is not None and now - previous < throttle_seconds:
                return None
            self._last_capture_at[throttle_key] = now

        payload = self.emit(
            state,
            event="incident",
            source=source,
            reason=reason,
            node=node,
            detail=detail,
        )
        run_id = payload["run_id"] or "unconfigured"
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        incident_dir = INCIDENT_ROOT / str(run_id) / f"{stamp}-{reason}"
        try:
            incident_dir.mkdir(parents=True, exist_ok=True)
            (incident_dir / "metadata.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if image is None and controller is not None:
                image = controller.post_screencap().get(wait=True)
            if image is None:
                return incident_dir
            pixels = np.asarray(image)
            if pixels.ndim == 3 and pixels.shape[2] >= 3:
                rgb = pixels[..., :3][..., ::-1]
                Image.fromarray(rgb.astype(np.uint8), "RGB").save(
                    incident_dir / "screenshot.png"
                )
            else:
                np.save(incident_dir / "screenshot.npy", pixels)
            return incident_dir
        except Exception as error:
            print(
                f"[MarvelRuntimeEvent] capture_failed reason={reason} error={error}",
                flush=True,
            )
            return None


DIAGNOSTICS = RuntimeDiagnostics()
