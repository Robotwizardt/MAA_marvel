from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


MUMU_EXTRAS_SCREENSHOT = 1 << 6
ADB_LOSSLESS_SCREENSHOTS = 1 | (1 << 1) | (1 << 2)
MUMU_WITH_ADB_FALLBACK_SCREENSHOTS = (
    MUMU_EXTRAS_SCREENSHOT | ADB_LOSSLESS_SCREENSHOTS
)


def _catalog_version(task_file: Path) -> str:
    return hashlib.sha256(task_file.read_bytes()).hexdigest()[:16]


def _default_option(name: str, definition: dict[str, Any]) -> dict[str, Any]:
    option_type = definition.get("type")
    result: dict[str, Any] = {"name": name, "index": 0}
    if option_type == "input":
        result["data"] = {
            str(item["name"]): str(item.get("default", ""))
            for item in definition.get("inputs", [])
        }
        return result

    cases = definition.get("cases", [])
    default_case = definition.get("default_case")
    for index, case in enumerate(cases):
        if case.get("name") == default_case:
            result["index"] = index
            break
    return result


def _task_item(
    task: dict[str, Any],
    option_definitions: dict[str, Any],
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = {
        key: deepcopy(value)
        for key, value in task.items()
        if key != "option"
    }
    previous_options = {
        str(item.get("name")): item
        for item in (existing or {}).get("option", [])
        if isinstance(item, dict) and item.get("name")
    }
    options = []
    for name in task.get("option", []):
        if name in previous_options:
            options.append(deepcopy(previous_options[name]))
            continue
        definition = option_definitions.get(name, {})
        options.append(_default_option(name, definition))
    merged["option"] = options
    return merged


def merge_task_catalog(
    instance: dict[str, Any],
    catalog: dict[str, Any],
    *,
    version: str,
) -> tuple[dict[str, Any], bool]:
    """按 entry 合并任务清单；保留已有 option 的 index/data。"""
    output = deepcopy(instance)
    existing_items = {
        str(item.get("entry")): item
        for item in output.get("TaskItems", [])
        if isinstance(item, dict) and item.get("entry")
    }
    catalog_entries: set[str] = set()
    merged_items: list[dict[str, Any]] = []
    options = catalog.get("option", {})
    for task in catalog.get("task", []):
        entry = str(task.get("entry", ""))
        if not entry:
            continue
        catalog_entries.add(entry)
        merged_items.append(
            _task_item(task, options, existing_items.get(entry))
        )

    for entry, item in existing_items.items():
        if entry not in catalog_entries:
            merged_items.append(deepcopy(item))

    current_tasks = [
        f"{item.get('name', '')}<|||>{item.get('entry', '')}"
        for item in merged_items
        if item.get("name") and item.get("entry")
    ]
    output["TaskItems"] = merged_items
    output["CurrentTasks"] = current_tasks
    output["TaskCatalogVersion"] = version
    _ensure_adb_screencap_fallback(output)
    return output, output != instance


def _ensure_adb_screencap_fallback(instance: dict[str, Any]) -> None:
    device = instance.get("AdbDevice")
    if not isinstance(device, dict):
        return
    if device.get("ScreencapMethods") != MUMU_EXTRAS_SCREENSHOT:
        return
    device["ScreencapMethods"] = MUMU_WITH_ADB_FALLBACK_SCREENSHOTS


def migrate_instance_file(task_file: Path, instance_file: Path) -> bool:
    catalog = json.loads(task_file.read_text(encoding="utf-8"))
    instance = json.loads(instance_file.read_text(encoding="utf-8"))
    merged, changed = merge_task_catalog(
        instance,
        catalog,
        version=_catalog_version(task_file),
    )
    if not changed:
        return False
    temporary = instance_file.with_suffix(instance_file.suffix + ".tmp")
    temporary.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(instance_file)
    return True


def migrate_runtime_task_cache(root: Path | None = None) -> list[Path]:
    """在 MFA/Agent 启动时合并实例缓存；找不到布局时安静跳过。"""
    base = Path.cwd() if root is None else root
    task_candidates = (
        base / "tasks" / "征服模式.json",
        base / "assets" / "tasks" / "征服模式.json",
    )
    task_file = next((path for path in task_candidates if path.is_file()), None)
    config_root = base / "config" / "instances"
    if task_file is None or not config_root.is_dir():
        return []

    changed: list[Path] = []
    for instance_file in sorted(config_root.glob("*.json")):
        try:
            if migrate_instance_file(task_file, instance_file):
                changed.append(instance_file)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(
                f"[MarvelTaskCache] migrate_failed file={instance_file} error={error}",
                flush=True,
            )
    if changed:
        print(
            "[MarvelTaskCache] migrated="
            + ",".join(str(path) for path in changed),
            flush=True,
        )
    return changed
