import unittest

from agent.runtime.task_cache import merge_task_catalog


class TaskCacheTests(unittest.TestCase):
    def test_new_tasks_and_options_are_merged_by_entry(self) -> None:
        catalog = {
            "task": [
                {
                    "name": "征服",
                    "entry": "conquest",
                    "default_check": True,
                    "option": ["卡组", "新选项"],
                },
                {
                    "name": "邮箱",
                    "entry": "mail",
                    "default_check": False,
                    "option": [],
                },
            ],
            "option": {
                "卡组": {
                    "type": "input",
                    "inputs": [{"name": "名称", "default": "0"}],
                },
                "新选项": {
                    "type": "select",
                    "default_case": "on",
                    "cases": [{"name": "off"}, {"name": "on"}],
                },
            },
        }
        instance = {
            "TaskItems": [
                {
                    "name": "旧征服名",
                    "entry": "conquest",
                    "option": [
                        {"name": "卡组", "index": 0, "data": {"名称": "动物园"}}
                    ],
                }
            ]
        }

        merged, changed = merge_task_catalog(instance, catalog, version="abc")

        self.assertTrue(changed)
        self.assertEqual([item["entry"] for item in merged["TaskItems"]], ["conquest", "mail"])
        conquest = merged["TaskItems"][0]
        self.assertEqual(conquest["option"][0]["data"]["名称"], "动物园")
        self.assertEqual(conquest["option"][1]["index"], 1)
        self.assertEqual(merged["TaskCatalogVersion"], "abc")

    def test_mumu_only_screencap_gets_adb_lossless_fallback(self) -> None:
        instance = {
            "AdbDevice": {
                "ScreencapMethods": 64,
            },
            "TaskItems": [],
        }
        catalog = {"task": [], "option": {}}

        merged, changed = merge_task_catalog(instance, catalog, version="abc")

        self.assertTrue(changed)
        self.assertEqual(merged["AdbDevice"]["ScreencapMethods"], 71)

    def test_existing_screencap_fallback_is_preserved(self) -> None:
        instance = {
            "AdbDevice": {
                "ScreencapMethods": 71,
            },
            "TaskItems": [],
        }
        catalog = {"task": [], "option": {}}

        merged, changed = merge_task_catalog(instance, catalog, version="abc")

        self.assertTrue(changed)
        self.assertEqual(merged["AdbDevice"]["ScreencapMethods"], 71)


if __name__ == "__main__":
    unittest.main()
