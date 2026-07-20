import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ValidatorOutputTests(unittest.TestCase):
    def test_schema_validator_runs_with_windows_gbk_output(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "gbk"
        result = subprocess.run(
            [
                sys.executable,
                "tools/validate_schema.py",
                "--schema-dir",
                "deps/tools",
                "--resource-dirs",
                "assets/resource",
                "--interface-files",
                "assets/interface.json",
                "--task-dirs",
                "assets/tasks",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="gbk",
            errors="replace",
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("All validations passed", result.stdout)
