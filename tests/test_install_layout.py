from contextlib import contextmanager
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


@contextmanager
def imported_installer():
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    original_argv = sys.argv
    sys.argv = ["install.py", "v-test", "win", "x86_64"]
    try:
        yield importlib.import_module("tools.install")
    finally:
        sys.argv = original_argv
        sys.path.remove(tools_dir)


class InstallLayoutTests(unittest.TestCase):
    def test_release_payload_contains_runnable_agent_and_user_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, imported_installer() as installer:
            destination = Path(temp_dir) / "install"
            installer.install_project_files(
                source_root=ROOT,
                destination=destination,
                release_version="v-test",
            )

            expected = {
                "interface.json",
                "resource",
                "tasks",
                "tasks/征服模式.json",
                "agent",
                "agent/requirements.txt",
                "README.md",
                "LICENSE",
            }
            for relative in expected:
                self.assertTrue((destination / relative).exists(), relative)

            self.assertFalse((destination / "tests").exists())
            self.assertFalse((destination / ".tmp").exists())
            self.assertFalse((destination / ".venv").exists())
            self.assertFalse(any(destination.rglob("__pycache__")))
            self.assertFalse(any(destination.rglob("*.pyc")))

            interface = installer.jsonc.loads(
                (destination / "interface.json").read_text("utf-8")
            )
            for imported in interface["import"]:
                self.assertTrue((destination / imported).is_file(), imported)

    def test_release_uses_bundled_agent_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, imported_installer() as installer:
            source = Path(temp_dir) / "source"
            shutil.copytree(ROOT / "assets", source / "assets")
            shutil.copytree(ROOT / "agent", source / "agent")
            shutil.copy2(ROOT / "README.md", source)
            shutil.copy2(ROOT / "LICENSE", source)
            bundle = source / "agent_dist" / "MAA_marvel_agent"
            bundle.mkdir(parents=True)
            (bundle / "MAA_marvel_agent.exe").write_bytes(b"agent")

            destination = Path(temp_dir) / "install"
            installer.install_project_files(source, destination, "v-test", "win")

            interface = installer.jsonc.loads(
                (destination / "interface.json").read_text("utf-8")
            )
            self.assertEqual(
                interface["agent"]["child_exec"],
                "./agent_runtime/MAA_marvel_agent.exe",
            )
            self.assertEqual(interface["agent"]["child_args"], [])
            self.assertTrue(
                (destination / "agent_runtime" / "MAA_marvel_agent.exe").is_file()
            )

            completed = subprocess.run(
                [str(PYTHON), "-c", "import agent.main"],
                cwd=destination,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_user_facing_files_have_no_template_identity(self) -> None:
        checked = [ROOT / "README.md", *sorted((ROOT / "docs" / "zh_cn").rglob("*.md"))]
        forbidden = ("MaaPracticeBoilerplate", "MaaXXX", "123456789")
        for path in checked:
            text = path.read_text("utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, f"{marker!r} remains in {path}")

    def test_obsolete_root_images_are_removed(self) -> None:
        image_root = ROOT / "assets" / "resource" / "image"
        obsolete = (
            "不在弹出勾.png",
            "empty.png",
            "1费.png",
            "2费.png",
            "3费.png",
            "5费.png",
        )
        self.assertFalse(
            [name for name in obsolete if (image_root / name).exists()]
        )


if __name__ == "__main__":
    unittest.main()
