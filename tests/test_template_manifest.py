import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from tools.crop_templates import crop_image
from tools.validate_schema import load_jsonc


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "screens" / "manifest.json"
PIPELINE_ROOT = ROOT / "assets" / "resource" / "pipeline"


class LandscapeFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text("utf-8"))

    def test_all_reference_screens_are_native_landscape(self) -> None:
        self.assertEqual(self.manifest["screen_size"], [1920, 1080])
        self.assertTrue(self.manifest["screens"])
        for relative_path in self.manifest["screens"]:
            with self.subTest(source=relative_path):
                source = ROOT / relative_path
                self.assertTrue(source.is_file())
                with Image.open(source) as image:
                    self.assertEqual(image.size, (1920, 1080))

    def test_pipeline_no_longer_references_portrait_templates(self) -> None:
        self.assertEqual(self.manifest["templates"], [])
        for path in PIPELINE_ROOT.rglob("*.json"):
            with self.subTest(path=path):
                nodes = load_jsonc(path)
                for node in nodes.values():
                    recognition = node.get("recognition", {})
                    if isinstance(recognition, dict):
                        self.assertNotEqual(recognition.get("type"), "TemplateMatch")

    def test_crop_accepts_landscape_source(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "crop.png"
            source = ROOT / self.manifest["screens"][0]
            crop_image(source, output, [100, 100, 200, 150])
            with Image.open(output) as cropped:
                self.assertEqual(cropped.size, (200, 150))

    def test_crop_rejects_portrait_source(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "portrait.png"
            Image.new("RGB", (1080, 1920)).save(source)
            with self.assertRaises(ValueError):
                crop_image(source, temporary / "output.png", [0, 0, 10, 10])


if __name__ == "__main__":
    unittest.main()
