import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from tools.crop_templates import crop_image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "screens" / "manifest.json"


class TemplateManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entries = json.loads(MANIFEST.read_text("utf-8"))["templates"]

    def test_manifest_has_unique_outputs_and_existing_samples(self) -> None:
        outputs = [entry["output"] for entry in self.entries]
        self.assertEqual(len(outputs), len(set(outputs)))
        for entry in self.entries:
            with self.subTest(output=entry["output"]):
                self.assertTrue((ROOT / entry["source"]).is_file())
                self.assertTrue(entry["positive_nodes"])
                self.assertTrue(entry["negative_sources"])
                for negative in entry["negative_sources"]:
                    self.assertTrue((ROOT / negative).is_file())

    def test_sources_are_720_by_1280_and_boxes_are_inside(self) -> None:
        for entry in self.entries:
            source = ROOT / entry["source"]
            with Image.open(source) as image:
                self.assertEqual(image.size, (720, 1280))
            x, y, width, height = entry["box"]
            self.assertGreater(width, 0)
            self.assertGreater(height, 0)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + width, 720)
            self.assertLessEqual(y + height, 1280)
            self.assertNotEqual((width, height), (720, 1280))

    def test_every_manifest_crop_has_the_declared_size(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            for index, entry in enumerate(self.entries):
                output = temporary / f"{index}.png"
                crop_image(ROOT / entry["source"], output, entry["box"])
                with Image.open(output) as cropped:
                    self.assertEqual(cropped.size, tuple(entry["box"][2:]))

    def test_crop_rejects_non_720p_source(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "wrong.png"
            Image.new("RGB", (1080, 1920)).save(source)
            with self.assertRaises(ValueError):
                crop_image(source, temporary / "output.png", [0, 0, 10, 10])


if __name__ == "__main__":
    unittest.main()
