#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Sequence

from PIL import Image


EXPECTED_SCREEN_SIZE = (1920, 1080)


def crop_image(
    source: Path,
    output: Path,
    box: Sequence[int],
) -> None:
    if len(box) != 4:
        raise ValueError("crop box must contain x, y, width and height")
    x, y, width, height = (int(value) for value in box)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid crop box: {box}")
    if x + width > EXPECTED_SCREEN_SIZE[0] or y + height > EXPECTED_SCREEN_SIZE[1]:
        raise ValueError(f"crop box is outside 1920x1080: {box}")

    with Image.open(source) as image:
        if image.size != EXPECTED_SCREEN_SIZE:
            raise ValueError(
                f"{source} must be 1920x1080, got {image.size[0]}x{image.size[1]}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x, y, x + width, y + height)).save(output, format="PNG")


def crop_manifest(manifest_path: Path, root: Path) -> int:
    manifest = json.loads(manifest_path.read_text("utf-8"))
    entries = manifest.get("templates")
    if not isinstance(entries, list):
        raise ValueError("manifest must contain a templates list")

    for entry in entries:
        crop_image(
            root / str(entry["source"]),
            root / str(entry["output"]),
            entry["box"],
        )
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop MAA templates from 1920x1080 fixtures")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    count = crop_manifest(args.manifest, args.root.resolve())
    print(f"[SUCCESS] Cropped {count} templates")


if __name__ == "__main__":
    main()
