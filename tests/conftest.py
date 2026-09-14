"""Test configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


@pytest.fixture
def cub_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "CUB_200_2011"
    images = root / "images"
    images.mkdir(parents=True)
    rows = []
    labels = []
    image_id = 1
    for label in range(1, 5):
        for offset in range(5):
            relative = f"{label:03d}.class/image_{offset}.jpg"
            (images / relative).parent.mkdir(exist_ok=True)
            from PIL import Image

            Image.new("RGB", (8, 8), color=(label * 20, offset * 20, 0)).save(
                images / relative
            )
            rows.append(f"{image_id} {relative}")
            labels.append(f"{image_id} {label}")
            image_id += 1
    (root / "images.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "image_class_labels.txt").write_text(
        "\n".join(labels) + "\n",
        encoding="utf-8",
    )
    return root

