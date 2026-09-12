from pathlib import Path

import pytest
from PIL import Image

from src.dataset import CUBirds, collate_pil_batch


def make_class(root, name):
    class_dir = root / "CUB_200_2011" / "images" / name
    class_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2), color="white").save(class_dir / "sample.jpg")


def test_class_disjoint_splits(tmp_path):
    for class_id in range(1, 201):
        make_class(tmp_path, f"{class_id:03d}.Synthetic")
    train = CUBirds(tmp_path, "train")
    evaluation = CUBirds(tmp_path, "eval")
    assert len(train) == 100
    assert len(evaluation) == 100
    assert set(train.ys) == set(range(100))
    assert set(evaluation.ys) == set(range(100, 200))


def test_invalid_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Split"):
        CUBirds(tmp_path, "test")


def test_collate_keeps_images_as_list():
    image = Image.new("RGB", (2, 2))
    images, labels = collate_pil_batch([(image, 3), (image, 4)])
    assert images == [image, image]
    assert labels.tolist() == [3, 4]
