"""CUB-200-2011 class-disjoint retrieval dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CUBRecord:
    """One image referenced by the original CUB manifests."""

    image_id: int
    relative_path: str
    original_label: int
    split: str


def _read_indexed_manifest(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CUB manifest: {path}")
    values: dict[int, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed {path.name} line {line_number}")
        index = int(parts[0])
        if index in values:
            raise ValueError(f"Duplicate image id {index} in {path.name}")
        values[index] = parts[1]
    return values


def load_cub_records(
    root: str | Path,
    development_classes: int = 100,
    total_classes: int = 200,
) -> list[CUBRecord]:
    """Load records using the fixed class split, ignoring image split metadata."""
    root_path = Path(root)
    paths = _read_indexed_manifest(root_path / "images.txt")
    labels_raw = _read_indexed_manifest(root_path / "image_class_labels.txt")
    if paths.keys() != labels_raw.keys():
        raise ValueError("images.txt and image_class_labels.txt IDs differ")
    records: list[CUBRecord] = []
    for image_id in sorted(paths):
        original_label = int(labels_raw[image_id]) - 1
        if not 0 <= original_label < total_classes:
            raise ValueError(f"Invalid class label for image {image_id}")
        split = "development" if original_label < development_classes else "test"
        records.append(
            CUBRecord(image_id, paths[image_id], original_label, split)
        )
    return records


def split_development_records(
    records: Iterable[CUBRecord],
    validation_fraction: float,
    seed: int,
) -> tuple[list[CUBRecord], list[CUBRecord]]:
    """Create a deterministic, stratified per-class image split."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    by_class: dict[int, list[CUBRecord]] = {}
    for record in records:
        if record.split != "development":
            continue
        by_class.setdefault(record.original_label, []).append(record)
    if not by_class:
        raise ValueError("No development records were provided")

    import random

    fit: list[CUBRecord] = []
    validation: list[CUBRecord] = []
    for label, class_records in sorted(by_class.items()):
        ordered = sorted(class_records, key=lambda record: record.image_id)
        random.Random(seed + label).shuffle(ordered)
        count = max(1, round(len(ordered) * validation_fraction))
        if count >= len(ordered):
            raise ValueError(f"Class {label} has too few images to split")
        validation.extend(ordered[:count])
        fit.extend(ordered[count:])
    return (
        sorted(fit, key=lambda record: record.image_id),
        sorted(validation, key=lambda record: record.image_id),
    )


def validate_protocol_counts(
    records: Iterable[CUBRecord],
    expected_development: int,
    expected_test: int,
) -> None:
    """Fail early when the local CUB manifests do not match the fixed protocol."""
    counts = {"development": 0, "test": 0}
    for record in records:
        counts[record.split] += 1
    expected = {
        "development": expected_development,
        "test": expected_test,
    }
    if counts != expected:
        raise ValueError(
            f"CUB protocol count mismatch: got {counts}, expected {expected}"
        )


class CUBDataset(Dataset):
    """Decode CUB images on CPU and apply an optional processor."""

    def __init__(
        self,
        root: str | Path,
        records: list[CUBRecord],
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image_path = self.root / "images" / record.relative_path
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Cannot decode CUB image: {image_path}") from error
        pixel_values = self.transform(image) if self.transform else image
        return {
            "pixel_values": pixel_values,
            "image_id": record.image_id,
            "label": record.original_label,
            "split": record.split,
            "relative_path": record.relative_path,
        }
