from pathlib import Path

import pytest

from uncertainty_retrieval.data.cub import (
    CUBDataset,
    load_cub_records,
    split_development_records,
    validate_protocol_counts,
)


def test_uses_class_disjoint_split(cub_fixture: Path) -> None:
    records = load_cub_records(
        cub_fixture,
        development_classes=2,
        total_classes=4,
    )
    assert len(records) == 20
    development_labels = {
        record.original_label
        for record in records
        if record.split == "development"
    }
    assert development_labels == {
        0,
        1,
    }
    assert {record.original_label for record in records if record.split == "test"} == {
        2,
        3,
    }


def test_development_split_is_stratified_and_deterministic(
    cub_fixture: Path,
) -> None:
    records = load_cub_records(cub_fixture, 2, 4)
    first = split_development_records(records, 0.2, 42)
    second = split_development_records(records, 0.2, 42)
    assert first == second
    fit, validation = first
    assert len(fit) == 8
    assert len(validation) == 2
    assert {record.original_label for record in validation} == {0, 1}
    fit_ids = {record.image_id for record in fit}
    validation_ids = {record.image_id for record in validation}
    assert not fit_ids & validation_ids


def test_dataset_decodes_rgb(cub_fixture: Path) -> None:
    records = load_cub_records(cub_fixture, 2, 4)
    item = CUBDataset(cub_fixture, records)[0]
    assert item["pixel_values"].mode == "RGB"


def test_corrupt_image_has_actionable_error(cub_fixture: Path) -> None:
    records = load_cub_records(cub_fixture, 2, 4)
    image = cub_fixture / "images" / records[0].relative_path
    image.write_bytes(b"not an image")
    with pytest.raises(RuntimeError, match="Cannot decode CUB image"):
        CUBDataset(cub_fixture, records)[0]


def test_rejects_manifest_id_mismatch(cub_fixture: Path) -> None:
    labels = cub_fixture / "image_class_labels.txt"
    labels.write_text(
        labels.read_text(encoding="utf-8").replace("20 4", "21 4"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="IDs differ"):
        load_cub_records(cub_fixture, 2, 4)


def test_validates_protocol_counts(cub_fixture: Path) -> None:
    records = load_cub_records(cub_fixture, 2, 4)
    validate_protocol_counts(records, expected_development=10, expected_test=10)
    with pytest.raises(ValueError, match="count mismatch"):
        validate_protocol_counts(records, expected_development=11, expected_test=9)
