from dataclasses import replace

import pytest
import torch

from uncertainty_retrieval.config_e2a import E2AConfig
from uncertainty_retrieval.data.cub import CUBRecord
from uncertainty_retrieval.data.feature_cache import validate_m1_cache, validation_ids_hash


def _fixture() -> tuple[dict, list[CUBRecord], E2AConfig]:
    config = E2AConfig()
    config = replace(
        config,
        dataset=replace(config.dataset, expected_development_images=2, expected_test_images=1),
        model=replace(config.model, embedding_dim=3),
    )
    records = [
        CUBRecord(1, "a.jpg", 0, "development"),
        CUBRecord(2, "b.jpg", 1, "development"),
        CUBRecord(3, "c.jpg", 100, "test"),
    ]
    metadata = {
        "model_id": config.model.model_id,
        "requested_revision": config.model.revision,
        "resolved_revision": config.model.revision,
        "token": "cls",
        "source_layer": "final",
        "register_tokens": config.model.register_tokens,
        "processor_settings": {"size": 224},
        "cub_manifest_hash": "manifest",
    }
    payload = {
        "schema_version": config.cache.schema_version,
        "features": torch.ones(3, 3),
        "image_ids": [1, 2, 3],
        "labels": [0, 1, 100],
        "splits": ["development", "development", "test"],
        "metadata": metadata,
    }
    return payload, records, config


def test_m1_cache_accepts_complete_provenance() -> None:
    payload, records, config = _fixture()
    validate_m1_cache(payload, records, config, "manifest")


@pytest.mark.parametrize("fault", ("duplicate", "label", "nan", "schema"))
def test_m1_cache_rejects_invalid_content(fault: str) -> None:
    payload, records, config = _fixture()
    if fault == "duplicate":
        payload["image_ids"] = [1, 1, 3]
    elif fault == "label":
        payload["labels"][0] = 9
    elif fault == "nan":
        payload["features"][0, 0] = torch.nan
    else:
        payload["schema_version"] = 99
    with pytest.raises(ValueError):
        validate_m1_cache(payload, records, config, "manifest")


def test_validation_hash_is_order_independent() -> None:
    assert validation_ids_hash([3, 1, 2]) == validation_ids_hash([1, 2, 3])
