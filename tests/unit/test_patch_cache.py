from pathlib import Path

import pytest
import torch

from uncertainty_retrieval.data.patch_cache import (
    load_feature_cache,
    save_feature_cache,
)
from uncertainty_retrieval.training.evidential import (
    load_evidential_outputs,
    save_evidential_outputs,
)


def test_feature_cache_round_trip_is_cpu_portable(tmp_path: Path) -> None:
    path = tmp_path / "features.pt"
    payload = {
        "schema_version": 1,
        "features": torch.randn(3, 4),
        "image_ids": [1, 2, 3],
        "labels": [0, 0, 1],
        "splits": ["development"] * 3,
        "metadata": {"model_id": "mock"},
    }
    save_feature_cache(payload, path)
    loaded = load_feature_cache(path)
    assert torch.equal(loaded["features"], payload["features"])
    assert loaded["features"].device.type == "cpu"


def test_feature_cache_rejects_inconsistent_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        save_feature_cache(
            {
                "schema_version": 1,
                "features": torch.randn(2, 4),
                "image_ids": [1],
                "labels": [0, 1],
                "splits": ["development", "development"],
                "metadata": {},
            },
            tmp_path / "bad.pt",
        )


def test_full_evidential_output_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "evidential.pt"
    evidence = torch.rand(2, 3)
    payload = {
        "image_ids": torch.tensor([1, 2]),
        "labels": torch.tensor([0, 1]),
        "evidence": evidence,
        "alpha": evidence + 1,
        "probabilities": (evidence + 1) / (evidence + 1).sum(1, keepdim=True),
        "uncertainty": 3 / (evidence + 1).sum(1),
        "predictions": torch.tensor([0, 1]),
        "class_order": [0, 1, 2],
        "dtype": "torch.float32",
        "schema_version": 1,
    }
    save_evidential_outputs(payload, path)
    loaded = load_evidential_outputs(path)
    assert torch.equal(loaded["evidence"], payload["evidence"])
    assert torch.equal(loaded["alpha"], payload["alpha"])
    assert torch.equal(loaded["uncertainty"], payload["uncertainty"])
