import json
from dataclasses import replace

import torch

from uncertainty_retrieval.config_e2a import E2AConfig, RepresentationConfig
from uncertainty_retrieval.data.feature_cache import sha256_file
from uncertainty_retrieval.data.fused_cache import load_fused_feature_cache
from uncertainty_retrieval.data.patch_cache import save_feature_cache


def _save(path, features, ids, labels, splits, metadata, method) -> str:
    save_feature_cache(
        {
            "schema_version": 2,
            "features": features,
            "image_ids": ids,
            "labels": labels,
            "splits": splits,
            "metadata": metadata,
        },
        path,
    )
    manifest = path.with_suffix(".json")
    manifest.write_text(
        json.dumps(
            {
                "method": method,
                "cache_sha256": sha256_file(path),
                "metadata": metadata,
            }
        ),
        encoding="utf-8",
    )
    return str(manifest)


def test_fused_cache_aligns_permuted_rows_by_image_id(tmp_path) -> None:
    shared = {
        "model_id": "model",
        "requested_revision": "rev",
        "resolved_revision": "rev",
        "register_tokens": 4,
        "processor_settings": {"size": 224},
        "cub_manifest_hash": "cub",
    }
    cls_path = tmp_path / "cls.pt"
    patch_path = tmp_path / "patch.pt"
    cls_manifest = _save(
        cls_path,
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        [1, 2],
        [0, 1],
        ["development", "test"],
        {**shared, "token": "cls"},
        "m1",
    )
    patch_manifest = _save(
        patch_path,
        torch.tensor([[20.0, 20.0], [10.0, 10.0]]),
        [2, 1],
        [1, 0],
        ["test", "development"],
        {
            **shared,
            "token": "patch",
            "pooling": "mean",
            "patch_tokens": 3,
            "embedding_dim": 2,
        },
        "m2",
    )
    config = E2AConfig(
        dataset=replace(
            E2AConfig().dataset,
            expected_development_images=1,
            expected_test_images=1,
        ),
        model=replace(
            E2AConfig().model, embedding_dim=2, expected_patch_tokens=3
        ),
        representation=RepresentationConfig(
            "m3", "cls_mean_projection", "final", "projection", "l2"
        ),
        cache=replace(
            E2AConfig().cache,
            cls_path=str(cls_path),
            mean_patch_path=str(patch_path),
            cls_manifest=cls_manifest,
            mean_patch_manifest=patch_manifest,
        ),
    )
    fused = load_fused_feature_cache(config)
    assert fused.image_ids.tolist() == [1, 2]
    assert fused.mean_patch.tolist() == [[10.0, 10.0], [20.0, 20.0]]
    assert fused.labels.tolist() == [0, 1]
