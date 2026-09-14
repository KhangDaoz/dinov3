"""Strict image-ID alignment for the accepted M1 and M2 caches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from uncertainty_retrieval.config_e2a import E2AConfig
from uncertainty_retrieval.data.feature_cache import sha256_file
from uncertainty_retrieval.data.patch_cache import load_feature_cache


@dataclass(frozen=True)
class FusedFeatureCache:
    cls: Tensor
    mean_patch: Tensor
    image_ids: Tensor
    labels: Tensor
    splits: tuple[str, ...]
    cls_sha256: str
    mean_patch_sha256: str


def _load_verified(path: str, manifest_path: str) -> tuple[dict, dict, str]:
    cache_path = Path(path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    digest = sha256_file(cache_path)
    if manifest.get("cache_sha256") != digest:
        raise ValueError(f"Cache hash differs from manifest: {cache_path}")
    payload = load_feature_cache(cache_path)
    if manifest.get("metadata") != payload.get("metadata"):
        raise ValueError(f"Cache metadata differs from manifest: {cache_path}")
    return payload, manifest, digest


def load_fused_feature_cache(config: E2AConfig) -> FusedFeatureCache:
    """Join M1/M2 rows by ID and reject every provenance mismatch."""
    values = (
        config.cache.cls_path,
        config.cache.cls_manifest,
        config.cache.mean_patch_path,
        config.cache.mean_patch_manifest,
    )
    if any(value is None for value in values):
        raise ValueError("M3 cache paths and manifests are required")
    cls, cls_manifest, cls_hash = _load_verified(values[0], values[1])
    patch, patch_manifest, patch_hash = _load_verified(values[2], values[3])
    if cls_manifest.get("method") != "m1" or patch_manifest.get("method") != "m2":
        raise ValueError("M3 inputs must be accepted M1 and M2 artifacts")
    if cls["metadata"].get("token") != "cls":
        raise ValueError("M3 CLS cache has the wrong token source")
    patch_expected = {
        "token": "patch",
        "pooling": "mean",
        "patch_tokens": config.model.expected_patch_tokens,
        "embedding_dim": config.model.embedding_dim,
    }
    for key, expected in patch_expected.items():
        if patch["metadata"].get(key) != expected:
            raise ValueError(f"M3 mean-patch metadata mismatch for {key}")
    shared = (
        "model_id",
        "requested_revision",
        "resolved_revision",
        "register_tokens",
        "processor_settings",
        "cub_manifest_hash",
    )
    if any(cls["metadata"].get(key) != patch["metadata"].get(key) for key in shared):
        raise ValueError("M1/M2 shared provenance differs")
    expected_total = (
        config.dataset.expected_development_images
        + config.dataset.expected_test_images
    )
    expected_shape = (expected_total, config.model.embedding_dim)
    for name, payload in (("M1", cls), ("M2", patch)):
        features = payload["features"]
        if features.shape != expected_shape or features.dtype != torch.float32:
            raise ValueError(f"{name} feature shape or dtype differs")
        if not torch.isfinite(features).all():
            raise ValueError(f"{name} features contain NaN or Inf")

    cls_rows = {int(value): index for index, value in enumerate(cls["image_ids"])}
    patch_rows = {int(value): index for index, value in enumerate(patch["image_ids"])}
    if len(cls_rows) != expected_total or len(patch_rows) != expected_total:
        raise ValueError("M3 inputs contain duplicate or missing image IDs")
    if cls_rows.keys() != patch_rows.keys():
        raise ValueError("M1/M2 image-ID sets differ")
    ordered_ids = sorted(cls_rows)
    cls_indices = [cls_rows[value] for value in ordered_ids]
    patch_indices = [patch_rows[value] for value in ordered_ids]
    labels: list[int] = []
    splits: list[str] = []
    for image_id, left, right in zip(
        ordered_ids, cls_indices, patch_indices, strict=True
    ):
        cls_label, patch_label = cls["labels"][left], patch["labels"][right]
        cls_split, patch_split = cls["splits"][left], patch["splits"][right]
        if int(cls_label) != int(patch_label) or cls_split != patch_split:
            raise ValueError(f"M1/M2 label or split differs for image {image_id}")
        labels.append(int(cls_label))
        splits.append(cls_split)
    return FusedFeatureCache(
        cls=cls["features"][cls_indices].contiguous(),
        mean_patch=patch["features"][patch_indices].contiguous(),
        image_ids=torch.tensor(ordered_ids, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        splits=tuple(splits),
        cls_sha256=cls_hash,
        mean_patch_sha256=patch_hash,
    )
