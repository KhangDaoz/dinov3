"""Sharded patch-token caches for E2A-M4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from uncertainty_retrieval.config_e2a import E2AConfig
from uncertainty_retrieval.data.cub import CUBRecord
from uncertainty_retrieval.data.feature_cache import sha256_file
from uncertainty_retrieval.utils import write_json


def fp16_fidelity_statistics(reference: Tensor) -> dict[str, float]:
    """Measure storage-only FP16 round-trip error against FP32 tokens."""
    if reference.ndim != 3 or reference.dtype != torch.float32:
        raise ValueError("Fidelity reference must be FP32 [images, patches, dim]")
    restored = reference.half().float()
    difference = (restored - reference).flatten(1)
    relative = difference.norm(dim=1) / reference.flatten(1).norm(dim=1).clamp_min(1e-12)
    patch_cosine = functional.cosine_similarity(reference, restored, dim=2)
    mean_cosine = functional.cosine_similarity(
        reference.mean(dim=1), restored.mean(dim=1), dim=1
    )
    return {
        "max_relative_l2": float(relative.max()),
        "min_patch_cosine": float(patch_cosine.min()),
        "min_mean_cosine": float(mean_cosine.min()),
    }


def validate_fidelity(statistics: dict[str, float], config: E2AConfig) -> None:
    cache = config.cache
    failures = []
    if statistics["max_relative_l2"] > cache.fidelity_relative_l2_max:
        failures.append("relative_l2")
    if statistics["min_patch_cosine"] < cache.fidelity_patch_cosine_min:
        failures.append("patch_cosine")
    if statistics["min_mean_cosine"] < cache.fidelity_mean_cosine_min:
        failures.append("mean_cosine")
    if failures:
        raise ValueError(f"FP16 patch-cache fidelity failed: {failures}")


@dataclass
class PatchTokenCache:
    shards: list[dict[str, Any]]
    image_ids: Tensor
    labels: Tensor
    locations: dict[int, tuple[int, int]]
    manifest: dict[str, Any]

    def batch(self, image_ids: Tensor | list[int]) -> Tensor:
        rows = []
        for value in image_ids:
            image_id = int(value)
            if image_id not in self.locations:
                raise KeyError(f"Patch cache does not contain image {image_id}")
            shard, row = self.locations[image_id]
            rows.append(self.shards[shard]["features"][row])
        return torch.stack(rows)


def load_patch_token_cache(
    config: E2AConfig,
    records: list[CUBRecord],
    scope: str = "validation",
) -> PatchTokenCache:
    if scope not in {"validation", "test"}:
        raise ValueError(f"Unsupported patch-cache scope: {scope}")
    manifest_path = Path(config.cache.patch_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing M4 patch manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = json.loads(
        Path(config.cache.reference_manifest).read_text(encoding="utf-8")
    )
    metadata = manifest.get("metadata", {})
    reference_metadata = reference.get("metadata", {})
    shared = (
        "model_id", "requested_revision", "resolved_revision",
        "register_tokens", "processor_settings", "cub_manifest_hash",
    )
    if any(metadata.get(key) != reference_metadata.get(key) for key in shared):
        raise ValueError("M4 patch-cache provenance differs from accepted M2")
    expected_split = "development" if scope == "validation" else "test"
    expected_metadata = {
        "token": "patch",
        "source_layer": "final",
        "patch_tokens": config.model.expected_patch_tokens,
        "embedding_dim": config.model.embedding_dim,
        "scope": f"{expected_split}_only",
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("M4 patch-cache representation metadata differs")
    accepted_fp16 = manifest["fidelity"].get("accepted_fp16") is True
    if accepted_fp16:
        validate_fidelity(manifest["fidelity"]["statistics"], config)
        if metadata.get("storage_dtype") != "float16":
            raise ValueError("Accepted FP16 cache has the wrong dtype metadata")
        expected_dtype = torch.float16
    else:
        if metadata.get("storage_dtype") != "float32_fallback":
            raise ValueError("Failed FP16 fidelity requires FP32 fallback shards")
        expected_dtype = torch.float32

    shards = []
    locations: dict[int, tuple[int, int]] = {}
    labels: dict[int, int] = {}
    for shard_index, item in enumerate(manifest.get("shards", [])):
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"M4 patch shard hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        features = payload["features"]
        if (
            features.ndim != 3
            or tuple(features.shape[1:])
            != (config.model.expected_patch_tokens, config.model.embedding_dim)
            or features.dtype != expected_dtype
            or not torch.isfinite(features).all()
        ):
            raise ValueError(f"Invalid M4 patch shard: {path}")
        if len(features) != len(payload["image_ids"]):
            raise ValueError(f"M4 patch shard fields differ: {path}")
        for row, (image_id, label, split) in enumerate(zip(
            payload["image_ids"], payload["labels"], payload["splits"], strict=True
        )):
            image_id = int(image_id)
            if image_id in locations or split != expected_split:
                raise ValueError(f"Duplicate or non-{expected_split} M4 patch row")
            locations[image_id] = (shard_index, row)
            labels[image_id] = int(label)
        shards.append(payload)
    expected = {
        record.image_id: record.original_label
        for record in records if record.split == expected_split
    }
    if labels != expected:
        raise ValueError("M4 patch-cache IDs or labels differ from CUB development")
    ordered = sorted(locations)
    return PatchTokenCache(
        shards,
        torch.tensor(ordered, dtype=torch.long),
        torch.tensor([labels[value] for value in ordered], dtype=torch.long),
        locations,
        manifest,
    )


def write_patch_manifest(payload: dict[str, Any], path: str | Path) -> None:
    write_json(payload, path)
