#!/usr/bin/env python
"""Extract development-only final patch tokens for E2A-M4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from uncertainty_retrieval.config_e2a import load_e2a_config
from uncertainty_retrieval.data.cub import CUBDataset, load_cub_records, validate_protocol_counts
from uncertainty_retrieval.data.feature_cache import cub_manifest_hash, sha256_file
from uncertainty_retrieval.data.patch_cache import save_feature_cache
from uncertainty_retrieval.data.patch_token_cache import (
    fp16_fidelity_statistics,
    validate_fidelity,
    write_patch_manifest,
)
from uncertainty_retrieval.models.dinov3 import DINOv3Backbone, processor_transform
from uncertainty_retrieval.utils import (
    distributed_barrier, environment_metadata, initialize_distributed, seed_everything,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def _fidelity_ids(records, count: int) -> set[int]:
    by_class = {}
    for record in records:
        by_class.setdefault(record.original_label, []).append(record.image_id)
    selected = []
    offset = 0
    while len(selected) < count:
        for label in sorted(by_class):
            rows = sorted(by_class[label])
            if offset < len(rows):
                selected.append(rows[offset])
                if len(selected) == count:
                    break
        offset += 1
    return set(selected)


def main() -> None:
    args = _args()
    config = load_e2a_config(args.config)
    if config.representation.method != "m4":
        raise ValueError("Patch-token extraction requires an M4 config")
    rank, world_size, _, device = initialize_distributed()
    if world_size != config.runtime.world_size:
        raise RuntimeError(f"Expected {config.runtime.world_size} ranks")
    seed_everything(config.training.seed)
    records = load_cub_records(
        config.dataset.root, config.dataset.development_classes,
        config.dataset.total_classes,
    )
    validate_protocol_counts(
        records, config.dataset.expected_development_images,
        config.dataset.expected_test_images,
    )
    development = [record for record in records if record.split == "development"]
    local_records = development[rank::world_size]
    fidelity_ids = _fidelity_ids(development, config.cache.fidelity_samples)
    backbone, processor = DINOv3Backbone.from_pretrained(
        config.model.model_id, config.model.revision, config.model.register_tokens
    )
    backbone.to(device)
    dataset = CUBDataset(
        config.dataset.root, local_records, processor_transform(processor)
    )
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.runtime.num_workers > 0,
        prefetch_factor=(config.runtime.prefetch_factor if config.runtime.num_workers else None),
    )
    features = []
    image_ids = []
    labels = []
    fidelity_reference = []
    fidelity_local_ids = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            enabled=bool(config.cache.extraction_amp and device.type == "cuda"),
            dtype=torch.float16,
        ):
            tokens = backbone(pixels)
        patches = tokens.patches.float()
        expected = (config.model.expected_patch_tokens, config.model.embedding_dim)
        if tuple(patches.shape[1:]) != expected:
            raise ValueError(f"Unexpected DINOv3 patch shape: {tuple(patches.shape)}")
        ids = batch["image_id"].tolist()
        for index, image_id in enumerate(ids):
            if image_id in fidelity_ids:
                fidelity_reference.append(patches[index].cpu())
                fidelity_local_ids.append(int(image_id))
        features.append(patches.cpu())
        image_ids.extend(ids)
        labels.extend(batch["label"].tolist())
    fidelity = fp16_fidelity_statistics(torch.stack(fidelity_reference))
    local_failed = int(
        fidelity["max_relative_l2"] > config.cache.fidelity_relative_l2_max
        or fidelity["min_patch_cosine"] < config.cache.fidelity_patch_cosine_min
        or fidelity["min_mean_cosine"] < config.cache.fidelity_mean_cosine_min
    )
    failure_flag = torch.tensor(local_failed, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(failure_flag, op=torch.distributed.ReduceOp.MAX)
    accepted_fp16 = not bool(failure_flag.item())
    stored_features = torch.cat(features)
    storage_dtype = "float16" if accepted_fp16 else "float32_fallback"
    if accepted_fp16:
        stored_features = stored_features.half()
    shard_directory = Path(config.cache.shard_directory)
    shard_path = shard_directory / f"rank{rank}.pt"
    resolved = getattr(backbone.model.config, "_commit_hash", None)
    processor_settings = processor.to_dict()
    metadata = {
        "model_id": config.model.model_id,
        "requested_revision": config.model.revision,
        "resolved_revision": resolved or config.model.revision,
        "register_tokens": config.model.register_tokens,
        "processor_settings": processor_settings,
        "cub_manifest_hash": cub_manifest_hash(config.dataset.root),
        "token": "patch",
        "source_layer": "final",
        "patch_tokens": config.model.expected_patch_tokens,
        "embedding_dim": config.model.embedding_dim,
        "storage_dtype": storage_dtype,
        "scope": "development_only",
        "fidelity_ids": fidelity_local_ids,
        "fidelity_statistics": fidelity,
    }
    save_feature_cache(
        {
            "schema_version": config.cache.schema_version,
            "features": stored_features,
            "image_ids": image_ids,
            "labels": labels,
            "splits": ["development"] * len(image_ids),
            "metadata": metadata,
        },
        shard_path,
    )
    distributed_barrier(device)
    if rank == 0:
        shard_items = []
        all_ids = []
        statistics = []
        fidelity_all_ids = []
        common_metadata = None
        for index in range(world_size):
            path = shard_directory / f"rank{index}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            item_metadata = payload["metadata"]
            statistics.append(item_metadata["fidelity_statistics"])
            fidelity_all_ids.extend(item_metadata["fidelity_ids"])
            all_ids.extend(int(value) for value in payload["image_ids"])
            if common_metadata is None:
                common_metadata = {
                    key: value for key, value in item_metadata.items()
                    if not key.startswith("fidelity_")
                }
            shard_items.append(
                {"path": str(path), "sha256": sha256_file(path), "rows": len(payload["image_ids"])}
            )
        combined = {
            "max_relative_l2": max(item["max_relative_l2"] for item in statistics),
            "min_patch_cosine": min(item["min_patch_cosine"] for item in statistics),
            "min_mean_cosine": min(item["min_mean_cosine"] for item in statistics),
        }
        if len(all_ids) != len(set(all_ids)) or set(all_ids) != {
            record.image_id for record in development
        }:
            raise ValueError("M4 patch shards do not cover development exactly")
        write_patch_manifest(
            {
                "schema_version": config.cache.schema_version,
                "metadata": common_metadata,
                "shards": shard_items,
                "fidelity": {
                    "accepted_fp16": accepted_fp16,
                    "sample_ids": sorted(fidelity_all_ids),
                    "statistics": combined,
                    "thresholds": {
                        "max_relative_l2": config.cache.fidelity_relative_l2_max,
                        "min_patch_cosine": config.cache.fidelity_patch_cosine_min,
                        "min_mean_cosine": config.cache.fidelity_mean_cosine_min,
                    },
                },
                "environment": environment_metadata(sys.argv),
            },
            config.cache.patch_manifest,
        )
    distributed_barrier(device)


if __name__ == "__main__":
    main()
