#!/usr/bin/env python
"""Extract provenance-complete frozen DINOv3 features for E2A."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from uncertainty_retrieval.config_e2a import load_e2a_config
from uncertainty_retrieval.data.cub import CUBDataset, load_cub_records, validate_protocol_counts
from uncertainty_retrieval.data.feature_cache import cub_manifest_hash
from uncertainty_retrieval.data.patch_cache import load_feature_cache, save_feature_cache
from uncertainty_retrieval.models.dinov3 import DINOv3Backbone, processor_transform
from uncertainty_retrieval.models.representations import (
    CLSRepresentation,
    MeanPatchRepresentation,
)
from uncertainty_retrieval.utils import distributed_barrier, initialize_distributed, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def _processor_settings(processor: object) -> dict:
    if not hasattr(processor, "to_dict"):
        raise TypeError("DINOv3 processor must expose to_dict() for provenance")
    settings = processor.to_dict()
    if not isinstance(settings, dict):
        raise TypeError("Processor settings must be a mapping")
    return settings


def _build_representation(config):
    if config.representation.method == "m1":
        return CLSRepresentation(config.model.embedding_dim)
    if config.representation.method == "m2":
        return MeanPatchRepresentation(
            config.model.embedding_dim,
            config.model.expected_patch_tokens,
        )
    raise ValueError(f"Unsupported E2A method: {config.representation.method}")


def main() -> None:
    args = parse_args()
    config = load_e2a_config(args.config)
    rank, world_size, _, device = initialize_distributed()
    if world_size != config.runtime.world_size:
        raise RuntimeError(
            f"Expected {config.runtime.world_size} processes, got {world_size}"
        )
    seed_everything(config.dataset.split_seed)
    records = load_cub_records(
        config.dataset.root,
        config.dataset.development_classes,
        config.dataset.total_classes,
    )
    validate_protocol_counts(
        records,
        config.dataset.expected_development_images,
        config.dataset.expected_test_images,
    )
    backbone, processor = DINOv3Backbone.from_pretrained(
        config.model.model_id,
        config.model.revision,
        config.model.register_tokens,
    )
    backbone.to(device)
    representation = _build_representation(config).to(device)
    dataset = CUBDataset(config.dataset.root, records, processor_transform(processor))
    sampler = DistributedSampler(dataset, shuffle=False)
    loader_kwargs = {
        "batch_size": config.runtime.batch_size,
        "sampler": sampler,
        "num_workers": config.runtime.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.runtime.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=config.runtime.prefetch_factor,
        )
    loader = DataLoader(dataset, **loader_kwargs)

    features: list[torch.Tensor] = []
    image_ids: list[int] = []
    labels: list[int] = []
    splits: list[str] = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            enabled=config.runtime.amp and device.type == "cuda",
            dtype=torch.float16,
        ):
            tokens = backbone(pixels)
        # Representation aggregation is FP32 even when backbone AMP is active.
        embedding = representation(tokens)
        features.append(embedding.cpu())
        image_ids.extend(batch["image_id"].tolist())
        labels.extend(batch["label"].tolist())
        splits.extend(batch["split"])

    cache_path = Path(config.cache.path)
    shard_path = cache_path.with_suffix(f".rank{rank}.pt")
    resolved = getattr(backbone.model.config, "_commit_hash", None)
    if resolved not in (None, config.model.revision):
        raise RuntimeError(f"Resolved checkpoint revision drifted: {resolved}")
    metadata = {
        "model_id": config.model.model_id,
        "requested_revision": config.model.revision,
        "resolved_revision": resolved or config.model.revision,
        "token": "cls" if config.representation.method == "m1" else "patch",
        "source_layer": "final",
        "register_tokens": config.model.register_tokens,
        "processor_settings": _processor_settings(processor),
        "cub_manifest_hash": cub_manifest_hash(config.dataset.root),
    }
    if config.representation.method == "m2":
        metadata.update(
            pooling="mean",
            patch_tokens=config.model.expected_patch_tokens,
            embedding_dim=config.model.embedding_dim,
        )
    save_feature_cache(
        {
            "schema_version": config.cache.schema_version,
            "features": torch.cat(features),
            "image_ids": image_ids,
            "labels": labels,
            "splits": splits,
            "metadata": metadata,
        },
        shard_path,
    )
    distributed_barrier(device)
    if rank == 0:
        shards = [
            load_feature_cache(cache_path.with_suffix(f".rank{index}.pt"))
            for index in range(world_size)
        ]
        rows = []
        for shard in shards:
            rows.extend(zip(
                shard["image_ids"], shard["labels"], shard["splits"],
                shard["features"], strict=True,
            ))
        unique = {int(row[0]): row for row in rows}
        ordered = [unique[key] for key in sorted(unique)]
        save_feature_cache(
            {
                "schema_version": config.cache.schema_version,
                "features": torch.stack([row[3] for row in ordered]),
                "image_ids": [int(row[0]) for row in ordered],
                "labels": [int(row[1]) for row in ordered],
                "splits": [row[2] for row in ordered],
                "metadata": metadata,
            },
            cache_path,
        )
        for index in range(world_size):
            cache_path.with_suffix(f".rank{index}.pt").unlink()
    distributed_barrier(device)


if __name__ == "__main__":
    main()
