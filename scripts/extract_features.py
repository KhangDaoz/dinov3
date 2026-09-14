#!/usr/bin/env python
"""Extract frozen DINOv3 CLS features into a portable cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from uncertainty_retrieval.config import load_config
from uncertainty_retrieval.data.cub import (
    CUBDataset,
    load_cub_records,
    validate_protocol_counts,
)
from uncertainty_retrieval.data.patch_cache import save_feature_cache
from uncertainty_retrieval.models.dinov3 import (
    DINOv3Backbone,
    processor_transform,
)
from uncertainty_retrieval.utils import (
    distributed_barrier,
    initialize_distributed,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rank, world_size, _, device = initialize_distributed()
    seed_everything(config.training.seed, config.training.deterministic)
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
    dataset = CUBDataset(
        config.dataset.root,
        records,
        processor_transform(processor),
    )
    sampler = (
        DistributedSampler(dataset, shuffle=False)
        if world_size > 1
        else None
    )
    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": config.training.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.training.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=config.training.prefetch_factor,
        )
    loader = DataLoader(dataset, **loader_kwargs)

    features = []
    image_ids = []
    labels = []
    splits = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            enabled=config.training.amp and device.type == "cuda",
            dtype=torch.float16,
        ):
            tokens = backbone(pixels)
        features.append(tokens.cls.float().cpu())
        image_ids.extend(batch["image_id"].tolist())
        labels.extend(batch["label"].tolist())
        splits.extend(batch["split"])

    cache_path = Path(config.output.feature_cache)
    shard_path = cache_path.with_suffix(f".rank{rank}.pt")
    save_feature_cache(
        {
            "schema_version": config.output.schema_version,
            "features": torch.cat(features),
            "image_ids": image_ids,
            "labels": labels,
            "splits": splits,
            "metadata": {
                "model_id": config.model.model_id,
                "requested_revision": config.model.revision,
                "resolved_revision": getattr(
                    backbone.model.config,
                    "_commit_hash",
                    None,
                ),
                "token": "cls",
                "register_tokens": config.model.register_tokens,
            },
        },
        shard_path,
    )
    distributed_barrier(device)
    if rank == 0:
        from uncertainty_retrieval.data.patch_cache import load_feature_cache

        shards = [
            load_feature_cache(cache_path.with_suffix(f".rank{index}.pt"))
            for index in range(world_size)
        ]
        rows = []
        for shard in shards:
            rows.extend(
                zip(
                    shard["image_ids"],
                    shard["labels"],
                    shard["splits"],
                    shard["features"],
                    strict=True,
                )
            )
        # DistributedSampler may pad; image ID is the canonical unique key.
        unique = {int(row[0]): row for row in rows}
        ordered = [unique[key] for key in sorted(unique)]
        save_feature_cache(
            {
                "schema_version": config.output.schema_version,
                "features": torch.stack([row[3] for row in ordered]),
                "image_ids": [int(row[0]) for row in ordered],
                "labels": [int(row[1]) for row in ordered],
                "splits": [row[2] for row in ordered],
                "metadata": shards[0]["metadata"],
            },
            cache_path,
        )
        for index in range(world_size):
            cache_path.with_suffix(f".rank{index}.pt").unlink()
    distributed_barrier(device)


if __name__ == "__main__":
    main()
