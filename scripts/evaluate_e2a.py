#!/usr/bin/env python
"""Evaluate one E2A representation with final-test isolation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from uncertainty_retrieval.config_e2a import load_e2a_config, save_e2a_config
from uncertainty_retrieval.data.cub import load_cub_records, split_development_records, validate_protocol_counts
from uncertainty_retrieval.data.feature_cache import (
    cub_manifest_hash,
    find_reusable_m1_cache,
    validation_ids_hash,
    write_embedding_manifest,
)
from uncertainty_retrieval.data.patch_cache import load_feature_cache
from uncertainty_retrieval.evaluation.representation import (
    evaluate_top100,
    save_ranking_artifact,
    verify_selection_lock,
)
from uncertainty_retrieval.utils import environment_metadata, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("validation", "test"), default="validation")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    config = load_e2a_config(args.config)
    if args.stage == "test":
        lock = verify_selection_lock(config.output.selection_lock)
        if lock["winner"] != config.representation.method:
            raise PermissionError("M1 is not the locked E2A winner")

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
    cache_path, diagnostics = find_reusable_m1_cache(config, records)
    if cache_path is None:
        raise FileNotFoundError(f"No compatible M1 cache: {diagnostics}")
    payload = load_feature_cache(cache_path)
    by_id = {int(value): index for index, value in enumerate(payload["image_ids"])}
    _, validation = split_development_records(
        records, config.dataset.validation_fraction, config.dataset.split_seed
    )
    selected = validation if args.stage == "validation" else [
        record for record in records if record.split == "test"
    ]
    positions = torch.tensor([by_id[record.image_id] for record in selected])
    features = payload["features"][positions]
    image_ids = torch.tensor([record.image_id for record in selected], dtype=torch.long)
    labels = torch.tensor([record.original_label for record in selected], dtype=torch.long)
    metrics, ranking = evaluate_top100(
        features,
        image_ids,
        labels,
        resolve_device(),
        config.runtime.similarity_chunk_size,
        config.evaluation.ranking_depth,
    )
    ranking["split"] = args.stage

    root = Path(config.output.root)
    save_e2a_config(config, root / "config_resolved.yaml")
    write_json(environment_metadata(sys.argv), root / "environment.json")
    save_ranking_artifact(ranking, root / "rankings" / f"{args.stage}_top100.pt")
    write_json(metrics, root / "metrics" / f"{args.stage}.json")
    write_embedding_manifest(cache_path, config, root / "embeddings" / "manifest.json")
    metadata = {
        "method": "m1",
        "stage": args.stage,
        "query_count": len(selected),
        "trainable_parameters": 0,
        "cache_diagnostics": diagnostics,
        "cub_manifest_hash": cub_manifest_hash(config.dataset.root),
        "wall_time_seconds": time.perf_counter() - started,
    }
    write_json(metadata, root / "metadata.json")
    if args.stage == "validation":
        split_dir = root / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(image_ids, split_dir / "validation_image_ids.pt")
        (split_dir / "manifest_hash.txt").write_text(
            validation_ids_hash(image_ids.tolist()) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
