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
    find_reusable_feature_cache,
    sha256_file,
    validation_ids_hash,
    write_embedding_manifest,
)
from uncertainty_retrieval.data.fused_cache import load_fused_feature_cache
from uncertainty_retrieval.data.patch_cache import load_feature_cache
from uncertainty_retrieval.evaluation.representation import (
    evaluate_top100,
    save_ranking_artifact,
    verify_selection_lock,
)
from uncertainty_retrieval.models.representations import CLSMeanPatchProjection
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
            raise PermissionError(
                f"{config.representation.method.upper()} is not the locked E2A winner"
            )

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
    fit, validation = split_development_records(
        records, config.dataset.validation_fraction, config.dataset.split_seed
    )
    if config.representation.method in {"m2", "m3"}:
        reference_path = Path(config.evaluation.reference_validation_ids)
        if not reference_path.is_file():
            raise FileNotFoundError(
                "Accepted M1 validation IDs are required for E2A evaluation"
            )
        reference_ids = torch.load(
            reference_path, map_location="cpu", weights_only=True
        )
        current_ids = torch.tensor(
            [record.image_id for record in validation], dtype=torch.long
        )
        if not torch.equal(reference_ids, current_ids):
            raise ValueError(
                f"{config.representation.method.upper()} validation IDs "
                "differ from accepted M1"
            )
    selected = validation if args.stage == "validation" else [
        record for record in records if record.split == "test"
    ]
    image_ids = torch.tensor([record.image_id for record in selected], dtype=torch.long)
    labels = torch.tensor([record.original_label for record in selected], dtype=torch.long)
    device = resolve_device()
    cache_path = None
    checkpoint_path = None
    diagnostics: dict[str, str]
    if config.representation.method == "m3":
        fused = load_fused_feature_cache(config)
        by_id = {int(value): index for index, value in enumerate(fused.image_ids)}
        positions = torch.tensor([by_id[record.image_id] for record in selected])
        checkpoint_path = Path(config.output.root) / "checkpoints" / "best.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        expected_provenance = {
            "config_sha256": sha256_file(args.config),
            "cls_cache_sha256": fused.cls_sha256,
            "mean_patch_cache_sha256": fused.mean_patch_sha256,
            "split_hash": validation_ids_hash(
                [record.image_id for record in validation]
            ),
        }
        for key, expected in expected_provenance.items():
            if checkpoint["provenance"].get(key) != expected:
                raise ValueError(f"M3 checkpoint provenance mismatch for {key}")
        projection = CLSMeanPatchProjection(config.model.embedding_dim).to(device)
        projection.load_state_dict(checkpoint["projection"])
        projection.eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(positions), config.runtime.batch_size):
                index = positions[start : start + config.runtime.batch_size]
                chunks.append(
                    projection(
                        fused.cls[index].to(device, non_blocking=True),
                        fused.mean_patch[index].to(device, non_blocking=True),
                    ).cpu()
                )
        features = torch.cat(chunks)
        diagnostics = {
            str(config.cache.cls_path): "accepted",
            str(config.cache.mean_patch_path): "accepted",
            str(checkpoint_path): "accepted",
        }
    else:
        cache_path, diagnostics = find_reusable_feature_cache(config, records)
        if cache_path is None:
            raise FileNotFoundError(
                f"No compatible {config.representation.method.upper()} cache: "
                f"{diagnostics}"
            )
        payload = load_feature_cache(cache_path)
        by_id = {
            int(value): index for index, value in enumerate(payload["image_ids"])
        }
        positions = torch.tensor([by_id[record.image_id] for record in selected])
        features = payload["features"][positions]
    metrics, ranking = evaluate_top100(
        features,
        image_ids,
        labels,
        device,
        config.runtime.similarity_chunk_size,
        config.evaluation.ranking_depth,
    )
    ranking["split"] = args.stage

    root = Path(config.output.root)
    save_e2a_config(config, root / "config_resolved.yaml")
    environment_path = root / "environment.json"
    evaluation_environment = environment_metadata(sys.argv)
    if config.representation.method == "m3" and environment_path.is_file():
        training_environment = json.loads(
            environment_path.read_text(encoding="utf-8")
        )
        write_json(
            {
                "training": training_environment,
                "evaluation": evaluation_environment,
            },
            environment_path,
        )
    else:
        write_json(evaluation_environment, environment_path)
    save_ranking_artifact(ranking, root / "rankings" / f"{args.stage}_top100.pt")
    write_json(metrics, root / "metrics" / f"{args.stage}.json")
    if config.representation.method == "m3":
        embedding_payload = {
            "schema_version": config.cache.schema_version,
            "features": features,
            "image_ids": image_ids,
            "labels": labels,
            "split": args.stage,
        }
        embedding_path = root / "embeddings" / f"{args.stage}.pt"
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding_payload, embedding_path)
        if args.stage == "validation":
            fit_positions = torch.tensor(
                [by_id[record.image_id] for record in fit], dtype=torch.long
            )
            fit_chunks = []
            with torch.inference_mode():
                for start in range(0, len(fit_positions), config.runtime.batch_size):
                    index = fit_positions[
                        start : start + config.runtime.batch_size
                    ]
                    fit_chunks.append(
                        projection(
                            fused.cls[index].to(device, non_blocking=True),
                            fused.mean_patch[index].to(device, non_blocking=True),
                        ).cpu()
                    )
            torch.save(
                {
                    "schema_version": config.cache.schema_version,
                    "features": torch.cat(fit_chunks),
                    "image_ids": torch.tensor(
                        [record.image_id for record in fit], dtype=torch.long
                    ),
                    "labels": torch.tensor(
                        [record.original_label for record in fit],
                        dtype=torch.long,
                    ),
                    "split": "fit",
                },
                root / "embeddings" / "fit.pt",
            )
        write_json(
            {
                "schema_version": config.cache.schema_version,
                "method": "m3",
                "representation": "cls_mean_projection",
                "normalization": "l2_at_retrieval",
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "cls_cache_sha256": fused.cls_sha256,
                "mean_patch_cache_sha256": fused.mean_patch_sha256,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "concatenation_order": ["cls", "mean_patch"],
                "selected_epoch": checkpoint["epoch"],
                "total_trainable_parameters": checkpoint["provenance"][
                    "total_trainable_parameters"
                ],
                "deployable_parameters": checkpoint["provenance"][
                    "deployable_parameters"
                ],
            },
            root / "embeddings" / "manifest.json",
        )
    else:
        write_embedding_manifest(
            cache_path, config, root / "embeddings" / "manifest.json"
        )
    metadata = {
        "method": config.representation.method,
        "stage": args.stage,
        "query_count": len(selected),
        "trainable_parameters": 0,
        "cache_diagnostics": diagnostics,
        "cub_manifest_hash": cub_manifest_hash(config.dataset.root),
        "wall_time_seconds": time.perf_counter() - started,
    }
    metadata_path = root / "metadata.json"
    if config.representation.method == "m3" and metadata_path.is_file():
        training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = {**training_metadata, "evaluation": metadata}
    write_json(metadata, metadata_path)
    if args.stage == "validation":
        split_dir = root / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(image_ids, split_dir / "validation_image_ids.pt")
        (split_dir / "manifest_hash.txt").write_text(
            validation_ids_hash(image_ids.tolist()) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
