#!/usr/bin/env python
"""Evaluate the selected E2A-M4 checkpoint on validation or test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from uncertainty_retrieval.config_e2a import load_e2a_config, save_e2a_config
from uncertainty_retrieval.data.cub import load_cub_records, split_development_records
from uncertainty_retrieval.data.feature_cache import sha256_file, validation_ids_hash
from uncertainty_retrieval.data.patch_token_cache import load_patch_token_cache
from uncertainty_retrieval.evaluation.representation import evaluate_top100, save_ranking_artifact
from uncertainty_retrieval.models.representations import AttentionPatchPooling
from uncertainty_retrieval.training.representation import atomic_torch_save
from uncertainty_retrieval.utils import environment_metadata, resolve_device, write_json


def _forward(model, cache, records, device, batch_size):
    embeddings = []
    weights = []
    ids = [record.image_id for record in records]
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            output = model(cache.batch(batch_ids).to(device).float())
            embeddings.append(output.embedding.cpu())
            weights.append(output.weights.cpu())
    return torch.cat(embeddings), torch.cat(weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_e2a_config(args.config)
    if config.representation.method != "m4":
        raise ValueError("evaluate_e2a_m4.py requires an M4 config")
    records = load_cub_records(
        config.dataset.root, config.dataset.development_classes,
        config.dataset.total_classes,
    )
    fit, validation = split_development_records(
        records, config.dataset.validation_fraction, config.dataset.split_seed
    )
    if args.stage == "validation":
        selected = validation
        selected_ids = torch.tensor([record.image_id for record in selected])
        reference_ids = torch.load(
            config.evaluation.reference_validation_ids, map_location="cpu", weights_only=True
        )
        if not torch.equal(reference_ids, selected_ids):
            raise ValueError("M4 validation IDs differ from accepted M1--M3")
        cache_config = config
    else:
        selected = [record for record in records if record.split == "test"]
        selected_ids = torch.tensor([record.image_id for record in selected])
        test_dir = Path(config.cache.shard_directory).parent / "test"
        cache_config = replace(config, cache=replace(
            config.cache,
            shard_directory=str(test_dir),
            patch_manifest=str(test_dir / "manifest.json"),
        ))
    cache = load_patch_token_cache(cache_config, records, scope=args.stage)
    checkpoint_path = Path(config.output.root) / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {
        "config_sha256": sha256_file(args.config),
        "patch_manifest_sha256": sha256_file(config.cache.patch_manifest),
        "split_hash": validation_ids_hash(validation_ids.tolist()),
    }
    for key, value in expected.items():
        if checkpoint["provenance"].get(key) != value:
            raise ValueError(f"M4 checkpoint provenance mismatch for {key}")
    device = resolve_device()
    model = AttentionPatchPooling(
        config.model.embedding_dim, config.representation.attention_hidden_dim,
        config.model.expected_patch_tokens,
    ).to(device)
    model.load_state_dict(checkpoint["attention"])
    model.eval()
    selected_features, selected_weights = _forward(
        model, cache, selected, device, config.runtime.batch_size
    )
    selected_labels = torch.tensor([record.original_label for record in selected])
    metrics, ranking = evaluate_top100(
        selected_features, selected_ids, selected_labels, device,
        config.runtime.similarity_chunk_size, config.evaluation.ranking_depth,
    )
    ranking["split"] = args.stage
    root = Path(config.output.root)
    save_ranking_artifact(ranking, root / "rankings" / f"{args.stage}_top100.pt")
    write_json(metrics, root / "metrics" / f"{args.stage}.json")
    atomic_torch_save(
        {"schema_version": 2, "features": selected_features,
         "image_ids": selected_ids, "labels": selected_labels, "split": args.stage},
        root / "embeddings" / f"{args.stage}.pt",
    )
    if args.stage == "validation":
        fit_features, _ = _forward(model, cache, fit, device, config.runtime.batch_size)
        atomic_torch_save(
            {"schema_version": 2, "features": fit_features,
             "image_ids": torch.tensor([r.image_id for r in fit]),
             "labels": torch.tensor([r.original_label for r in fit]), "split": "fit"},
            root / "embeddings" / "fit.pt",
        )
    entropy = -(selected_weights * selected_weights.clamp_min(1e-12).log()).sum(dim=1)
    atomic_torch_save(
        {"image_ids": selected_ids, "weights": selected_weights.half(),
         "diagnostic_only": True}, root / "attention" / f"{args.stage}_weights.pt"
    )
    atomic_torch_save(
        {"image_ids": selected_ids, "entropy": entropy.float(),
         "diagnostic_only": True}, root / "attention" / f"{args.stage}_entropy.pt"
    )
    write_json(
        {
            "schema_version": 2, "method": "m4", "representation": "attention_pool",
            "normalization": "l2_at_retrieval", "shape": list(selected_features.shape),
            "dtype": str(selected_features.dtype),
            "patch_manifest_sha256": sha256_file(config.cache.patch_manifest),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "selected_epoch": checkpoint["epoch"],
            "total_trainable_parameters": checkpoint["provenance"]["total_trainable_parameters"],
            "deployable_parameters": checkpoint["provenance"]["deployable_parameters"],
            "attention_interpretation": "diagnostic_only_not_localization_evidence",
        }, root / "embeddings" / "manifest.json"
    )
    save_e2a_config(config, root / "config_resolved.yaml")
    environment_path = root / "environment.json"
    old = json.loads(environment_path.read_text(encoding="utf-8"))
    training_environment = old.get("training", old)
    evaluations = dict(old.get("evaluations", {}))
    if "evaluation" in old and "validation" not in evaluations:
        evaluations["validation"] = old["evaluation"]
    evaluations[args.stage] = environment_metadata(sys.argv)
    write_json(
        {"training": training_environment, "evaluations": evaluations},
        environment_path,
    )
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    previous_evaluation = metadata.pop("evaluation", None)
    evaluations = dict(metadata.get("evaluations", {}))
    if previous_evaluation is not None and "validation" not in evaluations:
        evaluations["validation"] = previous_evaluation
    evaluations[args.stage] = {
        "stage": args.stage, "query_count": len(selected),
        "wall_time_seconds": time.perf_counter() - started,
        "trainable_parameters": checkpoint["provenance"]["total_trainable_parameters"],
        "deployable_parameters": checkpoint["provenance"]["deployable_parameters"],
    }
    metadata["evaluations"] = evaluations
    write_json(metadata, metadata_path)


if __name__ == "__main__":
    main()
