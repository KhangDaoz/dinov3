#!/usr/bin/env python
"""Select the E2A winner from completed M1--M4 test artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from uncertainty_retrieval.config_e2a import load_e2a_config
from uncertainty_retrieval.data.feature_cache import sha256_file
from uncertainty_retrieval.evaluation.representation import (
    METHOD_ORDER,
    hits_from_ranking,
    select_representation_winner,
    validate_recall_against_hits,
)
from uncertainty_retrieval.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs=4, required=True, type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/e2a_selection/test_selection.json"),
    )
    args = parser.parse_args()
    configs = [load_e2a_config(path) for path in args.configs]
    by_method = {config.representation.method: (path, config)
                 for path, config in zip(args.configs, configs, strict=True)}
    if set(by_method) != set(METHOD_ORDER):
        raise ValueError("Exactly one config for each of M1--M4 is required")

    hits = {}
    metrics = {}
    parameters = {}
    artifact_hashes = {}
    reference_ids = reference_labels = None
    for method in METHOD_ORDER:
        config_path, config = by_method[method]
        root = Path(config.output.root)
        ranking_path = root / "rankings" / "test_top100.pt"
        metrics_path = root / "metrics" / "test.json"
        ranking = torch.load(ranking_path, map_location="cpu", weights_only=True)
        if ranking.get("split") != "test" or ranking.get("ranking_depth") != 100:
            raise ValueError(f"{method.upper()} is not a valid Top-100 test artifact")
        ids, labels = ranking["query_image_ids"], ranking["query_labels"]
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        elif not torch.equal(ids, reference_ids) or not torch.equal(labels, reference_labels):
            raise ValueError("M1--M4 test queries or labels are not identical")
        hits[method] = hits_from_ranking(ranking)
        metrics[method] = json.loads(metrics_path.read_text(encoding="utf-8"))
        for k in (1, 2, 4, 8):
            try:
                validate_recall_against_hits(
                    metrics[method][f"recall_at_{k}"],
                    hits[method][f"hits_at_{k}"],
                    len(ids),
                )
            except ValueError as error:
                raise ValueError(f"{method.upper()} Recall@{k}: {error}") from error
        if method in {"m1", "m2"}:
            parameters[method] = 0
        else:
            manifest = json.loads(
                (root / "embeddings" / "manifest.json").read_text(encoding="utf-8")
            )
            parameters[method] = int(manifest["total_trainable_parameters"])
        artifact_hashes[method] = {
            "config": sha256_file(config_path),
            "ranking": sha256_file(ranking_path),
            "metrics": sha256_file(metrics_path),
        }

    winner, trace = select_representation_winner(hits, parameters)
    write_json(
        {
            "schema_version": 1,
            "selection_split": "test_classes_100_199",
            "selection_bias_notice": (
                "This test split was used for method selection and is not an "
                "untouched estimate of final generalization."
            ),
            "completed_methods": list(METHOD_ORDER),
            "query_count": len(reference_ids),
            "winner": winner,
            "test_hits": hits,
            "test_metrics": metrics,
            "optimized_parameters": parameters,
            "tie_rule": ["hits_at_1", "hits_at_2", "hits_at_4", "hits_at_8",
                         "optimized_parameters", "fixed_m1_m4_order"],
            "tie_trace": trace,
            "artifact_hashes": artifact_hashes,
        },
        args.output,
    )
    print(f"E2A test-selected winner: {winner.upper()}")
    print(f"Selection artifact: {args.output}")


if __name__ == "__main__":
    main()
