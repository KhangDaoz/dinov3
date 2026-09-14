"""E2A representation evaluation, winner selection, and test isolation."""

from __future__ import annotations

import json
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .retrieval import cosine_rankings, recall_at_k


METHOD_ORDER = ("m1", "m2", "m3", "m4")
METRIC_ORDER = ("recall_at_1", "recall_at_2", "recall_at_4", "recall_at_8")
HIT_ORDER = ("hits_at_1", "hits_at_2", "hits_at_4", "hits_at_8")


def evaluate_top100(
    features: Tensor,
    image_ids: Tensor,
    labels: Tensor,
    device: torch.device,
    chunk_size: int,
    ranking_depth: int = 100,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate cosine retrieval and retain candidate IDs/scores only."""
    if not (len(features) == len(image_ids) == len(labels)):
        raise ValueError("Evaluation fields have inconsistent lengths")
    if len(torch.unique(image_ids)) != len(image_ids):
        raise ValueError("Evaluation image IDs must be unique")
    if ranking_depth <= 0 or ranking_depth >= len(features):
        raise ValueError("ranking_depth must be in [1, number of items)")
    result = cosine_rankings(
        features.to(device),
        query_ids=image_ids.to(device),
        gallery_ids=image_ids.to(device),
        chunk_size=chunk_size,
        ranking_depth=ranking_depth,
    )
    indices = result.indices
    scores = result.scores
    recalls = recall_at_k(indices, labels, labels, (1, 2, 4, 8))
    metrics = {f"recall_at_{key}": value for key, value in recalls.items()}
    artifact = {
        "schema_version": 1,
        "query_image_ids": image_ids.cpu(),
        "query_labels": labels.cpu(),
        "candidate_image_ids": image_ids[indices].cpu(),
        "cosine_scores": scores.cpu(),
        "ranking_depth": ranking_depth,
        "self_match_exclusion": True,
        "tie_policy": "stable_gallery_index",
    }
    return metrics, artifact


def select_representation_winner(
    validation_hits: dict[str, dict[str, int]],
    trainable_parameters: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply the preregistered integer Hits@K lexicographic rule."""
    if set(validation_hits) != set(METHOD_ORDER):
        raise ValueError("Winner selection requires validation for M1--M4")
    if set(trainable_parameters) != set(METHOD_ORDER):
        raise ValueError("Parameter counts are required for M1--M4")
    for method, hits in validation_hits.items():
        missing = set(HIT_ORDER) - hits.keys()
        if missing:
            raise ValueError(f"{method} hit counts missing: {sorted(missing)}")
        if any(not isinstance(hits[key], int) or hits[key] < 0 for key in HIT_ORDER):
            raise ValueError(f"{method} hit counts must be non-negative integers")

    trace: list[dict[str, Any]] = []

    def compare(left: str, right: str) -> int:
        for metric in HIT_ORDER:
            delta = validation_hits[left][metric] - validation_hits[right][metric]
            if delta:
                preferred = left if delta > 0 else right
                trace.append(
                    {
                        "left": left,
                        "right": right,
                        "criterion": metric,
                        "delta": delta,
                        "preferred": preferred,
                    }
                )
                return -1 if preferred == left else 1
        parameter_delta = trainable_parameters[left] - trainable_parameters[right]
        if parameter_delta:
            preferred = left if parameter_delta < 0 else right
            trace.append(
                {
                    "left": left,
                    "right": right,
                    "criterion": "trainable_parameters",
                    "delta": parameter_delta,
                    "preferred": preferred,
                }
            )
            return -1 if preferred == left else 1
        return -1 if METHOD_ORDER.index(left) < METHOD_ORDER.index(right) else 1

    winner = sorted(METHOD_ORDER, key=cmp_to_key(compare))[0]
    return winner, trace


def verify_selection_lock(path: str | Path) -> dict[str, Any]:
    """Fail closed unless a complete immutable E2A lock opens final test."""
    lock_path = Path(path)
    if not lock_path.is_file():
        raise PermissionError(
            "Final test is locked until M1--M4 validation is complete"
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "test_unlocked",
        "completed_methods",
        "winner",
        "validation_metrics",
        "config_hashes",
        "cache_hash",
        "split_hash",
        "git_commit",
        "tie_rule",
    }
    missing = required - lock.keys()
    if missing:
        raise PermissionError(f"Selection lock is incomplete: {sorted(missing)}")
    if lock["test_unlocked"] is not True:
        raise PermissionError("Selection lock does not authorize final test")
    if tuple(lock["completed_methods"]) != METHOD_ORDER:
        raise PermissionError("Selection lock does not contain M1--M4")
    if lock["winner"] not in METHOD_ORDER:
        raise PermissionError("Selection lock winner is invalid")
    if set(lock["validation_metrics"]) != set(METHOD_ORDER):
        raise PermissionError("Selection lock validation metrics are incomplete")
    if set(lock["config_hashes"]) != set(METHOD_ORDER):
        raise PermissionError("Selection lock config hashes are incomplete")
    return lock


def save_ranking_artifact(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
