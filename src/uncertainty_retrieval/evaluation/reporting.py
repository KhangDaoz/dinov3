"""E1 result selection and JSON-ready evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from .evidential import (
    aurc,
    binary_auprc,
    binary_auroc,
    hits_at_k,
    paired_bootstrap_recall_delta,
)
from .retrieval import (
    certainty_fusion_rerank,
    recall_at_k,
    uncertainty_rerank,
)


def evaluate_ranking(
    rankings: Tensor,
    query_labels: Tensor,
    gallery_labels: Tensor,
    k_values: Iterable[int],
) -> dict[str, float]:
    return {
        f"recall_at_{k}": value
        for k, value in recall_at_k(
            rankings,
            query_labels,
            gallery_labels,
            k_values,
        ).items()
    }


def select_r1_top_n(
    rankings: Tensor,
    uncertainty: Tensor,
    labels: Tensor,
    top_n_grid: Iterable[int],
) -> tuple[int, dict[int, dict[str, float]]]:
    """Select top-N by validation Recall@1, then smaller N."""
    candidates: dict[int, dict[str, float]] = {}
    for top_n in top_n_grid:
        reranked = uncertainty_rerank(rankings, uncertainty, top_n)
        candidates[top_n] = evaluate_ranking(reranked, labels, labels, (1, 2, 4, 8))
    best = max(candidates, key=lambda n: (candidates[n]["recall_at_1"], -n))
    return best, candidates


def select_r2_parameters(
    rankings: Tensor,
    ranked_scores: Tensor,
    uncertainty: Tensor,
    labels: Tensor,
    top_n_grid: Iterable[int],
    beta_grid: Iterable[float],
) -> tuple[tuple[int, float], dict[str, dict[str, float]]]:
    candidates: dict[str, dict[str, float]] = {}
    keys = []
    for top_n in top_n_grid:
        for beta in beta_grid:
            reranked = certainty_fusion_rerank(
                rankings,
                ranked_scores,
                uncertainty,
                top_n,
                beta,
            )
            key = f"n={top_n},beta={beta:g}"
            candidates[key] = evaluate_ranking(
                reranked,
                labels,
                labels,
                (1, 2, 4, 8),
            )
            keys.append((top_n, beta, key))
    best_n, best_beta, _ = max(
        keys,
        key=lambda item: (
            candidates[item[2]]["recall_at_1"],
            -item[0],
            -item[1],
        ),
    )
    return (best_n, best_beta), candidates


def final_e1_statistics(
    baseline: Tensor,
    proposed: Tensor,
    labels: Tensor,
    candidate_uncertainty: Tensor,
    k_values: Iterable[int],
    bootstrap_samples: int,
    seed: int,
    maximum_secondary_regression: float = 0.005,
) -> dict[str, Any]:
    """Compute primary paired deltas and uncertainty-quality metrics."""
    baseline_hits = hits_at_k(baseline, labels, labels, k_values)
    proposed_hits = hits_at_k(proposed, labels, labels, k_values)
    intervals = {
        f"recall_at_{k}": paired_bootstrap_recall_delta(
            baseline_hits[k],
            proposed_hits[k],
            samples=bootstrap_samples,
            seed=seed + k,
        )
        for k in baseline_hits
    }
    top1_correct = baseline_hits[1]
    top1_uncertainty = candidate_uncertainty[baseline[:, 0]]
    errors = ~top1_correct
    primary_interval = intervals["recall_at_1"]
    secondary_deltas = [
        intervals[f"recall_at_{k}"]["delta"]
        for k in baseline_hits
        if k != 1
    ]
    primary_supported = (
        primary_interval["lower"] > 0
        and all(
            delta >= -maximum_secondary_regression
            for delta in secondary_deltas
        )
    )
    return {
        "baseline": evaluate_ranking(baseline, labels, labels, k_values),
        "r1": evaluate_ranking(proposed, labels, labels, k_values),
        "paired_delta": intervals,
        "uncertainty": {
            "error_auroc": binary_auroc(top1_uncertainty, errors.long()),
            "error_auprc": binary_auprc(top1_uncertainty, errors.long()),
            "aurc": aurc(top1_uncertainty, errors.long()),
        },
        "primary_decision": {
            "comparison": "R1 vs R0",
            "supported": primary_supported,
            "criterion": (
                "Recall@1 paired CI lower bound > 0 and no Recall@2/4/8 "
                f"delta below {-maximum_secondary_regression:g}"
            ),
        },
    }
