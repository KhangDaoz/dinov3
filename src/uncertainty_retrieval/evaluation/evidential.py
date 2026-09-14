"""Uncertainty quality and paired retrieval statistics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import Tensor


def _binary_inputs(scores: Tensor, targets: Tensor) -> tuple[np.ndarray, np.ndarray]:
    scores_np = scores.detach().cpu().numpy().astype(np.float64)
    targets_np = targets.detach().cpu().numpy().astype(np.int64)
    if scores_np.ndim != 1 or targets_np.shape != scores_np.shape:
        raise ValueError("scores and targets must be matching vectors")
    if not np.isin(targets_np, [0, 1]).all():
        raise ValueError("targets must be binary")
    if not np.isfinite(scores_np).all():
        raise ValueError("scores must be finite")
    return scores_np, targets_np


def binary_auroc(scores: Tensor, targets: Tensor) -> float:
    """AUROC with average ranks for ties; higher score predicts target 1."""
    scores_np, targets_np = _binary_inputs(scores, targets)
    positives = int(targets_np.sum())
    negatives = len(targets_np) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(scores_np, kind="stable")
    sorted_scores = scores_np[order]
    ranks = np.empty(len(scores_np), dtype=np.float64)
    start = 0
    while start < len(scores_np):
        end = start + 1
        while end < len(scores_np) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[targets_np == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def binary_auprc(scores: Tensor, targets: Tensor) -> float:
    """Average precision; higher score predicts target 1."""
    scores_np, targets_np = _binary_inputs(scores, targets)
    positives = int(targets_np.sum())
    if positives == 0:
        raise ValueError("AUPRC requires at least one positive")
    order = np.argsort(-scores_np, kind="stable")
    ordered_targets = targets_np[order]
    precision = np.cumsum(ordered_targets) / np.arange(1, len(targets_np) + 1)
    return float((precision * ordered_targets).sum() / positives)


def aurc(uncertainty: Tensor, errors: Tensor) -> float:
    """Area under empirical risk-coverage curve, lower is better."""
    uncertainty_np, errors_np = _binary_inputs(uncertainty, errors)
    order = np.argsort(uncertainty_np, kind="stable")
    risks = np.cumsum(errors_np[order]) / np.arange(1, len(errors_np) + 1)
    return float(risks.mean())


def paired_bootstrap_recall_delta(
    baseline_hits: Tensor,
    proposed_hits: Tensor,
    samples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap a paired confidence interval for a Recall@K delta."""
    if baseline_hits.shape != proposed_hits.shape or baseline_hits.ndim != 1:
        raise ValueError("Hit vectors must have identical one-dimensional shapes")
    if samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap settings")
    delta = proposed_hits.float() - baseline_hits.float()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        len(delta),
        (samples, len(delta)),
        generator=generator,
    )
    bootstrapped = delta.cpu()[indices].mean(dim=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "delta": delta.mean().item(),
        "lower": torch.quantile(bootstrapped, tail).item(),
        "upper": torch.quantile(bootstrapped, 1.0 - tail).item(),
        "confidence": confidence,
        "samples": samples,
    }


def hits_at_k(
    rankings: Tensor,
    query_labels: Tensor,
    gallery_labels: Tensor,
    k_values: Iterable[int],
) -> dict[int, Tensor]:
    relevant = gallery_labels[rankings] == query_labels[:, None]
    return {int(k): relevant[:, :k].any(dim=1) for k in k_values}
