"""Cosine retrieval, deterministic reranking, and Recall@K."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True)
class RetrievalResult:
    indices: Tensor
    scores: Tensor


def l2_normalize(features: Tensor) -> Tensor:
    if features.ndim != 2:
        raise ValueError("features must have shape [items, dimensions]")
    return functional.normalize(features.float(), p=2, dim=-1)


def cosine_rankings(
    query_features: Tensor,
    gallery_features: Tensor | None = None,
    query_ids: Tensor | None = None,
    gallery_ids: Tensor | None = None,
    chunk_size: int = 1024,
    ranking_depth: int | None = None,
) -> RetrievalResult:
    """Compute exact cosine rankings in GPU-friendly query chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    gallery_features = (
        query_features if gallery_features is None else gallery_features
    )
    if ranking_depth is not None and not 0 < ranking_depth <= len(gallery_features):
        raise ValueError("ranking_depth exceeds gallery size")
    queries = l2_normalize(query_features)
    gallery = l2_normalize(gallery_features).to(queries.device)
    same_collection = gallery_features is query_features
    if (query_ids is None) != (gallery_ids is None):
        raise ValueError("query_ids and gallery_ids must be provided together")
    if query_ids is not None:
        query_ids = query_ids.to(queries.device)
        gallery_ids = gallery_ids.to(queries.device)

    all_indices = []
    all_scores = []
    for start in range(0, len(queries), chunk_size):
        end = min(start + chunk_size, len(queries))
        scores = queries[start:end] @ gallery.T
        if query_ids is not None:
            self_mask = query_ids[start:end, None] == gallery_ids[None, :]
            scores = scores.masked_fill(self_mask, -torch.inf)
        elif same_collection:
            rows = torch.arange(end - start, device=scores.device)
            columns = torch.arange(start, end, device=scores.device)
            scores[rows, columns] = -torch.inf
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        if ranking_depth is not None:
            order = order[:, :ranking_depth]
        ranked_scores = torch.gather(scores, 1, order)
        all_indices.append(order.cpu())
        all_scores.append(ranked_scores.cpu())
    return RetrievalResult(torch.cat(all_indices), torch.cat(all_scores))


def recall_at_k(
    rankings: Tensor,
    query_labels: Tensor,
    gallery_labels: Tensor,
    k_values: Iterable[int],
) -> dict[int, float]:
    """Return the fraction of queries with a relevant result in top K."""
    if rankings.ndim != 2 or rankings.shape[0] != len(query_labels):
        raise ValueError("Rankings and query labels have incompatible shapes")
    if rankings.numel() and (
        rankings.min() < 0 or rankings.max() >= len(gallery_labels)
    ):
        raise ValueError("Rankings contain an invalid gallery index")
    relevant = gallery_labels[rankings] == query_labels[:, None]
    metrics: dict[int, float] = {}
    for k in k_values:
        if not 0 < k <= rankings.shape[1]:
            raise ValueError(f"Invalid Recall@K value: {k}")
        metrics[int(k)] = relevant[:, :k].any(dim=1).float().mean().item()
    return metrics


def uncertainty_rerank(rankings: Tensor, uncertainty: Tensor, top_n: int) -> Tensor:
    """R1: reorder top-N by ascending candidate uncertainty."""
    if rankings.ndim != 2 or uncertainty.ndim != 1:
        raise ValueError("Invalid ranking or uncertainty shape")
    if not 0 < top_n <= rankings.shape[1]:
        raise ValueError("top_n exceeds ranking width")
    candidate_uncertainty = uncertainty[rankings[:, :top_n]]
    local_order = torch.argsort(
        candidate_uncertainty,
        dim=1,
        descending=False,
        stable=True,
    )
    reranked = rankings.clone()
    reranked[:, :top_n] = torch.gather(rankings[:, :top_n], 1, local_order)
    return reranked


def _minmax(values: Tensor) -> Tensor:
    minimum = values.min(dim=1, keepdim=True).values
    value_range = values.max(dim=1, keepdim=True).values - minimum
    return torch.where(value_range > 0, (values - minimum) / value_range, 0.0)


def certainty_fusion_rerank(
    rankings: Tensor,
    ranked_cosine_scores: Tensor,
    uncertainty: Tensor,
    top_n: int,
    beta: float,
) -> Tensor:
    """R2: fuse normalized cosine and image-level certainty in top-N."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if rankings.shape != ranked_cosine_scores.shape:
        raise ValueError("Rankings and scores must have identical shapes")
    if not 0 < top_n <= rankings.shape[1]:
        raise ValueError("top_n exceeds ranking width")
    cosine = _minmax(ranked_cosine_scores[:, :top_n].float())
    certainty = 1.0 - uncertainty[rankings[:, :top_n]].float()
    certainty = _minmax(certainty)
    fused = (1.0 - beta) * cosine + beta * certainty
    local_order = torch.argsort(fused, dim=1, descending=True, stable=True)
    reranked = rankings.clone()
    reranked[:, :top_n] = torch.gather(rankings[:, :top_n], 1, local_order)
    return reranked


def random_rerank(
    rankings: Tensor,
    top_n: int,
    seed: int,
) -> Tensor:
    """R3: deterministic random-permutation negative control."""
    if not 0 < top_n <= rankings.shape[1]:
        raise ValueError("top_n exceeds ranking width")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand(rankings.shape[0], top_n, generator=generator)
    order = torch.argsort(noise, dim=1)
    reranked = rankings.clone()
    reranked[:, :top_n] = torch.gather(rankings[:, :top_n], 1, order)
    return reranked
