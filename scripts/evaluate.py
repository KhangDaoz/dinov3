#!/usr/bin/env python
"""Evaluate B0 and tune/evaluate E1 reranking without changing embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from uncertainty_retrieval.config import load_config
from uncertainty_retrieval.data.patch_cache import load_feature_cache
from uncertainty_retrieval.evaluation.reporting import (
    final_e1_statistics,
    select_r1_top_n,
    select_r2_parameters,
)
from uncertainty_retrieval.evaluation.retrieval import (
    certainty_fusion_rerank,
    cosine_rankings,
    uncertainty_rerank,
)
from uncertainty_retrieval.training.evidential import load_evidential_outputs
from uncertainty_retrieval.utils import resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=("tune", "test"), default="tune")
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--beta", type=float)
    return parser.parse_args()


def _subset(cache: dict, image_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    index = {int(value): position for position, value in enumerate(cache["image_ids"])}
    positions = [index[int(value)] for value in image_ids.tolist()]
    return (
        cache["features"][positions],
        torch.tensor(cache["labels"])[positions],
    )


def _save_rankings(
    path: Path,
    rankings: torch.Tensor,
    scores: torch.Tensor | None,
    image_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "query_image_ids": image_ids,
            "gallery_image_ids": image_ids,
            "labels": labels,
            "indices": rankings,
            "scores": scores,
        },
        temporary,
    )
    temporary.replace(path)


def _failure_cases(
    baseline: torch.Tensor,
    proposed: torch.Tensor,
    labels: torch.Tensor,
    image_ids: torch.Tensor,
    uncertainty: torch.Tensor,
    limit: int = 100,
) -> list[dict]:
    baseline_correct = labels[baseline[:, 0]] == labels
    proposed_correct = labels[proposed[:, 0]] == labels
    changed = torch.nonzero(baseline_correct != proposed_correct).flatten()
    cases = []
    for query_index in changed[:limit].tolist():
        baseline_index = int(baseline[query_index, 0])
        proposed_index = int(proposed[query_index, 0])
        cases.append(
            {
                "query_image_id": int(image_ids[query_index]),
                "query_label": int(labels[query_index]),
                "baseline_top1_image_id": int(image_ids[baseline_index]),
                "baseline_top1_label": int(labels[baseline_index]),
                "baseline_top1_uncertainty": float(uncertainty[baseline_index]),
                "r1_top1_image_id": int(image_ids[proposed_index]),
                "r1_top1_label": int(labels[proposed_index]),
                "r1_top1_uncertainty": float(uncertainty[proposed_index]),
                "outcome": (
                    "improved"
                    if proposed_correct[query_index]
                    else "regressed"
                ),
            }
        )
    return cases


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device()
    seed_root = Path(config.output.root) / f"seed_{args.seed}"
    root = seed_root / ("tuning" if args.stage == "tune" else "final")
    evidential = load_evidential_outputs(
        root / "uncertainty" / f"{'validation' if args.stage == 'tune' else 'test'}.pt"
    )
    cache = load_feature_cache(config.output.feature_cache)
    features, labels = _subset(cache, evidential["image_ids"])
    result = cosine_rankings(
        features.to(device),
        chunk_size=config.retrieval.similarity_chunk_size,
    )
    uncertainty = evidential["uncertainty"]
    if args.stage == "tune":
        top_n, r1_grid = select_r1_top_n(
            result.indices,
            uncertainty,
            labels,
            config.retrieval.top_n_grid,
        )
        (r2_n, beta), r2_grid = select_r2_parameters(
            result.indices,
            result.scores,
            uncertainty,
            labels,
            config.retrieval.top_n_grid,
            config.retrieval.beta_grid,
        )
        write_json(
            {
                "r1": {"top_n": top_n, "grid": r1_grid},
                "r2": {"top_n": r2_n, "beta": beta, "grid": r2_grid},
            },
            root / "metrics" / "selection.json",
        )
        return
    if args.top_n is None:
        raise ValueError("--top-n must be the locked validation value for test")
    r1 = uncertainty_rerank(result.indices, uncertainty, args.top_n)
    statistics = final_e1_statistics(
        result.indices,
        r1,
        labels,
        uncertainty,
        config.retrieval.recall_k,
        config.retrieval.bootstrap_samples,
        args.seed,
    )
    if args.beta is not None:
        r2 = certainty_fusion_rerank(
            result.indices,
            result.scores,
            uncertainty,
            args.top_n,
            args.beta,
        )
        from uncertainty_retrieval.evaluation.reporting import evaluate_ranking

        statistics["r2_secondary"] = evaluate_ranking(
            r2,
            labels,
            labels,
            config.retrieval.recall_k,
        )
        _save_rankings(
            root / "rankings" / "r2_certainty_fusion.pt",
            r2,
            None,
            evidential["image_ids"],
            labels,
        )
    _save_rankings(
        root / "rankings" / "r0_cosine.pt",
        result.indices,
        result.scores,
        evidential["image_ids"],
        labels,
    )
    _save_rankings(
        root / "rankings" / "r1_uncertainty.pt",
        r1,
        None,
        evidential["image_ids"],
        labels,
    )
    write_json(
        _failure_cases(
            result.indices,
            r1,
            labels,
            evidential["image_ids"],
            uncertainty,
        ),
        root / "failure_cases" / "top1_changes.json",
    )
    write_json(statistics, root / "metrics" / "test.json")


if __name__ == "__main__":
    main()
