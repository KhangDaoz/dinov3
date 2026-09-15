"""Static cosine hard-negative pools and epoch-only resampling."""

import torch
from torch import Tensor


def validate_pair_endpoints(pairs: Tensor, labels: Tensor, targets: Tensor) -> None:
    if pairs.ndim != 2 or pairs.shape[1] != 2 or targets.shape != (len(pairs),):
        raise ValueError("Invalid pair fields")
    if not len(pairs) or pairs.min() < 0 or pairs.max() >= len(labels):
        raise ValueError("Pair endpoint outside the permitted split")
    if (pairs[:, 0] == pairs[:, 1]).any():
        raise ValueError("Self-pairs are forbidden")
    expected = labels[pairs[:, 0]].eq(labels[pairs[:, 1]]).float()
    if not torch.equal(expected, targets.float()):
        raise ValueError("Pair target does not match endpoint classes")


def sample_epoch_pairs(labels: Tensor, static_candidates: Tensor, epoch: int,
                       seed: int = 42) -> tuple[Tensor, Tensor, dict]:
    if labels.ndim != 1 or static_candidates.shape[0] != len(labels):
        raise ValueError("Static pool and labels differ")
    if static_candidates.min() < 0 or static_candidates.max() >= len(labels):
        raise ValueError("Static candidate outside fit split")
    generator = torch.Generator().manual_seed(seed + epoch)
    rows, targets = [], []
    fallbacks = 0
    for query in range(len(labels)):
        positive = torch.where(labels == labels[query])[0]
        positive = positive[positive != query]
        negative = torch.where(labels != labels[query])[0]
        hard = static_candidates[query]
        hard = hard[labels[hard] != labels[query]]
        if not len(positive) or not len(negative):
            raise ValueError("Empty positive or negative fit pool")
        if not len(hard):
            hard = negative
            fallbacks += 1
        for pool, count, target in ((positive, 16, 1), (hard, 8, 0), (negative, 8, 0)):
            selected = pool[torch.randint(len(pool), (count,), generator=generator)]
            rows.append(torch.stack((torch.full_like(selected, query), selected), 1))
            targets.append(torch.full((count,), float(target)))
    pairs, target = torch.cat(rows), torch.cat(targets)
    swap = torch.rand(len(pairs), generator=generator) < 0.5
    pairs[swap] = pairs[swap].flip(1)
    order = torch.randperm(len(pairs), generator=generator)
    pairs, target = pairs[order], target[order]
    validate_pair_endpoints(pairs, labels, target)
    unique = len(torch.unique(pairs, dim=0))
    return pairs, target, {
        "seed": seed, "epoch": epoch, "pairs": len(pairs),
        "repeated_pair_fraction": 1 - unique / len(pairs),
        "hard_negative_fallback_queries": fallbacks, "mining": "static_m1_cosine",
        "random_swap": "training_only", "positive_fraction": float(target.mean()),
    }
