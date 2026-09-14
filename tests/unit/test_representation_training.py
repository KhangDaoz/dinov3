import torch

from uncertainty_retrieval.training.representation import (
    DistributedClassBalancedBatchSampler,
    epoch_hit_key,
)


def test_balanced_sampler_is_deterministic_and_globally_balanced() -> None:
    labels = torch.arange(4).repeat_interleave(5)
    left = DistributedClassBalancedBatchSampler(labels, 2, 2, 0, 2, 42)
    right = DistributedClassBalancedBatchSampler(labels, 2, 2, 1, 2, 42)
    left_batch = next(iter(left))
    right_batch = next(iter(right))
    global_labels = labels[left_batch + right_batch]
    counts = torch.bincount(global_labels, minlength=4)
    assert sorted(counts[counts > 0].tolist()) == [2, 2]
    assert left_batch == next(iter(left))


def test_epoch_selection_uses_integer_hits_then_earliest_epoch() -> None:
    assert epoch_hit_key({1: 10, 2: 20, 4: 30, 8: 40}, 2) > epoch_hit_key(
        {1: 9, 2: 99, 4: 99, 8: 99}, 1
    )
    assert epoch_hit_key({1: 10, 2: 20, 4: 30, 8: 40}, 1) > epoch_hit_key(
        {1: 10, 2: 20, 4: 30, 8: 40}, 2
    )
