import torch
import torch.nn.functional as F

from src.pair_sampling import (
    PairIndexDataset,
    build_hard_banks,
    generate_pair_indices,
)


def sample_data():
    torch.manual_seed(42)
    embeddings = F.normalize(torch.randn(12, 4), dim=1)
    labels = torch.arange(3).repeat_interleave(4)
    indices = torch.arange(12)
    return embeddings, labels, indices


def test_banks_and_pairs_respect_labels_and_self_exclusion():
    embeddings, labels, indices = sample_data()
    positives, negatives = build_hard_banks(
        embeddings, labels, indices, negative_bank_size=3, device="cpu"
    )
    anchors, partners, targets = generate_pair_indices(
        labels, indices, positives, negatives, seed=42, epoch=1
    )
    assert len(anchors) == len(indices) * 4
    assert torch.all(anchors != partners)
    assert torch.all(labels[anchors[targets == 1]] == labels[partners[targets == 1]])
    assert torch.all(labels[anchors[targets == 0]] != labels[partners[targets == 0]])
    for anchor, bank in negatives.items():
        assert torch.all(labels[bank] != labels[anchor])


def test_pair_generation_is_deterministic_and_changes_by_epoch():
    embeddings, labels, indices = sample_data()
    positives, negatives = build_hard_banks(
        embeddings, labels, indices, negative_bank_size=3, device="cpu"
    )
    first = generate_pair_indices(labels, indices, positives, negatives, 42, 1)
    repeated = generate_pair_indices(labels, indices, positives, negatives, 42, 1)
    changed = generate_pair_indices(labels, indices, positives, negatives, 42, 2)
    assert all(torch.equal(a, b) for a, b in zip(first, repeated))
    assert any(not torch.equal(a, b) for a, b in zip(first, changed))


def test_bidirectional_dataset_doubles_pairs():
    embeddings, _, _ = sample_data()
    anchors = torch.tensor([0, 1])
    partners = torch.tensor([2, 3])
    targets = torch.tensor([1.0, 0.0])
    dataset = PairIndexDataset(
        embeddings, anchors, partners, targets, bidirectional=True
    )
    assert len(dataset) == 4
    assert torch.equal(dataset[0][0], dataset[2][1])
