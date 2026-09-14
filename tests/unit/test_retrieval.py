import pytest
import torch

from uncertainty_retrieval.evaluation.retrieval import (
    certainty_fusion_rerank,
    cosine_rankings,
    l2_normalize,
    random_rerank,
    recall_at_k,
    uncertainty_rerank,
)


def test_l2_normalization() -> None:
    normalized = l2_normalize(torch.tensor([[3.0, 4.0], [1.0, 0.0]]))
    assert torch.allclose(normalized.norm(dim=1), torch.ones(2))


def test_cosine_rankings_exclude_self() -> None:
    features = torch.eye(3)
    result = cosine_rankings(features, chunk_size=2)
    assert torch.all(result.indices[:, 0] != torch.arange(3))
    assert torch.isneginf(result.scores[:, -1]).all()


def test_cosine_ties_use_gallery_index_order() -> None:
    queries = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    result = cosine_rankings(queries, gallery, chunk_size=1)
    assert result.indices.tolist() == [[0, 1]]


def test_recall_at_k_hand_computed() -> None:
    rankings = torch.tensor([[1, 2, 0], [0, 2, 1]])
    labels = torch.tensor([0, 1])
    gallery_labels = torch.tensor([0, 0, 1])
    metrics = recall_at_k(rankings, labels, gallery_labels, [1, 2])
    assert metrics[1] == pytest.approx(0.5)
    assert metrics[2] == pytest.approx(1.0)


def test_uncertainty_rerank_preserves_tail_and_ties() -> None:
    rankings = torch.tensor([[2, 0, 1, 3]])
    uncertainty = torch.tensor([0.2, 0.2, 0.8, 0.1])
    reranked = uncertainty_rerank(rankings, uncertainty, top_n=3)
    assert reranked.tolist() == [[0, 1, 2, 3]]


def test_certainty_fusion_uses_certainty_not_pair_confidence() -> None:
    rankings = torch.tensor([[0, 1, 2]])
    scores = torch.tensor([[0.9, 0.8, 0.1]])
    uncertainty = torch.tensor([0.9, 0.1, 0.0])
    reranked = certainty_fusion_rerank(
        rankings,
        scores,
        uncertainty,
        top_n=2,
        beta=1.0,
    )
    assert reranked.tolist() == [[1, 0, 2]]


def test_random_control_is_deterministic_and_preserves_tail() -> None:
    rankings = torch.arange(12).reshape(2, 6)
    first = random_rerank(rankings, 4, seed=42)
    second = random_rerank(rankings, 4, seed=42)
    assert torch.equal(first, second)
    assert torch.equal(first[:, 4:], rankings[:, 4:])

