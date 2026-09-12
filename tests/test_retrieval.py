import pytest
import torch
import torch.nn.functional as F

from src.retrieval import (
    cosine_top_candidates,
    fuse_candidate_rankings,
    cosine_top_indices,
    recall_at_k,
    recall_from_indices,
    rerank_top_n_by_uncertainty,
)


def sample_embeddings():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    return F.normalize(embeddings, dim=1), torch.tensor([0, 0, 1, 1])


def test_recall_excludes_self_match():
    embeddings, labels = sample_embeddings()
    result = recall_at_k(
        embeddings, labels, [1, 2], chunk_size=2, device="cpu"
    )
    assert result == {"recall@1": 1.0, "recall@2": 1.0}


def test_chunk_size_does_not_change_result():
    embeddings, labels = sample_embeddings()
    whole = recall_at_k(
        embeddings, labels, [1, 2], chunk_size=4, device="cpu"
    )
    chunked = recall_at_k(
        embeddings, labels, [1, 2], chunk_size=1, device="cpu"
    )
    assert chunked == whole


def test_retrieval_requires_l2_normalization():
    embeddings, labels = sample_embeddings()
    embeddings[0] *= 2
    with pytest.raises(ValueError, match="L2"):
        recall_at_k(embeddings, labels, [1], chunk_size=2, device="cpu")


def test_retrieval_rejects_k_larger_than_gallery_without_self():
    embeddings, labels = sample_embeddings()
    with pytest.raises(ValueError, match="không được vượt"):
        recall_at_k(embeddings, labels, [4], chunk_size=2, device="cpu")


def test_cosine_indices_reproduce_recall():
    embeddings, labels = sample_embeddings()
    indices = cosine_top_indices(
        embeddings, top_n=2, chunk_size=1, device="cpu"
    )
    assert recall_from_indices(indices, labels, [1, 2]) == recall_at_k(
        embeddings, labels, [1, 2],
        chunk_size=2, device="cpu"
    )


def test_uncertainty_reranking_changes_only_requested_prefix_stably():
    indices = torch.tensor([[3, 2, 1, 0]])
    uncertainty = torch.tensor([0.9, 0.2, 0.2, 0.5])
    reranked = rerank_top_n_by_uncertainty(indices, uncertainty, top_n=3)
    assert reranked.tolist() == [[2, 1, 3, 0]]


def test_fused_reranking_preserves_suffix_and_cosine_control():
    indices = torch.tensor([[1, 2, 3]])
    cosine = torch.tensor([[0.9, 0.8, 0.7]])
    confidence = torch.tensor([[0.1, 0.9, 1.0]])
    reranked, _ = fuse_candidate_rankings(
        indices, cosine, confidence, weight=0, candidate_top_n=2
    )
    assert reranked.tolist() == [[2, 1, 3]]
    control, _ = fuse_candidate_rankings(
        indices, cosine, confidence, weight=1, candidate_top_n=3
    )
    assert torch.equal(control, indices)


def test_cosine_candidates_return_matching_scores():
    embeddings, _ = sample_embeddings()
    indices, scores = cosine_top_candidates(
        embeddings, top_n=2, chunk_size=2, device="cpu"
    )
    expected = torch.gather(embeddings @ embeddings.T, 1, indices)
    assert torch.allclose(scores, expected)
