import pytest
import torch
import torch.nn.functional as F

from src.retrieval import recall_at_k


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
