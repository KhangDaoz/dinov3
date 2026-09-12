import pytest
import torch
import torch.nn.functional as F

from src.pair_confidence import (
    PairConfidenceNetwork,
    build_pair_features,
    combine_scores,
    pair_bce_loss,
    select_pair_epoch,
    select_retrieval_parameters,
)


def test_full_pair_features_have_expected_order():
    first = torch.tensor([[1.0, 2.0]])
    second = torch.tensor([[3.0, 1.0]])
    features = build_pair_features(first, second)
    assert features.tolist() == [[1.0, 2.0, 3.0, 1.0, 2.0, 1.0]]


def test_symmetric_confidence_is_order_invariant():
    torch.manual_seed(42)
    model = PairConfidenceNetwork(
        embedding_dim=4, hidden_dims=(8, 4), dropout=0
    )
    first = F.normalize(torch.randn(5, 4), dim=1)
    second = F.normalize(torch.randn(5, 4), dim=1)
    assert torch.allclose(
        model.symmetric_confidence(first, second),
        model.symmetric_confidence(second, first),
    )


def test_pair_bce_backpropagates():
    model = PairConfidenceNetwork(
        embedding_dim=2, hidden_dims=(4, 2), dropout=0
    )
    loss = pair_bce_loss(
        model(torch.randn(4, 2), torch.randn(4, 2)),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_combine_score_endpoints():
    cosine = torch.tensor([-1.0, 1.0])
    confidence = torch.tensor([0.8, 0.2])
    assert torch.equal(combine_scores(cosine, confidence, 0), confidence)
    assert torch.equal(combine_scores(cosine, confidence, 1), torch.tensor([0.0, 1.0]))


def test_pair_selection_tie_breaks():
    history = [
        {"epoch": 1, "validation_auroc": 0.8, "validation_bce": 0.4},
        {"epoch": 2, "validation_auroc": 0.8, "validation_bce": 0.3},
    ]
    assert select_pair_epoch(history) == 2
    results = [
        {
            "candidate_top_n": 64,
            "lambda": 0.5,
            "metrics": {"recall@1": 0.8, "recall@2": 0.9, "recall@4": 1.0},
        },
        {
            "candidate_top_n": 32,
            "lambda": 0.75,
            "metrics": {"recall@1": 0.8, "recall@2": 0.9, "recall@4": 1.0},
        },
    ]
    assert select_retrieval_parameters(results) == {
        "candidate_top_n": 32,
        "lambda": 0.75,
    }
