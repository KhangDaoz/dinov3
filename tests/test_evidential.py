import pytest
import torch

from src.evidential import (
    EvidentialHead,
    annealing_coefficient,
    binary_ranking_metrics,
    classification_metrics,
    dirichlet_statistics,
    edl_mse_loss,
    risk_coverage_metrics,
    select_evidential_epoch,
    select_top_n,
)


def test_evidential_head_outputs_valid_dirichlet_and_backpropagates():
    torch.manual_seed(42)
    head = EvidentialHead(input_dim=4, hidden_dim=8, num_classes=3, dropout=0)
    embeddings = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    evidence = head(embeddings)
    alpha, _, probabilities, uncertainty = dirichlet_statistics(evidence)
    loss = edl_mse_loss(evidence, labels, annealing=0.5)
    loss.backward()

    assert evidence.shape == (6, 3)
    assert torch.all(evidence >= 0)
    assert torch.all(alpha > 1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(6))
    assert torch.all((uncertainty > 0) & (uncertainty <= 1))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_edl_loss_rejects_label_outside_head():
    with pytest.raises(ValueError, match="vượt số lớp"):
        edl_mse_loss(torch.ones(2, 3), torch.tensor([0, 3]))


def test_annealing_coefficient_reaches_one():
    assert annealing_coefficient(0, 10) == 0
    assert annealing_coefficient(5, 10) == 0.5
    assert annealing_coefficient(20, 10) == 1


def test_classification_metrics_are_exact_for_one_hot_predictions():
    probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    metrics = classification_metrics(probabilities, torch.tensor([0, 1]), bins=2)
    assert metrics["accuracy"] == 1
    assert metrics["negative_log_likelihood"] == 0
    assert metrics["brier_score"] == 0
    assert metrics["ece"] == 0


def test_uncertainty_detects_all_errors_in_ranking_metrics():
    metrics = binary_ranking_metrics(
        torch.tensor([0.1, 0.2, 0.8, 0.9]),
        torch.tensor([False, False, True, True]),
    )
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    risk = risk_coverage_metrics(
        torch.tensor([0.1, 0.2, 0.8, 0.9]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        coverage_points=(0.5, 1.0),
    )
    assert risk["risk_at_coverage"]["0.50"] == 0
    assert risk["risk_at_coverage"]["1.00"] == 0.5


def test_selection_tie_breaks_are_deterministic():
    history = [
        {"epoch": 1, "validation_loss": 0.5, "validation_accuracy": 0.8},
        {"epoch": 2, "validation_loss": 0.5, "validation_accuracy": 0.9},
        {"epoch": 3, "validation_loss": 0.6, "validation_accuracy": 1.0},
    ]
    assert select_evidential_epoch(history) == 2
    results = [
        {"top_n": 16, "metrics": {"recall@1": 0.8, "recall@2": 0.9}},
        {"top_n": 8, "metrics": {"recall@1": 0.8, "recall@2": 0.9}},
        {"top_n": 32, "metrics": {"recall@1": 0.7, "recall@2": 1.0}},
    ]
    assert select_top_n(results) == 8
