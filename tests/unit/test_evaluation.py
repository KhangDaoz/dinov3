import pytest
import torch

from uncertainty_retrieval.evaluation.evidential import (
    aurc,
    binary_auprc,
    binary_auroc,
    paired_bootstrap_recall_delta,
)
from uncertainty_retrieval.evaluation.reporting import final_e1_statistics


def test_binary_metrics_for_perfect_ordering() -> None:
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    targets = torch.tensor([0, 0, 1, 1])
    assert binary_auroc(scores, targets) == pytest.approx(1.0)
    assert binary_auprc(scores, targets) == pytest.approx(1.0)


def test_aurc_prefers_early_correct_predictions() -> None:
    uncertainty = torch.tensor([0.1, 0.2, 0.8, 0.9])
    good_errors = torch.tensor([0, 0, 1, 1])
    bad_errors = torch.tensor([1, 1, 0, 0])
    assert aurc(uncertainty, good_errors) < aurc(uncertainty, bad_errors)


def test_paired_bootstrap_is_deterministic() -> None:
    baseline = torch.tensor([0, 1, 0, 1], dtype=torch.bool)
    proposed = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
    first = paired_bootstrap_recall_delta(baseline, proposed, 100, 42)
    second = paired_bootstrap_recall_delta(baseline, proposed, 100, 42)
    assert first == second
    assert first["delta"] == pytest.approx(0.5)


def test_primary_decision_is_strictly_r1_vs_r0() -> None:
    # Half of R0 top-1 results are wrong; R1 corrects every one.
    labels = torch.arange(100).repeat_interleave(2)
    correct_partner = torch.arange(200).reshape(100, 2).flip(1).flatten()
    wrong_partner = (correct_partner + 2) % 200
    baseline_top1 = correct_partner.clone()
    baseline_top1[:100] = wrong_partner[:100]
    second = correct_partner
    baseline = torch.stack((baseline_top1, second), dim=1)
    proposed = torch.stack((correct_partner, wrong_partner), dim=1)
    uncertainty = torch.linspace(0.1, 0.9, 200)
    statistics = final_e1_statistics(
        baseline,
        proposed,
        labels,
        uncertainty,
        k_values=(1, 2),
        bootstrap_samples=500,
        seed=42,
    )
    assert statistics["primary_decision"]["comparison"] == "R1 vs R0"
    assert statistics["primary_decision"]["supported"]
