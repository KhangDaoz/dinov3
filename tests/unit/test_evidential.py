import pytest
import torch

from uncertainty_retrieval.models.evidential import (
    EvidentialHead,
    dirichlet_kl_to_uniform,
    edl_mse_loss,
)
from uncertainty_retrieval.training.evidential import (
    FeatureDataset,
    export_evidential_outputs,
    make_feature_loader,
    train_evidential_fixed_epochs,
)


def test_evidential_shapes_and_invariants() -> None:
    head = EvidentialHead(8, 3)
    output = head(torch.randn(4, 8))
    assert output.evidence.shape == (4, 3)
    assert output.alpha.shape == (4, 3)
    assert output.uncertainty.shape == (4,)
    assert torch.all(output.evidence >= 0)
    assert torch.all(output.alpha >= 1)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(4))
    assert torch.all((output.uncertainty > 0) & (output.uncertainty <= 1))


def test_zero_evidence_is_maximum_uncertainty() -> None:
    alpha = torch.ones(2, 5)
    uncertainty = 5 / alpha.sum(dim=1)
    assert torch.equal(uncertainty, torch.ones(2))


def test_kl_uniform_is_zero() -> None:
    kl = dirichlet_kl_to_uniform(torch.ones(3, 4))
    assert torch.allclose(kl, torch.zeros(3), atol=1e-6)


def test_kl_matches_torch_distribution() -> None:
    alpha = torch.tensor([[1.0, 2.0, 3.0]])
    expected = torch.distributions.kl_divergence(
        torch.distributions.Dirichlet(alpha),
        torch.distributions.Dirichlet(torch.ones_like(alpha)),
    )
    assert torch.allclose(dirichlet_kl_to_uniform(alpha), expected, atol=1e-6)


def test_loss_is_finite_and_has_gradients() -> None:
    head = EvidentialHead(6, 3)
    output = head(torch.randn(5, 6))
    loss, parts = edl_mse_loss(
        output.alpha,
        torch.tensor([0, 1, 2, 1, 0]),
        epoch=5,
        annealing_epochs=10,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["annealing"].item() == pytest.approx(0.5)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


def test_rejects_invalid_head_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        EvidentialHead(8, 3, architecture="unknown")


def test_export_retains_full_e_alpha_and_u() -> None:
    dataset = FeatureDataset(
        torch.randn(4, 8),
        torch.tensor([0, 1, 2, 1]),
        torch.tensor([4, 1, 3, 2]),
    )
    loader, _ = make_feature_loader(
        dataset,
        batch_size=2,
        num_workers=0,
        prefetch_factor=2,
        shuffle=False,
        distributed=False,
        seed=42,
    )
    payload = export_evidential_outputs(
        EvidentialHead(8, 3),
        loader,
        torch.device("cpu"),
        class_order=[0, 1, 2],
        schema_version=1,
    )
    assert payload["image_ids"].tolist() == [1, 2, 3, 4]
    assert payload["evidence"].shape == (4, 3)
    assert payload["alpha"].shape == (4, 3)
    assert payload["uncertainty"].shape == (4,)
    assert torch.allclose(payload["alpha"], payload["evidence"] + 1)


def test_fixed_epoch_retraining_saves_selected_epoch(tmp_path) -> None:
    dataset = FeatureDataset(
        torch.randn(8, 6),
        torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
        torch.arange(8),
    )
    loader, sampler = make_feature_loader(
        dataset,
        batch_size=4,
        num_workers=0,
        prefetch_factor=2,
        shuffle=True,
        distributed=False,
        seed=42,
    )
    checkpoint = tmp_path / "final.pt"
    history = train_evidential_fixed_epochs(
        EvidentialHead(6, 3),
        loader,
        torch.device("cpu"),
        epochs=2,
        annealing_epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        amp=False,
        checkpoint_path=checkpoint,
        sampler=sampler,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert len(history) == 2
    assert state["epoch"] == 2
    assert all(torch.isfinite(torch.tensor(row["development_loss"])) for row in history)
