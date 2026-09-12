import pytest
import torch

from src.representations import FusionProjection, raw_fusion_features
from src.train_projection import (
    BalancedBatchSampler,
    ProxyAnchorLoss,
    select_best_epoch,
    stratified_train_validation_split,
)
from src.utils import load_config


def test_raw_fusion_excludes_register_tokens_and_preserves_order():
    tokens = torch.tensor(
        [[[1.0, 2.0], [999.0, 999.0], [3.0, 4.0], [5.0, 6.0]]]
    )
    cls, mean_patch = raw_fusion_features(tokens, num_register_tokens=1)
    assert torch.equal(cls, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(mean_patch, torch.tensor([[4.0, 5.0]]))


def test_fusion_projection_shape_norm_and_gradient_scope():
    model = FusionProjection(feature_dim=2, projection_dim=3)
    cls = torch.tensor([[1.0, 2.0]], requires_grad=False)
    mean_patch = torch.tensor([[3.0, 4.0]], requires_grad=False)
    result = model(cls, mean_patch)
    assert result.shape == (1, 3)
    assert torch.allclose(result.norm(dim=1), torch.ones(1))
    result.sum().backward()
    assert model.projection.weight.grad is not None
    assert cls.grad is None
    assert mean_patch.grad is None


def test_proxy_anchor_matches_direct_formula():
    loss = ProxyAnchorLoss(2, 2, margin=0.1, alpha=2.0)
    with torch.no_grad():
        loss.proxies.copy_(torch.eye(2))
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    cosine = torch.eye(2)
    positive = torch.log1p(torch.exp(-2.0 * (torch.tensor(1.0) - 0.1)))
    negative = torch.log1p(torch.exp(2.0 * (torch.tensor(0.0) + 0.1)))
    expected = positive + negative
    assert torch.allclose(loss(embeddings, labels), expected)


def test_proxy_anchor_has_finite_gradients_with_absent_classes():
    criterion = ProxyAnchorLoss(3, 2)
    embeddings = torch.randn(4, 2, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    value = criterion(embeddings, labels)
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(embeddings.grad).all()
    assert torch.isfinite(criterion.proxies.grad).all()


def test_proxy_anchor_rejects_out_of_range_labels():
    criterion = ProxyAnchorLoss(2, 2)
    with pytest.raises(ValueError, match="ngoài khoảng"):
        criterion(torch.randn(2, 2), torch.tensor([0, 2]))


def test_balanced_sampler_is_deterministic_and_balanced():
    labels = torch.tensor([0] * 3 + [1] * 3 + [2] * 3)
    first = BalancedBatchSampler(labels, 2, 2, seed=7, num_batches=2)
    second = BalancedBatchSampler(labels, 2, 2, seed=7, num_batches=2)
    first_batches = list(first)
    assert first_batches == list(second)
    for batch in first_batches:
        counts = torch.bincount(labels[batch], minlength=3)
        assert sorted(counts[counts > 0].tolist()) == [2, 2]


def test_stratified_split_is_deterministic_and_disjoint():
    labels = torch.tensor([0] * 10 + [1] * 10)
    train_a, validation_a = stratified_train_validation_split(labels, 0.2, 42)
    train_b, validation_b = stratified_train_validation_split(labels, 0.2, 42)
    assert torch.equal(train_a, train_b)
    assert torch.equal(validation_a, validation_b)
    assert not set(train_a.tolist()) & set(validation_a.tolist())
    assert torch.bincount(labels[validation_a]).tolist() == [2, 2]


def test_best_epoch_tie_breaking():
    history = [
        {"epoch": 1, "validation_recall@1": 0.8, "validation_loss": 1.0},
        {"epoch": 2, "validation_recall@1": 0.8, "validation_loss": 0.9},
        {"epoch": 3, "validation_recall@1": 0.7, "validation_loss": 0.1},
    ]
    assert select_best_epoch(history) == 2


def test_m3_config_resolves_training_settings():
    config = load_config("configs/cub_e2a_m3.yaml")
    assert config["representation"] == "fusion"
    assert config["output_dir"] == "outputs/fusion"
    assert config["training"]["proxy_lr"] == 0.01
