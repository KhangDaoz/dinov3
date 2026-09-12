import pytest
import torch
import torch.nn.functional as F

from src.representations import (
    AttentionPool,
    mean_patch_embedding,
    raw_patch_tokens,
)
from src.utils import load_config


def test_raw_patch_tokens_excludes_cls_and_register_sentinels():
    tokens = torch.tensor(
        [[[999.0, 999.0], [-500.0, 700.0], [1.0, 3.0], [3.0, 1.0]]]
    )
    patches = raw_patch_tokens(tokens, num_register_tokens=1)
    assert torch.equal(patches, torch.tensor([[[1.0, 3.0], [3.0, 1.0]]]))


def test_uniform_attention_matches_mean_patch():
    tokens = torch.tensor(
        [[[100.0, 100.0], [1.0, 3.0], [3.0, 1.0], [2.0, 2.0]]]
    )
    patches = raw_patch_tokens(tokens, 0)
    model = AttentionPool(feature_dim=2, hidden_dim=3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result, weights = model(patches, return_weights=True)
    expected = mean_patch_embedding(tokens, 0)
    assert torch.allclose(result, expected)
    assert torch.allclose(weights, torch.full((1, 3), 1 / 3))


def test_attention_weights_and_embeddings_are_valid():
    torch.manual_seed(42)
    model = AttentionPool(feature_dim=4, hidden_dim=3)
    embeddings, weights = model(torch.randn(2, 5, 4), return_weights=True)
    assert embeddings.shape == (2, 4)
    assert weights.shape == (2, 5)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2))


def test_attention_gradients_only_reach_scorer():
    model = AttentionPool(feature_dim=2, hidden_dim=2)
    patches = torch.randn(2, 3, 2)
    model(patches).sum().backward()
    assert patches.grad is None
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("register_count", [-1, 1.5])
def test_raw_patch_tokens_rejects_invalid_register_count(register_count):
    with pytest.raises(ValueError, match="num_register_tokens"):
        raw_patch_tokens(torch.ones(1, 3, 2), register_count)


def test_attention_rejects_empty_or_non_finite_patches():
    model = AttentionPool(feature_dim=2, hidden_dim=2)
    with pytest.raises(ValueError, match="shape"):
        model(torch.empty(1, 0, 2))
    with pytest.raises(ValueError, match="NaN/Inf"):
        model(torch.tensor([[[float("nan"), 0.0]]]))


def test_m4_config_resolves():
    config = load_config("configs/cub_e2a_m4.yaml")
    assert config["representation"] == "attention_pool"
    assert config["patch_cache"]["dtype"] == "float32"
    assert config["patch_cache"]["prefetch_factor"] == 2
