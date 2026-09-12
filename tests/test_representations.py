import pytest
import torch

from src.representations import (
    build_embedding,
    cls_embedding,
    mean_patch_embedding,
)


def test_cls_embedding_selects_first_token_and_normalizes():
    tokens = torch.tensor(
        [
            [[3.0, 4.0], [100.0, 100.0]],
            [[0.0, 2.0], [-10.0, -10.0]],
        ]
    )
    result = cls_embedding(tokens)
    expected = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.allclose(result, expected)


def test_cls_embedding_rejects_non_finite_tokens():
    with pytest.raises(ValueError, match="NaN/Inf"):
        cls_embedding(torch.tensor([[[float("nan")]]]))


def test_cls_embedding_rejects_wrong_rank():
    with pytest.raises(ValueError, match="shape"):
        cls_embedding(torch.ones(2, 3))


def test_mean_patch_excludes_cls_and_multiple_register_tokens():
    tokens = torch.tensor(
        [[
            [1000.0, 1000.0],
            [-500.0, 900.0],
            [700.0, -800.0],
            [1.0, 3.0],
            [3.0, 1.0],
        ]]
    )
    result = mean_patch_embedding(tokens, num_register_tokens=2)
    expected = torch.tensor([[2.0, 2.0]])
    expected = torch.nn.functional.normalize(expected, dim=1)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.allclose(result, expected)


@pytest.mark.parametrize("register_count", [0, 1, 3])
def test_mean_patch_supports_dynamic_register_count(register_count):
    prefix = torch.full((1, 1 + register_count, 2), 100.0)
    patches = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]])
    tokens = torch.cat((prefix, patches), dim=1)
    result = mean_patch_embedding(tokens, register_count)
    assert torch.allclose(result, torch.tensor([[0.6, 0.8]]))


@pytest.mark.parametrize("register_count", [-1, 1.5])
def test_mean_patch_rejects_invalid_register_count(register_count):
    with pytest.raises(ValueError, match="num_register_tokens"):
        mean_patch_embedding(torch.ones(1, 3, 2), register_count)


def test_mean_patch_rejects_empty_patch_set():
    with pytest.raises(ValueError, match="Không còn patch token"):
        mean_patch_embedding(torch.ones(1, 3, 2), num_register_tokens=2)


def test_mean_patch_rejects_non_finite_patch():
    tokens = torch.tensor([[[0.0], [float("inf")]]])
    with pytest.raises(ValueError, match="NaN/Inf"):
        mean_patch_embedding(tokens, num_register_tokens=0)


def test_mean_patch_rejects_zero_vector():
    tokens = torch.tensor([[[9.0, 9.0], [1.0, -1.0], [-1.0, 1.0]]])
    with pytest.raises(ValueError, match="vector zero"):
        mean_patch_embedding(tokens, num_register_tokens=0)


def test_dispatcher_preserves_cls_and_mean_patch_behavior():
    tokens = torch.tensor([[[3.0, 4.0], [0.0, 2.0], [0.0, 2.0]]])
    assert torch.equal(build_embedding("cls", tokens), cls_embedding(tokens))
    assert torch.equal(
        build_embedding("mean_patch", tokens, 0),
        mean_patch_embedding(tokens, 0),
    )


def test_dispatcher_rejects_unknown_representation():
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        build_embedding("attention_pool", torch.ones(1, 2, 3))
