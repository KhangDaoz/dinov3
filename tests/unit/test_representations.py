import pytest
import torch

from uncertainty_retrieval.models.dinov3 import DINOv3Tokens
from uncertainty_retrieval.models.representations import (
    CLSMeanPatchProjection,
    CLSRepresentation,
    MeanPatchRepresentation,
)


def test_cls_representation_uses_only_cls_and_returns_fp32() -> None:
    cls = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    tokens = DINOv3Tokens(
        cls=cls,
        registers=torch.full((2, 4, 3), 99.0),
        patches=torch.full((2, 2, 3), -99.0),
    )
    module = CLSRepresentation(3)
    result = module(tokens)
    assert result.dtype == torch.float32
    assert torch.equal(result, cls.float())
    assert sum(parameter.numel() for parameter in module.parameters()) == 0


def test_cls_representation_rejects_bad_dimension() -> None:
    tokens = DINOv3Tokens(torch.ones(2, 2), torch.empty(2, 0, 2), torch.ones(2, 1, 2))
    with pytest.raises(ValueError, match="CLS token"):
        CLSRepresentation(3)(tokens)


def test_mean_patch_uses_only_patches_and_accumulates_fp32() -> None:
    patches = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]],
        dtype=torch.float16,
    )
    tokens = DINOv3Tokens(
        cls=torch.full((2, 2), 999.0),
        registers=torch.full((2, 4, 2), -999.0),
        patches=patches,
    )
    module = MeanPatchRepresentation(embedding_dim=2, patch_tokens=2)
    result = module(tokens)
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.tensor([[2.0, 3.0], [4.0, 6.0]]))
    assert sum(parameter.numel() for parameter in module.parameters()) == 0


@pytest.mark.parametrize("shape", ((2, 1, 2), (2, 2, 3)))
def test_mean_patch_rejects_token_shape(shape: tuple[int, ...]) -> None:
    tokens = DINOv3Tokens(
        cls=torch.zeros(2, 2),
        registers=torch.zeros(2, 4, 2),
        patches=torch.zeros(shape),
    )
    with pytest.raises(ValueError, match="Patch tokens"):
        MeanPatchRepresentation(embedding_dim=2, patch_tokens=2)(tokens)


def test_m3_projection_concatenates_cls_then_mean_patch() -> None:
    module = CLSMeanPatchProjection(embedding_dim=2)
    with torch.no_grad():
        module.projection.weight.copy_(
            torch.tensor([[1.0, 0.0, 10.0, 0.0], [0.0, 1.0, 0.0, 10.0]])
        )
        module.projection.bias.zero_()
    cls = torch.tensor([[1.0, 2.0]], requires_grad=True)
    patch = torch.tensor([[3.0, 4.0]], requires_grad=True)
    result = module(cls, patch)
    assert torch.equal(result, torch.tensor([[31.0, 42.0]]))
    result.sum().backward()
    assert cls.grad is not None and patch.grad is not None
    assert sum(parameter.numel() for parameter in module.parameters()) == 10
