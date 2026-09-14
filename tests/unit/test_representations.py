import pytest
import torch

from uncertainty_retrieval.models.dinov3 import DINOv3Tokens
from uncertainty_retrieval.models.representations import CLSRepresentation


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
