import pytest
import torch

from src.representations import cls_embedding


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
