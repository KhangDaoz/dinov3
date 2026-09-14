from types import SimpleNamespace

import pytest
import torch
from torch import nn

from uncertainty_retrieval.models.dinov3 import DINOv3Backbone


class FakeDINOv3(nn.Module):
    def __init__(self, tokens: int = 201, dimensions: int = 768) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.tokens = tokens
        self.dimensions = dimensions

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        hidden = torch.arange(
            len(pixel_values) * self.tokens * self.dimensions,
            dtype=torch.float32,
        ).reshape(len(pixel_values), self.tokens, self.dimensions)
        return SimpleNamespace(last_hidden_state=hidden)


def test_preserves_cls_register_patch_order_and_freezes() -> None:
    model = FakeDINOv3()
    backbone = DINOv3Backbone(model, register_tokens=4)
    tokens = backbone(torch.zeros(2, 3, 224, 224))
    assert tokens.cls.shape == (2, 768)
    assert tokens.registers.shape == (2, 4, 768)
    assert tokens.patches.shape == (2, 196, 768)
    assert torch.equal(tokens.all_tokens()[:, 0], tokens.cls)
    assert not any(parameter.requires_grad for parameter in backbone.parameters())
    backbone.train()
    assert not backbone.model.training


def test_rejects_output_without_patch_tokens() -> None:
    backbone = DINOv3Backbone(FakeDINOv3(tokens=5), register_tokens=4)
    with pytest.raises(ValueError, match="does not contain"):
        backbone(torch.zeros(1, 3, 224, 224))

