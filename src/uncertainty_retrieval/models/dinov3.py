"""Frozen Hugging Face DINOv3 token extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DINOv3Tokens:
    """Final-layer tokens in the required CLS, register, patch order."""

    cls: Tensor
    registers: Tensor
    patches: Tensor

    def all_tokens(self) -> Tensor:
        return torch.cat((self.cls.unsqueeze(1), self.registers, self.patches), dim=1)


class DINOv3Backbone(nn.Module):
    """A frozen DINOv3 wrapper with explicit token-boundary validation."""

    def __init__(self, model: nn.Module, register_tokens: int = 4) -> None:
        super().__init__()
        if register_tokens < 0:
            raise ValueError("register_tokens cannot be negative")
        self.model = model
        self.register_tokens = register_tokens
        self.freeze()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str,
        register_tokens: int = 4,
    ) -> tuple["DINOv3Backbone", Any]:
        """Load the official Hugging Face model and image processor."""
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(
            model_id,
            revision=revision,
        )
        model = AutoModel.from_pretrained(model_id, revision=revision)
        configured_registers = getattr(model.config, "num_register_tokens", None)
        if (
            configured_registers is not None
            and configured_registers != register_tokens
        ):
            raise ValueError(
                "Configured register token count does not match checkpoint: "
                f"{register_tokens} != {configured_registers}"
            )
        return cls(model, register_tokens=register_tokens), processor

    def freeze(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "DINOv3Backbone":
        """Keep the pretrained backbone in evaluation mode."""
        super().train(False)
        self.model.eval()
        return self

    def forward(self, pixel_values: Tensor) -> DINOv3Tokens:
        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        boundary = 1 + self.register_tokens
        if hidden.ndim != 3 or hidden.shape[1] <= boundary:
            raise ValueError(
                "DINOv3 output does not contain CLS, register, and patch tokens"
            )
        return DINOv3Tokens(
            cls=hidden[:, 0],
            registers=hidden[:, 1:boundary],
            patches=hidden[:, boundary:],
        )


def processor_transform(processor: Any):
    """Adapt a Hugging Face image processor to a Dataset transform."""

    def transform(image) -> Tensor:
        values = processor(images=image, return_tensors="pt")["pixel_values"]
        return values.squeeze(0)

    return transform
