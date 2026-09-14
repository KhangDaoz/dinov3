"""Image representation pipelines for E2A."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .dinov3 import DINOv3Tokens


class CLSRepresentation(nn.Module):
    """M1: return the final CLS token without learned transformation."""

    def __init__(self, embedding_dim: int = 768) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim

    def forward(self, tokens: DINOv3Tokens) -> Tensor:
        features = tokens.cls
        if features.ndim != 2 or features.shape[-1] != self.embedding_dim:
            raise ValueError(
                "CLS token must have shape "
                f"[batch, {self.embedding_dim}], got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise ValueError("CLS features contain NaN or Inf")
        return features.float()

