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


class MeanPatchRepresentation(nn.Module):
    """M2: average final patch tokens, excluding CLS and registers."""

    def __init__(self, embedding_dim: int = 768, patch_tokens: int = 196) -> None:
        super().__init__()
        if embedding_dim <= 0 or patch_tokens <= 0:
            raise ValueError("embedding_dim and patch_tokens must be positive")
        self.embedding_dim = embedding_dim
        self.patch_tokens = patch_tokens

    def forward(self, tokens: DINOv3Tokens) -> Tensor:
        patches = tokens.patches
        expected = (self.patch_tokens, self.embedding_dim)
        if patches.ndim != 3 or tuple(patches.shape[1:]) != expected:
            raise ValueError(
                "Patch tokens must have shape "
                f"[batch, {self.patch_tokens}, {self.embedding_dim}], "
                f"got {tuple(patches.shape)}"
            )
        if not torch.isfinite(patches).all():
            raise ValueError("Patch features contain NaN or Inf")
        return patches.float().mean(dim=1)


class CLSMeanPatchProjection(nn.Module):
    """M3 affine projection over ordered CLS and mean-patch inputs."""

    def __init__(self, embedding_dim: int = 768) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(2 * embedding_dim, embedding_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, cls: Tensor, mean_patch: Tensor) -> Tensor:
        expected = (self.embedding_dim,)
        if cls.ndim != 2 or tuple(cls.shape[1:]) != expected:
            raise ValueError("CLS input has an invalid shape")
        if mean_patch.shape != cls.shape:
            raise ValueError("CLS and mean-patch inputs must have equal shapes")
        if not torch.isfinite(cls).all() or not torch.isfinite(mean_patch).all():
            raise ValueError("M3 inputs contain NaN or Inf")
        fused = torch.cat((cls.float(), mean_patch.float()), dim=1)
        return self.projection(fused)
