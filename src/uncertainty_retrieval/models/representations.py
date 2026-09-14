"""Image representation pipelines for E2A."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class AttentionPoolingOutput:
    embedding: Tensor
    weights: Tensor


class AttentionPatchPooling(nn.Module):
    """M4 content-dependent weighting over final-layer patch tokens."""

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        patch_tokens: int = 196,
    ) -> None:
        super().__init__()
        if min(embedding_dim, hidden_dim, patch_tokens) <= 0:
            raise ValueError("Attention dimensions must be positive")
        self.embedding_dim = embedding_dim
        self.patch_tokens = patch_tokens
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.score.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.score.bias)

    def forward(self, patches: Tensor) -> AttentionPoolingOutput:
        expected = (self.patch_tokens, self.embedding_dim)
        if patches.ndim != 3 or tuple(patches.shape[1:]) != expected:
            raise ValueError(
                f"Patch input must have shape [batch, {expected[0]}, {expected[1]}]"
            )
        patches = patches.float()
        if not torch.isfinite(patches).all():
            raise ValueError("Attention input contains NaN or Inf")
        logits = self.score(torch.tanh(self.hidden(patches))).squeeze(-1)
        weights = torch.softmax(logits.float(), dim=1)
        embedding = torch.sum(weights.unsqueeze(-1) * patches, dim=1)
        if not torch.isfinite(embedding).all():
            raise FloatingPointError("Attention embedding is not finite")
        return AttentionPoolingOutput(embedding, weights)
