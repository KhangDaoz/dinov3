"""Ordered query--candidate compatibility, never bidirectional averaging."""

import torch
from torch import Tensor, nn


class PairConfidenceNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 768) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(3 * embedding_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1),
        )

    def forward(self, query: Tensor, candidate: Tensor) -> Tensor:
        if query.ndim != 2 or query.shape != candidate.shape:
            raise ValueError("Pair endpoints must have identical [B,D] shapes")
        if query.shape[1] != self.embedding_dim:
            raise ValueError("Unexpected pair embedding dimension")
        # Cached features are immutable; only the MLP receives gradients.
        q, x = query.detach().float(), candidate.detach().float()
        return self.network(torch.cat((q, x, (q - x).abs()), dim=1)).squeeze(-1)
