"""Metric-learning objectives used by learned E2A representations."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


def _log1p_sum_exp(logits: Tensor, dim: int) -> Tensor:
    zeros_shape = list(logits.shape)
    zeros_shape[dim] = 1
    zeros = torch.zeros(zeros_shape, dtype=logits.dtype, device=logits.device)
    return torch.logsumexp(torch.cat((zeros, logits), dim=dim), dim=dim)


class ProxyAnchorLoss(nn.Module):
    """Proxy Anchor loss with one trainable proxy per development class."""

    def __init__(
        self,
        classes: int,
        embedding_dim: int,
        alpha: float = 32.0,
        margin: float = 0.1,
    ) -> None:
        super().__init__()
        if classes <= 1 or embedding_dim <= 0 or alpha <= 0 or margin < 0:
            raise ValueError("Invalid Proxy Anchor dimensions or hyperparameters")
        self.classes = classes
        self.alpha = alpha
        self.margin = margin
        self.proxies = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_uniform_(self.proxies)

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.proxies.shape[1]:
            raise ValueError("Embeddings have an invalid shape")
        if labels.ndim != 1 or len(labels) != len(embeddings):
            raise ValueError("Labels and embeddings have incompatible shapes")
        if len(labels) == 0 or labels.min() < 0 or labels.max() >= self.classes:
            raise ValueError("Proxy Anchor labels are outside the class range")
        embeddings = functional.normalize(embeddings.float(), dim=1)
        proxies = functional.normalize(self.proxies.float(), dim=1)
        cosine = embeddings @ proxies.T
        positive = functional.one_hot(labels.long(), self.classes).bool()
        negative = ~positive
        positive_logits = -self.alpha * (cosine - self.margin)
        negative_logits = self.alpha * (cosine + self.margin)
        positive_logits = positive_logits.masked_fill(~positive, -torch.inf)
        negative_logits = negative_logits.masked_fill(~negative, -torch.inf)
        positive_present = positive.any(dim=0)
        positive_term = _log1p_sum_exp(positive_logits, dim=0)
        positive_loss = positive_term[positive_present].mean()
        negative_loss = _log1p_sum_exp(negative_logits, dim=0).mean()
        loss = positive_loss + negative_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Proxy Anchor loss is not finite")
        return loss
