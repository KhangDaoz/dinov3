"""Evidential classification head and EDL loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@dataclass(frozen=True)
class EvidentialOutput:
    """Full evidential quantities retained for E2 reuse."""

    evidence: Tensor
    alpha: Tensor
    probabilities: Tensor
    uncertainty: Tensor


class EvidentialHead(nn.Module):
    """Map frozen DINOv3 CLS features to Dirichlet evidence."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        architecture: str = "linear",
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError("input_dim and num_classes must be valid")
        if architecture == "linear":
            layers: list[nn.Module] = [
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, num_classes),
            ]
        elif architecture == "mlp":
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            layers = [
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_classes),
            ]
        else:
            raise ValueError(f"Unsupported head architecture: {architecture}")
        self.network = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, features: Tensor) -> EvidentialOutput:
        logits = self.network(features)
        evidence = functional.softplus(logits.float())
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        probabilities = alpha / strength
        uncertainty = self.num_classes / strength.squeeze(-1)
        if not all(
            torch.isfinite(value).all()
            for value in (evidence, alpha, probabilities, uncertainty)
        ):
            raise FloatingPointError("Non-finite evidential output")
        return EvidentialOutput(evidence, alpha, probabilities, uncertainty)


def dirichlet_kl_to_uniform(alpha: Tensor) -> Tensor:
    """KL[Dir(alpha) || Dir(1)] for each batch element in FP32."""
    alpha = alpha.float()
    num_classes = alpha.shape[-1]
    strength = alpha.sum(dim=-1)
    log_normalizer = (
        torch.lgamma(strength)
        - torch.lgamma(alpha).sum(dim=-1)
        - torch.lgamma(
            torch.tensor(
                float(num_classes),
                device=alpha.device,
                dtype=alpha.dtype,
            )
        )
    )
    expectation = (
        (alpha - 1.0)
        * (torch.digamma(alpha) - torch.digamma(strength).unsqueeze(-1))
    ).sum(dim=-1)
    return log_normalizer + expectation


def edl_mse_loss(
    alpha: Tensor,
    targets: Tensor,
    epoch: int,
    annealing_epochs: int,
    kl_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Expected squared-error Bayes risk plus annealed EDL KL."""
    if alpha.ndim != 2:
        raise ValueError("alpha must have shape [batch, classes]")
    if targets.shape != (alpha.shape[0],):
        raise ValueError("targets must have shape [batch]")
    if annealing_epochs <= 0 or epoch < 0:
        raise ValueError("Invalid annealing schedule")
    alpha = alpha.float()
    one_hot = functional.one_hot(targets, alpha.shape[-1]).float()
    strength = alpha.sum(dim=-1, keepdim=True)
    probabilities = alpha / strength
    error = (one_hot - probabilities).square()
    variance = alpha * (strength - alpha) / (
        strength.square() * (strength + 1.0)
    )
    data_loss = (error + variance).sum(dim=-1)
    adjusted_alpha = one_hot + (1.0 - one_hot) * alpha
    kl = dirichlet_kl_to_uniform(adjusted_alpha)
    annealing = min(1.0, epoch / annealing_epochs)
    total = data_loss + kl_weight * annealing * kl
    return total.mean(), {
        "data_loss": data_loss.mean().detach(),
        "kl_loss": kl.mean().detach(),
        "annealing": torch.tensor(annealing, device=alpha.device),
    }

