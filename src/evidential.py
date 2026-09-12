import torch
import torch.nn.functional as F
from torch import nn


class EvidentialHead(nn.Module):
    """Predict non-negative class evidence from frozen semantic embeddings."""

    def __init__(self, input_dim=768, hidden_dim=256, num_classes=100, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"Evidential input phải có shape [batch, {self.input_dim}]"
            )
        logits = self.network(embeddings.to(dtype=torch.float32))
        evidence = F.softplus(logits).clamp_min(1e-6)
        if not torch.isfinite(evidence).all().item():
            raise ValueError("Evidence chứa NaN/Inf")
        return evidence


def dirichlet_statistics(evidence):
    """Return alpha, strength, expected probabilities, and total uncertainty."""
    if evidence.ndim != 2 or evidence.shape[1] < 2:
        raise ValueError("Evidence phải có shape [batch, classes>=2]")
    if torch.any(evidence < 0).item() or not torch.isfinite(evidence).all().item():
        raise ValueError("Evidence phải hữu hạn và không âm")
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    probabilities = alpha / strength
    uncertainty = evidence.shape[1] / strength.squeeze(1)
    return alpha, strength, probabilities, uncertainty


def kl_dirichlet_uniform(alpha):
    """KL(Dir(alpha) || Dir(1)) for each item in a batch."""
    if alpha.ndim != 2 or torch.any(alpha <= 0).item():
        raise ValueError("Dirichlet alpha phải có shape [batch, classes] và dương")
    num_classes = alpha.shape[1]
    strength = alpha.sum(dim=1)
    log_normalizer = (
        torch.lgamma(strength)
        - torch.lgamma(alpha).sum(dim=1)
        - torch.lgamma(alpha.new_tensor(float(num_classes)))
    )
    expectation = (
        (alpha - 1.0)
        * (torch.digamma(alpha) - torch.digamma(strength).unsqueeze(1))
    ).sum(dim=1)
    return log_normalizer + expectation


def annealing_coefficient(epoch, annealing_epochs):
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch phải là số nguyên không âm")
    if not isinstance(annealing_epochs, int) or annealing_epochs <= 0:
        raise ValueError("annealing_epochs phải là số nguyên dương")
    return min(1.0, epoch / annealing_epochs)


def edl_mse_loss(evidence, labels, annealing=1.0):
    """EDL Bayes-risk MSE with annealed KL regularization."""
    if labels.ndim != 1 or labels.shape[0] != evidence.shape[0]:
        raise ValueError("Labels không khớp evidence")
    if not 0.0 <= annealing <= 1.0:
        raise ValueError("annealing phải nằm trong [0, 1]")
    num_classes = evidence.shape[1]
    if labels.numel() == 0 or labels.min().item() < 0:
        raise ValueError("Labels không hợp lệ")
    if labels.max().item() >= num_classes:
        raise ValueError("Label vượt số lớp evidential")

    alpha, strength, probabilities, _ = dirichlet_statistics(evidence)
    targets = F.one_hot(labels.long(), num_classes=num_classes).to(alpha.dtype)
    error = (targets - probabilities).square()
    variance = alpha * (strength - alpha) / (
        strength.square() * (strength + 1.0)
    )
    data_loss = (error + variance).sum(dim=1)
    adjusted_alpha = targets + (1.0 - targets) * alpha
    loss = data_loss + annealing * kl_dirichlet_uniform(adjusted_alpha)
    if not torch.isfinite(loss).all().item():
        raise FloatingPointError("EDL loss chứa NaN/Inf")
    return loss.mean()


def classification_metrics(probabilities, labels, bins=15):
    """Compute deterministic validation calibration metrics."""
    if probabilities.ndim != 2 or labels.shape != (len(probabilities),):
        raise ValueError("Probabilities và labels không khớp")
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins phải là số nguyên dương")
    probabilities = probabilities.float()
    labels = labels.long()
    if not torch.isfinite(probabilities).all().item():
        raise ValueError("Probabilities chứa NaN/Inf")
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    nll = -probabilities[
        torch.arange(len(labels), device=labels.device), labels
    ].clamp_min(1e-12).log().mean()
    targets = F.one_hot(labels, probabilities.shape[1]).to(probabilities.dtype)
    brier = (probabilities - targets).square().sum(dim=1).mean()
    ece = probabilities.new_zeros(())
    boundaries = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        if mask.any():
            gap = confidence[mask].mean() - correct[mask].float().mean()
            ece = ece + mask.float().mean() * gap.abs()
    return {
        "accuracy": correct.float().mean().item(),
        "negative_log_likelihood": nll.item(),
        "brier_score": brier.item(),
        "ece": ece.item(),
    }


def binary_ranking_metrics(scores, targets):
    """AUROC and AUPRC for scores where larger means more likely positive."""
    scores = torch.as_tensor(scores, dtype=torch.float64).cpu()
    targets = torch.as_tensor(targets, dtype=torch.bool).cpu()
    if scores.ndim != 1 or targets.shape != scores.shape:
        raise ValueError("Scores và targets phải là vector cùng shape")
    if not torch.isfinite(scores).all().item():
        raise ValueError("Scores chứa NaN/Inf")
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return {"auroc": None, "auprc": None}

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    group_end = torch.ones(len(scores), dtype=torch.bool)
    group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    tp = sorted_targets.cumsum(0).double()[group_end]
    fp = (~sorted_targets).cumsum(0).double()[group_end]
    tpr = torch.cat((torch.zeros(1), tp / positives))
    fpr = torch.cat((torch.zeros(1), fp / negatives))
    auroc = torch.trapz(tpr, fpr).item()
    recall = tp / positives
    previous_recall = torch.cat((torch.zeros(1), recall[:-1]))
    precision = tp / (tp + fp)
    auprc = ((recall - previous_recall) * precision).sum().item()
    return {"auroc": auroc, "auprc": auprc}


def risk_coverage_metrics(uncertainty, errors, coverage_points=(0.25, 0.5, 0.75, 1.0)):
    """Compute selective risk after retaining least-uncertain queries."""
    uncertainty = torch.as_tensor(uncertainty, dtype=torch.float64).cpu()
    errors = torch.as_tensor(errors, dtype=torch.float64).cpu()
    if uncertainty.ndim != 1 or errors.shape != uncertainty.shape:
        raise ValueError("Uncertainty và errors phải là vector cùng shape")
    if len(errors) == 0 or not torch.isfinite(uncertainty).all().item():
        raise ValueError("Risk-coverage input không hợp lệ")
    order = torch.argsort(uncertainty, stable=True)
    cumulative_risk = errors[order].cumsum(0) / torch.arange(
        1, len(errors) + 1, dtype=torch.float64
    )
    points = {}
    for coverage in coverage_points:
        if not 0 < coverage <= 1:
            raise ValueError("Coverage phải nằm trong (0, 1]")
        count = max(1, int(round(coverage * len(errors))))
        points[f"{coverage:.2f}"] = cumulative_risk[count - 1].item()
    return {"aurc": cumulative_risk.mean().item(), "risk_at_coverage": points}


def select_evidential_epoch(history):
    if not history:
        raise ValueError("Training history rỗng")
    best = min(
        history,
        key=lambda row: (
            row["validation_loss"],
            -row["validation_accuracy"],
            row["epoch"],
        ),
    )
    return int(best["epoch"])


def select_top_n(results):
    """Select by R@1, then R@2, then the smaller top-N."""
    if not results:
        raise ValueError("Top-N validation results rỗng")
    best = min(
        results,
        key=lambda row: (
            -row["metrics"]["recall@1"],
            -row["metrics"].get("recall@2", 0.0),
            row["top_n"],
        ),
    )
    return int(best["top_n"])
