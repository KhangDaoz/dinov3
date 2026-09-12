import torch
import torch.nn.functional as F
from torch import nn


def build_pair_features(first, second, mode="full"):
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Pair embeddings phải là hai ma trận cùng shape")
    if mode == "full":
        return torch.cat((first, second, (first - second).abs()), dim=1)
    if mode == "ordered":
        return torch.cat((first, second), dim=1)
    if mode == "difference":
        return (first - second).abs()
    raise ValueError(f"Pair feature mode không hỗ trợ: {mode}")


class PairConfidenceNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        hidden_dims=(512, 128),
        dropout=0.1,
        feature_mode="full",
    ):
        super().__init__()
        multipliers = {"full": 3, "ordered": 2, "difference": 1}
        if feature_mode not in multipliers:
            raise ValueError("feature_mode không hợp lệ")
        self.embedding_dim = embedding_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.feature_mode = feature_mode
        input_dim = embedding_dim * multipliers[feature_mode]
        layers = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dims[1], 1),
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, first, second):
        features = build_pair_features(first, second, self.feature_mode)
        logits = self.network(features.to(dtype=torch.float32)).squeeze(1)
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Pair logits chứa NaN/Inf")
        return logits

    def symmetric_confidence(self, first, second):
        forward = torch.sigmoid(self(first, second))
        backward = torch.sigmoid(self(second, first))
        return (forward + backward) / 2


def pair_bce_loss(logits, targets):
    if logits.ndim != 1 or targets.shape != logits.shape:
        raise ValueError("Pair logits và targets phải là vector cùng shape")
    loss = F.binary_cross_entropy_with_logits(logits, targets.float())
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Pair BCE chứa NaN/Inf")
    return loss


def combine_scores(cosine, confidence, weight):
    if cosine.shape != confidence.shape:
        raise ValueError("Cosine và confidence phải cùng shape")
    if not 0 <= weight <= 1:
        raise ValueError("lambda phải nằm trong [0, 1]")
    normalized_cosine = (cosine + 1.0) / 2.0
    return weight * normalized_cosine + (1.0 - weight) * confidence


def select_pair_epoch(history):
    if not history:
        raise ValueError("Pair training history rỗng")
    best = min(
        history,
        key=lambda row: (
            -row["validation_auroc"],
            row["validation_bce"],
            row["epoch"],
        ),
    )
    return int(best["epoch"])


def select_retrieval_parameters(results):
    if not results:
        raise ValueError("Retrieval validation results rỗng")
    best = min(
        results,
        key=lambda row: (
            -row["metrics"]["recall@1"],
            -row["metrics"].get("recall@2", 0.0),
            -row["metrics"].get("recall@4", 0.0),
            -row["lambda"],
            row["candidate_top_n"],
        ),
    )
    return {
        "candidate_top_n": int(best["candidate_top_n"]),
        "lambda": float(best["lambda"]),
    }
