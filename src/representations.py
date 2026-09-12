import torch
import torch.nn.functional as F
from torch import nn


def cls_embedding(last_hidden_state):
    """Select final-layer CLS tokens and return CPU float32 unit vectors."""
    if last_hidden_state.ndim != 3 or last_hidden_state.shape[1] < 1:
        raise ValueError(
            "last_hidden_state phải có shape [batch, tokens, hidden_size]"
        )
    cls_tokens = last_hidden_state[:, 0, :].to(dtype=torch.float32)
    if not torch.isfinite(cls_tokens).all().item():
        raise ValueError("CLS tokens chứa NaN/Inf")
    embeddings = F.normalize(cls_tokens, p=2, dim=1)
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("CLS embeddings chuẩn hóa chứa NaN/Inf")
    return embeddings.cpu()


def mean_patch_embedding(last_hidden_state, num_register_tokens):
    """Mean-pool spatial patches, excluding CLS and all register tokens."""
    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state phải có shape [batch, tokens, hidden_size]"
        )
    if not isinstance(num_register_tokens, int) or num_register_tokens < 0:
        raise ValueError("num_register_tokens phải là số nguyên không âm")
    patch_start = 1 + num_register_tokens
    if patch_start >= last_hidden_state.shape[1]:
        raise ValueError("Không còn patch token sau khi loại CLS/register tokens")
    patch_tokens = last_hidden_state[:, patch_start:, :].to(dtype=torch.float32)
    if not torch.isfinite(patch_tokens).all().item():
        raise ValueError("Patch tokens chứa NaN/Inf")
    pooled = patch_tokens.mean(dim=1)
    if torch.any(torch.linalg.vector_norm(pooled, dim=1) == 0).item():
        raise ValueError("Mean Patch tạo vector zero, không thể chuẩn hóa L2")
    embeddings = F.normalize(pooled, p=2, dim=1)
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("Mean Patch embeddings chứa NaN/Inf")
    return embeddings.cpu()


def build_embedding(representation, last_hidden_state, num_register_tokens=0):
    """Dispatch one supported frozen-backbone representation."""
    if representation == "cls":
        return cls_embedding(last_hidden_state)
    if representation == "mean_patch":
        return mean_patch_embedding(last_hidden_state, num_register_tokens)
    raise ValueError(f"Representation không được hỗ trợ: {representation!r}")


def raw_fusion_features(last_hidden_state, num_register_tokens):
    """Return unnormalized float32 CLS and Mean Patch features on CPU."""
    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state phải có shape [batch, tokens, hidden_size]"
        )
    if not isinstance(num_register_tokens, int) or num_register_tokens < 0:
        raise ValueError("num_register_tokens phải là số nguyên không âm")
    patch_start = 1 + num_register_tokens
    if patch_start >= last_hidden_state.shape[1]:
        raise ValueError("Không còn patch token cho fusion")
    tokens = last_hidden_state.to(dtype=torch.float32)
    cls = tokens[:, 0, :]
    mean_patch = tokens[:, patch_start:, :].mean(dim=1)
    if not torch.isfinite(cls).all().item() or not torch.isfinite(mean_patch).all().item():
        raise ValueError("Raw fusion features chứa NaN/Inf")
    return cls.cpu(), mean_patch.cpu()


class FusionProjection(nn.Module):
    def __init__(self, feature_dim=768, projection_dim=768):
        super().__init__()
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.projection = nn.Linear(feature_dim * 2, projection_dim, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, cls_features, mean_patch_features):
        if cls_features.shape != mean_patch_features.shape:
            raise ValueError("CLS và Mean Patch features phải cùng shape")
        if cls_features.ndim != 2 or cls_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Fusion input phải có shape [batch, {self.feature_dim}]"
            )
        fused = torch.cat((cls_features, mean_patch_features), dim=1)
        projected = self.projection(fused)
        if torch.any(torch.linalg.vector_norm(projected, dim=1) == 0).item():
            raise ValueError("Fusion projection tạo vector zero")
        return F.normalize(projected, p=2, dim=1)
