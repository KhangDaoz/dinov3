import torch
import torch.nn.functional as F


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
