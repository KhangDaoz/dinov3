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
