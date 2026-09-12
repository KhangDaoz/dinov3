import torch


def validate_retrieval_inputs(embeddings, labels, recall_k):
    if embeddings.ndim != 2:
        raise ValueError("embeddings phải có shape [N, D]")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels phải có shape [N] và khớp embeddings")
    if embeddings.shape[0] < 2:
        raise ValueError("Cần ít nhất hai mẫu để loại self-match")
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("embeddings chứa NaN/Inf")
    max_neighbors = embeddings.shape[0] - 1
    if max(recall_k) > max_neighbors:
        raise ValueError(f"Recall@K lớn nhất không được vượt {max_neighbors}")


def recall_at_k(embeddings, labels, recall_k, chunk_size, device=None):
    """Evaluate same-set cosine retrieval with deterministic index tie-breaking."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Retrieval yêu cầu CUDA nhưng CUDA không khả dụng")
    embeddings = embeddings.detach().to(device=device, dtype=torch.float32)
    labels = labels.detach().to(device=device, dtype=torch.long)
    recall_k = sorted(set(recall_k))
    validate_retrieval_inputs(embeddings, labels, recall_k)

    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
        raise ValueError("embeddings phải được chuẩn hóa L2 trước khi retrieval")

    num_samples = embeddings.shape[0]
    max_k = max(recall_k)
    hits = {k: 0 for k in recall_k}
    for start in range(0, num_samples, chunk_size):
        stop = min(start + chunk_size, num_samples)
        similarities = embeddings[start:stop] @ embeddings.T
        row_indices = torch.arange(stop - start)
        similarities[row_indices, torch.arange(start, stop)] = -torch.inf
        # Stable sorting makes lower gallery indices win exact similarity ties.
        neighbors = torch.argsort(
            similarities, dim=1, descending=True, stable=True
        )[:, :max_k]
        matches = labels[neighbors].eq(labels[start:stop, None])
        for k in recall_k:
            hits[k] += matches[:, :k].any(dim=1).sum().item()

    return {f"recall@{k}": hits[k] / num_samples for k in recall_k}
