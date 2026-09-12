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


def cosine_top_indices(embeddings, top_n, chunk_size, device=None):
    """Return stable same-set cosine neighbors without materializing all scores."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Retrieval yêu cầu CUDA nhưng CUDA không khả dụng")
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n phải là số nguyên dương")
    if embeddings.ndim != 2 or top_n > len(embeddings) - 1:
        raise ValueError("top_n vượt số gallery sau khi loại self-match")
    embeddings = embeddings.detach().to(device=device, dtype=torch.float32)
    dummy_labels = torch.zeros(len(embeddings), dtype=torch.long, device=device)
    validate_retrieval_inputs(embeddings, dummy_labels, [top_n])
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
        raise ValueError("embeddings phải được chuẩn hóa L2 trước khi retrieval")

    batches = []
    for start in range(0, len(embeddings), chunk_size):
        stop = min(start + chunk_size, len(embeddings))
        similarities = embeddings[start:stop] @ embeddings.T
        rows = torch.arange(stop - start, device=device)
        similarities[rows, torch.arange(start, stop, device=device)] = -torch.inf
        indices = torch.argsort(
            similarities, dim=1, descending=True, stable=True
        )[:, :top_n]
        batches.append(indices.cpu())
    return torch.cat(batches)


def rerank_top_n_by_uncertainty(indices, gallery_uncertainty, top_n):
    """Stable-sort the selected cosine prefix by ascending gallery uncertainty."""
    indices = torch.as_tensor(indices, dtype=torch.long).cpu()
    uncertainty = torch.as_tensor(gallery_uncertainty, dtype=torch.float32).cpu()
    if indices.ndim != 2 or uncertainty.ndim != 1:
        raise ValueError("Indices phải là ma trận và uncertainty phải là vector")
    if not 0 < top_n <= indices.shape[1]:
        raise ValueError("top_n không hợp lệ với số candidates")
    if indices.numel() and (indices.min() < 0 or indices.max() >= len(uncertainty)):
        raise ValueError("Candidate index vượt gallery")
    if not torch.isfinite(uncertainty).all().item():
        raise ValueError("Uncertainty chứa NaN/Inf")
    result = indices.clone()
    prefix = result[:, :top_n]
    values = uncertainty[prefix]
    order = torch.argsort(values, dim=1, descending=False, stable=True)
    result[:, :top_n] = torch.gather(prefix, 1, order)
    return result


def recall_from_indices(indices, labels, recall_k):
    """Compute Recall@K from precomputed neighbor indices."""
    indices = torch.as_tensor(indices, dtype=torch.long).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    recall_k = sorted(set(recall_k))
    if indices.ndim != 2 or labels.ndim != 1 or len(indices) != len(labels):
        raise ValueError("Rankings và labels không khớp")
    if not recall_k or min(recall_k) <= 0 or max(recall_k) > indices.shape[1]:
        raise ValueError("recall_k không hợp lệ với rankings")
    if indices.numel() and (indices.min() < 0 or indices.max() >= len(labels)):
        raise ValueError("Ranking index vượt gallery")
    matches = labels[indices].eq(labels[:, None])
    return {
        f"recall@{k}": matches[:, :k].any(dim=1).float().mean().item()
        for k in recall_k
    }
