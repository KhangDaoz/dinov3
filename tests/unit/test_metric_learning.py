import torch

from uncertainty_retrieval.models.metric_learning import ProxyAnchorLoss


def test_proxy_anchor_matches_direct_small_equation() -> None:
    module = ProxyAnchorLoss(classes=2, embedding_dim=2, alpha=2.0, margin=0.1)
    with torch.no_grad():
        module.proxies.copy_(torch.eye(2))
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss = module(embeddings, labels)
    positive = torch.log1p(torch.exp(torch.tensor(-2.0 * (1.0 - 0.1))))
    negative = torch.log1p(torch.exp(torch.tensor(2.0 * (0.0 + 0.1))))
    assert torch.allclose(loss, positive + negative)
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()
    assert torch.isfinite(module.proxies.grad).all()


def test_proxy_anchor_supports_classes_absent_from_batch() -> None:
    module = ProxyAnchorLoss(classes=3, embedding_dim=2)
    loss = module(torch.eye(2, requires_grad=True), torch.tensor([0, 1]))
    assert torch.isfinite(loss)


def test_m3_fp32_backward_is_finite_at_locked_proxy_scale() -> None:
    torch.manual_seed(42)
    module = ProxyAnchorLoss(classes=100, embedding_dim=768, alpha=32.0, margin=0.1)
    embeddings = torch.randn(80, 768, dtype=torch.float32, requires_grad=True)
    labels = torch.arange(20).repeat_interleave(4)
    loss = module(embeddings, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(embeddings.grad).all()
    assert torch.isfinite(module.proxies.grad).all()
