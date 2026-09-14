from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import spawn

from uncertainty_retrieval.training.representation import M3TrainingModel


def _global_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
    patch = torch.tensor([[0.9, 0.1], [1.0, 0.0], [0.1, 0.9], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    return cls, patch, labels


def _ddp_worker(rank: int, init_file: str, output: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(7)
        model = M3TrainingModel(2, 2, alpha=2.0, margin=0.1)
        ddp = DistributedDataParallel(model)
        cls, patch, labels = _global_batch()
        local = slice(rank * 2, (rank + 1) * 2)
        loss = ddp(cls[local], patch[local], labels[local])
        loss.backward()
        if rank == 0:
            torch.save(
                {
                    "loss": loss.detach(),
                    "weight_grad": model.representation.projection.weight.grad,
                    "bias_grad": model.representation.projection.bias.grad,
                    "proxy_grad": model.objective.proxies.grad,
                },
                output,
            )
    finally:
        torch.distributed.destroy_process_group()


def test_ddp_differentiable_gather_matches_single_global_batch(tmp_path: Path) -> None:
    init_file = tmp_path / "gloo_init"
    output = tmp_path / "ddp_gradients.pt"
    spawn(
        _ddp_worker,
        args=(str(init_file), str(output)),
        nprocs=2,
        join=True,
    )
    distributed = torch.load(output, map_location="cpu", weights_only=True)

    torch.manual_seed(7)
    reference = M3TrainingModel(2, 2, alpha=2.0, margin=0.1)
    cls, patch, labels = _global_batch()
    loss = reference(cls, patch, labels)
    loss.backward()
    assert torch.allclose(distributed["loss"], loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        distributed["weight_grad"],
        reference.representation.projection.weight.grad,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        distributed["bias_grad"],
        reference.representation.projection.bias.grad,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        distributed["proxy_grad"],
        reference.objective.proxies.grad,
        atol=1e-5,
        rtol=1e-5,
    )
