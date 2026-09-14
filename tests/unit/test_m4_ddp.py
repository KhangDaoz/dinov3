from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import spawn

from uncertainty_retrieval.training.representation import M4TrainingModel


def _batch():
    torch.manual_seed(11)
    return torch.randn(4, 3, 2), torch.tensor([0, 0, 1, 1])


def _worker(rank: int, init_file: str, output: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(7)
        model = M4TrainingModel(2, 2, 3, 2, alpha=2.0, margin=0.1)
        ddp = DistributedDataParallel(model)
        patches, labels = _batch()
        local = slice(rank * 2, (rank + 1) * 2)
        loss = ddp(patches[local], labels[local])
        loss.backward()
        if rank == 0:
            torch.save(
                {
                    "loss": loss.detach(),
                    "attention": {
                        key: value.grad.detach()
                        for key, value in model.representation.named_parameters()
                    },
                    "proxies": model.objective.proxies.grad.detach(),
                },
                output,
            )
    finally:
        torch.distributed.destroy_process_group()


def test_m4_ddp_gather_matches_single_global_batch(tmp_path: Path) -> None:
    output = tmp_path / "ddp.pt"
    spawn(
        _worker,
        args=(str(tmp_path / "init"), str(output)),
        nprocs=2,
        join=True,
    )
    distributed = torch.load(output, map_location="cpu", weights_only=True)
    torch.manual_seed(7)
    reference = M4TrainingModel(2, 2, 3, 2, alpha=2.0, margin=0.1)
    patches, labels = _batch()
    loss = reference(patches, labels)
    loss.backward()
    assert torch.allclose(distributed["loss"], loss, atol=1e-6, rtol=1e-6)
    for key, parameter in reference.representation.named_parameters():
        assert torch.allclose(
            distributed["attention"][key], parameter.grad, atol=1e-5, rtol=1e-5
        )
    assert torch.allclose(
        distributed["proxies"], reference.objective.proxies.grad,
        atol=1e-5, rtol=1e-5,
    )
