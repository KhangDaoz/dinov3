from pathlib import Path

import torch

from uncertainty_retrieval.utils import (
    configure_runtime_environment,
    distributed_barrier,
)


def test_runtime_cache_uses_writable_configured_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache = tmp_path / "torch-kernels"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache))
    configure_runtime_environment()
    assert cache.is_dir()


def test_barrier_is_noop_without_process_group(monkeypatch) -> None:
    called = False

    def unexpected_barrier(*args, **kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "barrier", unexpected_barrier)
    distributed_barrier(torch.device("cpu"))
    assert not called


def test_cuda_barrier_receives_explicit_device(monkeypatch) -> None:
    received = {}
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda **kwargs: received.update(kwargs),
    )
    distributed_barrier(torch.device("cuda", 1))
    assert received == {"device_ids": [1]}
