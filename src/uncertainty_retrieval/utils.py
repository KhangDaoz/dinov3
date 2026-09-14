"""Reproducibility, device, and run metadata helpers."""

from __future__ import annotations

import atexit
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


_DISTRIBUTED_CLEANUP_REGISTERED = False


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed local RNGs and configure deterministic behavior when requested."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def distributed_context() -> tuple[int, int, int]:
    """Return global rank, world size, and local rank from torchrun."""
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def resolve_device(local_rank: int = 0) -> torch.device:
    """Prefer the CUDA device assigned by torchrun, with explicit CPU fallback."""
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise ValueError(
                f"LOCAL_RANK={local_rank} exceeds {torch.cuda.device_count()} GPUs"
            )
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def configure_runtime_environment() -> None:
    """Place generated caches in writable locations used by hosted runtimes."""
    cache_directory = Path(
        os.environ.get(
            "TORCHINDUCTOR_CACHE_DIR",
            "/tmp/uncertainty_retrieval_torch_kernels",
        )
    )
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_directory))


def cleanup_distributed() -> None:
    """Release NCCL/Gloo resources on normal interpreter shutdown."""
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.destroy_process_group()


def distributed_barrier(device: torch.device) -> None:
    """Synchronize ranks without relying on NCCL device inference."""
    if not torch.distributed.is_initialized():
        return
    if device.type == "cuda":
        torch.distributed.barrier(device_ids=[device.index])
    else:
        torch.distributed.barrier()


def initialize_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize NCCL/Gloo when launched with multiple processes."""
    global _DISTRIBUTED_CLEANUP_REGISTERED

    configure_runtime_environment()
    rank, world_size, local_rank = distributed_context()
    device = resolve_device(local_rank)
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        kwargs = {"backend": backend}
        if device.type == "cuda":
            kwargs["device_id"] = device
        try:
            torch.distributed.init_process_group(**kwargs)
        except TypeError:
            # Compatibility with PyTorch releases before device_id support.
            kwargs.pop("device_id", None)
            torch.distributed.init_process_group(**kwargs)
        if not _DISTRIBUTED_CLEANUP_REGISTERED:
            atexit.register(cleanup_distributed)
            _DISTRIBUTED_CLEANUP_REGISTERED = True
    return rank, world_size, local_rank, device


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def environment_metadata(command: list[str] | None = None) -> dict[str, Any]:
    """Collect lightweight provenance without serializing credentials."""
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "command": command or [],
    }


def write_json(data: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
