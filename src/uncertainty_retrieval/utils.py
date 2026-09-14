"""Reproducibility, device, and run metadata helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


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


def initialize_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize NCCL/Gloo when launched with multiple processes."""
    rank, world_size, local_rank = distributed_context()
    device = resolve_device(local_rank)
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend)
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

