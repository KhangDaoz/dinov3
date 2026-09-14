"""Portable CPU feature-cache persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


REQUIRED_CACHE_KEYS = {
    "schema_version",
    "features",
    "image_ids",
    "labels",
    "splits",
    "metadata",
}


def save_feature_cache(payload: dict[str, Any], path: str | Path) -> None:
    """Validate and atomically save a CPU-only tensor cache."""
    missing = REQUIRED_CACHE_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Feature cache missing keys: {sorted(missing)}")
    if not isinstance(payload["features"], torch.Tensor):
        raise TypeError("features must be a tensor")
    lengths = {
        payload["features"].shape[0],
        len(payload["image_ids"]),
        len(payload["labels"]),
        len(payload["splits"]),
    }
    if len(lengths) != 1:
        raise ValueError("Feature cache fields have inconsistent lengths")
    serializable = dict(payload)
    serializable["features"] = payload["features"].detach().cpu().contiguous()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(serializable, temporary)
    temporary.replace(destination)


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    """Load and validate a portable feature cache on CPU."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Feature cache must contain a dictionary")
    missing = REQUIRED_CACHE_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Feature cache missing keys: {sorted(missing)}")
    if payload["features"].device.type != "cpu":
        raise ValueError("Persisted features must load on CPU")
    return payload
