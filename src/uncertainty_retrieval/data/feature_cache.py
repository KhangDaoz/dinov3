"""Provenance checks and manifests for reusable E2A feature caches."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch

from uncertainty_retrieval.config_e2a import E2AConfig
from uncertainty_retrieval.data.cub import CUBRecord
from uncertainty_retrieval.data.patch_cache import load_feature_cache
from uncertainty_retrieval.utils import write_json


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cub_manifest_hash(root: str | Path) -> str:
    """Hash the manifests that define image identity and class membership."""
    root_path = Path(root)
    digest = hashlib.sha256()
    for name in ("images.txt", "image_class_labels.txt"):
        digest.update(name.encode("utf-8"))
        with (root_path / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(b"class_split:0-99/development,100-199/test")
    return digest.hexdigest()


def validation_ids_hash(image_ids: Iterable[int]) -> str:
    canonical = "\n".join(str(value) for value in sorted(image_ids)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_m1_cache(
    payload: dict[str, Any],
    records: list[CUBRecord],
    config: E2AConfig,
    expected_manifest_hash: str,
) -> None:
    """Require exact feature and provenance compatibility for M1 reuse."""
    features = payload["features"]
    expected_total = (
        config.dataset.expected_development_images
        + config.dataset.expected_test_images
    )
    if features.shape != (expected_total, config.model.embedding_dim):
        raise ValueError(f"Unexpected M1 feature shape: {tuple(features.shape)}")
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise ValueError("M1 cache must contain finite FP32 features")
    image_ids = [int(value) for value in payload["image_ids"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("M1 cache contains duplicate image IDs")
    expected_records = {record.image_id: record for record in records}
    if set(image_ids) != set(expected_records):
        raise ValueError("M1 cache image IDs do not match CUB manifests")
    for image_id, label, split in zip(
        image_ids,
        payload["labels"],
        payload["splits"],
        strict=True,
    ):
        record = expected_records[image_id]
        if int(label) != record.original_label or split != record.split:
            raise ValueError(f"M1 cache label/split drift for image {image_id}")
    metadata = payload.get("metadata", {})
    required_metadata = {
        "model_id",
        "requested_revision",
        "resolved_revision",
        "token",
        "source_layer",
        "register_tokens",
        "processor_settings",
        "cub_manifest_hash",
    }
    missing = required_metadata - metadata.keys()
    if missing:
        raise ValueError(f"M1 cache provenance missing: {sorted(missing)}")
    expected_metadata = {
        "model_id": config.model.model_id,
        "requested_revision": config.model.revision,
        "resolved_revision": config.model.revision,
        "token": "cls",
        "source_layer": "final",
        "register_tokens": config.model.register_tokens,
        "cub_manifest_hash": expected_manifest_hash,
    }
    for key, expected in expected_metadata.items():
        if metadata[key] != expected:
            raise ValueError(
                f"M1 cache metadata mismatch for {key}: "
                f"{metadata[key]!r} != {expected!r}"
            )
    if not isinstance(metadata["processor_settings"], dict):
        raise ValueError("processor_settings must be a mapping")
    if int(payload["schema_version"]) != config.cache.schema_version:
        raise ValueError("M1 cache schema version mismatch")


def find_reusable_m1_cache(
    config: E2AConfig,
    records: list[CUBRecord],
) -> tuple[Path | None, dict[str, str]]:
    """Return the first strictly compatible cache and rejection diagnostics."""
    expected_hash = cub_manifest_hash(config.dataset.root)
    candidates = [Path(config.cache.path)]
    candidates.extend(Path(path) for path in config.cache.reuse_candidates)
    diagnostics: dict[str, str] = {}
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            diagnostics[str(path)] = "missing"
            continue
        try:
            payload = load_feature_cache(path)
            validate_m1_cache(payload, records, config, expected_hash)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            diagnostics[str(path)] = str(error)
            continue
        diagnostics[str(path)] = "accepted"
        return path, diagnostics
    return None, diagnostics


def write_embedding_manifest(
    cache_path: str | Path,
    config: E2AConfig,
    output_path: str | Path,
) -> dict[str, Any]:
    path = Path(cache_path)
    payload = load_feature_cache(path)
    manifest = {
        "schema_version": config.cache.schema_version,
        "method": "m1",
        "representation": "final_cls",
        "normalization": "l2_at_retrieval",
        "cache_path": str(path),
        "cache_sha256": sha256_file(path),
        "shape": list(payload["features"].shape),
        "dtype": str(payload["features"].dtype),
        "metadata": payload["metadata"],
    }
    write_json(manifest, output_path)
    return manifest

