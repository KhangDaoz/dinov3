"""Provenance checks and manifests for reusable E2A feature caches."""

from __future__ import annotations

import hashlib
import json
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


def _representation_metadata(config: E2AConfig) -> dict[str, Any]:
    if config.representation.method == "m1":
        return {"token": "cls", "source_layer": "final"}
    if config.representation.method == "m2":
        return {
            "token": "patch",
            "source_layer": "final",
            "pooling": "mean",
            "patch_tokens": config.model.expected_patch_tokens,
            "embedding_dim": config.model.embedding_dim,
        }
    raise ValueError(f"Unsupported E2A method: {config.representation.method}")


def validate_representation_cache(
    payload: dict[str, Any],
    records: list[CUBRecord],
    config: E2AConfig,
    expected_manifest_hash: str,
) -> None:
    """Require exact feature and provenance compatibility for cache reuse."""
    method = config.representation.method.upper()
    features = payload["features"]
    expected_total = (
        config.dataset.expected_development_images
        + config.dataset.expected_test_images
    )
    if features.shape != (expected_total, config.model.embedding_dim):
        raise ValueError(f"Unexpected {method} feature shape: {tuple(features.shape)}")
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise ValueError(f"{method} cache must contain finite FP32 features")
    image_ids = [int(value) for value in payload["image_ids"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{method} cache contains duplicate image IDs")
    expected_records = {record.image_id: record for record in records}
    if set(image_ids) != set(expected_records):
        raise ValueError(f"{method} cache image IDs do not match CUB manifests")
    for image_id, label, split in zip(
        image_ids,
        payload["labels"],
        payload["splits"],
        strict=True,
    ):
        record = expected_records[image_id]
        if int(label) != record.original_label or split != record.split:
            raise ValueError(f"{method} cache label/split drift for image {image_id}")
    metadata = payload.get("metadata", {})
    required_metadata = {
        "model_id",
        "requested_revision",
        "resolved_revision",
        "register_tokens",
        "processor_settings",
        "cub_manifest_hash",
    }
    required_metadata.update(_representation_metadata(config))
    missing = required_metadata - metadata.keys()
    if missing:
        raise ValueError(f"{method} cache provenance missing: {sorted(missing)}")
    expected_metadata = {
        "model_id": config.model.model_id,
        "requested_revision": config.model.revision,
        "resolved_revision": config.model.revision,
        "register_tokens": config.model.register_tokens,
        "cub_manifest_hash": expected_manifest_hash,
    }
    expected_metadata.update(_representation_metadata(config))
    for key, expected in expected_metadata.items():
        if metadata[key] != expected:
            raise ValueError(
                f"{method} cache metadata mismatch for {key}: "
                f"{metadata[key]!r} != {expected!r}"
            )
    if not isinstance(metadata["processor_settings"], dict):
        raise ValueError("processor_settings must be a mapping")
    if config.representation.method == "m2":
        reference_path = Path(config.cache.reference_manifest or "")
        if not config.cache.reference_manifest or not reference_path.is_file():
            raise ValueError("M2 requires the accepted M1 embedding manifest")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_metadata = reference.get("metadata", {})
        shared_keys = (
            "model_id",
            "requested_revision",
            "resolved_revision",
            "register_tokens",
            "processor_settings",
            "cub_manifest_hash",
        )
        mismatched = [
            key
            for key in shared_keys
            if metadata.get(key) != reference_metadata.get(key)
        ]
        if mismatched:
            raise ValueError(
                "M2 provenance differs from accepted M1 for: "
                f"{mismatched}"
            )
    if int(payload["schema_version"]) != config.cache.schema_version:
        raise ValueError(f"{method} cache schema version mismatch")


def validate_m1_cache(
    payload: dict[str, Any],
    records: list[CUBRecord],
    config: E2AConfig,
    expected_manifest_hash: str,
) -> None:
    """Backward-compatible M1 cache validator."""
    if config.representation.method != "m1":
        raise ValueError("validate_m1_cache requires an M1 configuration")
    validate_representation_cache(payload, records, config, expected_manifest_hash)


def find_reusable_feature_cache(
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
            validate_representation_cache(payload, records, config, expected_hash)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            diagnostics[str(path)] = str(error)
            continue
        diagnostics[str(path)] = "accepted"
        return path, diagnostics
    return None, diagnostics


def find_reusable_m1_cache(
    config: E2AConfig,
    records: list[CUBRecord],
) -> tuple[Path | None, dict[str, str]]:
    """Backward-compatible M1 cache lookup."""
    if config.representation.method != "m1":
        raise ValueError("find_reusable_m1_cache requires an M1 configuration")
    return find_reusable_feature_cache(config, records)


def write_embedding_manifest(
    cache_path: str | Path,
    config: E2AConfig,
    output_path: str | Path,
) -> dict[str, Any]:
    path = Path(cache_path)
    payload = load_feature_cache(path)
    manifest = {
        "schema_version": config.cache.schema_version,
        "method": config.representation.method,
        "representation": config.representation.name,
        "normalization": "l2_at_retrieval",
        "cache_path": str(path),
        "cache_sha256": sha256_file(path),
        "shape": list(payload["features"].shape),
        "dtype": str(payload["features"].dtype),
        "metadata": payload["metadata"],
    }
    write_json(manifest, output_path)
    return manifest
