"""Strict configuration for E2A representation pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_type_hints

import yaml


@dataclass(frozen=True)
class E2ADatasetConfig:
    root: str = "data/CUB_200_2011"
    validation_fraction: float = 0.2
    split_seed: int = 42
    development_classes: int = 100
    total_classes: int = 200
    expected_development_images: int = 5864
    expected_test_images: int = 5924


@dataclass(frozen=True)
class E2AModelConfig:
    model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    revision: str = "5931719e67bbdb9737e363e781fb0c67687896bc"
    embedding_dim: int = 768
    register_tokens: int = 4
    expected_patch_tokens: int = 196


@dataclass(frozen=True)
class RepresentationConfig:
    method: str = "m1"
    name: str = "cls"
    source_layer: str = "final"
    pooling: str = "none"
    normalization: str = "l2"
    attention_hidden_dim: int | None = None


@dataclass(frozen=True)
class E2ATrainingConfig:
    seed: int = 42
    epochs: int = 30
    classes_per_batch: int = 20
    images_per_class: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    proxy_weight_decay: float = 0.0
    proxy_alpha: float = 32.0
    proxy_margin: float = 0.1
    gradient_clip_norm: float = 5.0


@dataclass(frozen=True)
class E2ARuntimeConfig:
    batch_size: int = 128
    num_workers: int = 4
    prefetch_factor: int = 2
    amp: bool = True
    world_size: int = 2
    similarity_chunk_size: int = 1024


@dataclass(frozen=True)
class E2AEvaluationConfig:
    recall_k: tuple[int, ...] = (1, 2, 4, 8)
    ranking_depth: int = 100
    self_match_exclusion: bool = True
    tie_policy: str = "stable_gallery_index"
    reference_validation_ids: str = (
        "outputs/e2a_cls/m1/split/validation_image_ids.pt"
    )


@dataclass(frozen=True)
class E2ACacheConfig:
    path: str = "outputs/e2a_cls/cache/dinov3_cls.pt"
    reuse_candidates: tuple[str, ...] = (
        "outputs/e1_evidential/cache/dinov3_cls.pt",
    )
    schema_version: int = 2
    materialize_embeddings: bool = False
    reference_manifest: str | None = None
    cls_path: str | None = None
    mean_patch_path: str | None = None
    cls_manifest: str | None = None
    mean_patch_manifest: str | None = None
    shard_directory: str | None = None
    patch_manifest: str | None = None
    storage_dtype: str | None = None
    extraction_amp: bool | None = None
    fidelity_samples: int | None = None
    fidelity_relative_l2_max: float | None = None
    fidelity_patch_cosine_min: float | None = None
    fidelity_mean_cosine_min: float | None = None


@dataclass(frozen=True)
class E2AOutputConfig:
    root: str = "outputs/e2a_cls/m1"
    # Retained for compatibility with hashes of accepted M3/M4 checkpoints.
    # Test execution no longer depends on this legacy lock path.
    selection_lock: str = "outputs/e2a_selection/selection_lock.json"


@dataclass(frozen=True)
class E2AConfig:
    dataset: E2ADatasetConfig = E2ADatasetConfig()
    model: E2AModelConfig = E2AModelConfig()
    representation: RepresentationConfig = RepresentationConfig()
    runtime: E2ARuntimeConfig = E2ARuntimeConfig()
    evaluation: E2AEvaluationConfig = E2AEvaluationConfig()
    cache: E2ACacheConfig = E2ACacheConfig()
    output: E2AOutputConfig = E2AOutputConfig()
    training: E2ATrainingConfig | None = None


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any], path: str) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"Unknown configuration key(s) at {path}: "
            + ", ".join(sorted(unknown))
        )
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in values:
            continue
        value = values[field.name]
        field_type = hints[field.name]
        union_dataclass = next(
            (item for item in get_args(field_type) if is_dataclass(item)), None
        )
        if is_dataclass(field_type):
            if not isinstance(value, dict):
                raise TypeError(f"{path}.{field.name} must be a mapping")
            value = _construct(field_type, value, f"{path}.{field.name}")
        elif union_dataclass is not None and value is not None:
            if not isinstance(value, dict):
                raise TypeError(f"{path}.{field.name} must be a mapping")
            value = _construct(
                union_dataclass, value, f"{path}.{field.name}"
            )
        elif getattr(field_type, "__origin__", None) is tuple:
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{path}.{field.name} must be a sequence")
            value = tuple(value)
        kwargs[field.name] = value
    return cls(**kwargs)


def validate_e2a_config(config: E2AConfig) -> None:
    """Reject settings that would change an E2A treatment or protocol."""
    dataset = config.dataset
    if not 0.0 < dataset.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if dataset.split_seed != 42:
        raise ValueError("E2A uses the canonical validation seed 42")
    if dataset.development_classes != 100 or dataset.total_classes != 200:
        raise ValueError("E2A requires the fixed 100/100 CUB class split")
    if (
        dataset.expected_development_images != 5864
        or dataset.expected_test_images != 5924
    ):
        raise ValueError("E2A requires the fixed CUB image counts")
    if (
        config.model.embedding_dim != 768
        or config.model.register_tokens != 4
        or config.model.expected_patch_tokens != 196
    ):
        raise ValueError("E2A requires DINOv3 ViT-B/16 token dimensions")
    if not config.model.revision or config.model.revision == "main":
        raise ValueError("model.revision must be an immutable revision")
    representation = config.representation
    contracts = {
        "m1": ("m1", "cls", "final", "none", "l2", None),
        "m2": ("m2", "mean_patch", "final", "mean", "l2", None),
        "m3": (
            "m3",
            "cls_mean_projection",
            "final",
            "projection",
            "l2",
            None,
        ),
        "m4": ("m4", "attention_pool", "final_patch", "attention", "l2", 256),
    }
    actual = (
        representation.method,
        representation.name,
        representation.source_layer,
        representation.pooling,
        representation.normalization,
        representation.attention_hidden_dim,
    )
    expected = contracts.get(representation.method)
    if expected is None or actual != expected:
        raise ValueError(
            f"Invalid {representation.method.upper()} representation contract: "
            f"{actual}"
        )
    runtime = config.runtime
    if runtime.batch_size <= 0 or runtime.similarity_chunk_size <= 0:
        raise ValueError("Runtime batch and chunk sizes must be positive")
    if runtime.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if runtime.num_workers > 0 and runtime.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive with workers")
    if runtime.world_size != 2:
        raise ValueError("Target runtime requires two GPU processes")
    evaluation = config.evaluation
    if evaluation.recall_k != (1, 2, 4, 8):
        raise ValueError("E2A requires Recall@1/2/4/8")
    if evaluation.ranking_depth < max(evaluation.recall_k):
        raise ValueError("ranking_depth is smaller than requested Recall@K")
    if evaluation.ranking_depth != 100:
        raise ValueError("E2A downstream contract requires Top-100")
    if not evaluation.self_match_exclusion:
        raise ValueError("E2A requires self-match exclusion")
    if evaluation.tie_policy != "stable_gallery_index":
        raise ValueError("Unsupported tie policy")
    if config.cache.schema_version <= 0:
        raise ValueError("cache.schema_version must be positive")
    if representation.method == "m2":
        if config.cache.reuse_candidates:
            raise ValueError("M2 cannot reuse CLS cache candidates")
        if not config.cache.reference_manifest:
            raise ValueError("M2 requires the accepted M1 embedding manifest")
    if representation.method in {"m1", "m2"} and config.training is not None:
        raise ValueError(f"{representation.method.upper()} is not trainable")
    if representation.method == "m3":
        training = config.training
        if training is None:
            raise ValueError("M3 requires a training section")
        expected_training = E2ATrainingConfig()
        if training != expected_training:
            raise ValueError("M3 training recipe differs from the locked plan")
        required_inputs = (
            config.cache.cls_path,
            config.cache.mean_patch_path,
            config.cache.cls_manifest,
            config.cache.mean_patch_manifest,
        )
        if any(value is None for value in required_inputs):
            raise ValueError("M3 requires both accepted M1/M2 caches and manifests")
        if config.runtime.amp:
            raise ValueError(
                "M3 requires FP32 training because Proxy Anchor AMP gradients "
                "overflow on T4"
            )
    if representation.method == "m4":
        if config.training != E2ATrainingConfig():
            raise ValueError("M4 training recipe differs from the locked plan")
        if config.runtime.amp:
            raise ValueError("M4 attention and Proxy Anchor training require FP32")
        cache = config.cache
        required = (
            cache.shard_directory,
            cache.patch_manifest,
            cache.storage_dtype,
            cache.extraction_amp,
            cache.fidelity_samples,
            cache.fidelity_relative_l2_max,
            cache.fidelity_patch_cosine_min,
            cache.fidelity_mean_cosine_min,
            cache.reference_manifest,
        )
        if any(value is None for value in required):
            raise ValueError("M4 requires patch-cache and fidelity settings")
        expected_fidelity = ("float16", True, 128, 1e-3, 0.99999, 0.99999)
        actual_fidelity = (
            cache.storage_dtype,
            cache.extraction_amp,
            cache.fidelity_samples,
            cache.fidelity_relative_l2_max,
            cache.fidelity_patch_cosine_min,
            cache.fidelity_mean_cosine_min,
        )
        if actual_fidelity != expected_fidelity:
            raise ValueError("M4 FP16 fidelity contract differs from the plan")


def load_e2a_config(path: str | Path) -> E2AConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise TypeError("Configuration root must be a mapping")
    config = _construct(E2AConfig, raw, "config")
    validate_e2a_config(config)
    return config


def save_e2a_config(config: E2AConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(asdict(config), stream, sort_keys=False)
