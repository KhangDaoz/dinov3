"""Strict configuration loading for E1 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    root: str = "data/CUB_200_2011"
    validation_fraction: float = 0.2
    development_classes: int = 100
    total_classes: int = 200
    expected_development_images: int = 5864
    expected_test_images: int = 5924


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    revision: str = "main"
    embedding_dim: int = 768
    register_tokens: int = 4
    evidence_activation: str = "softplus"
    head: str = "linear"
    hidden_dim: int = 512


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    seeds: tuple[int, ...] = (42, 43, 44)
    epochs: int = 30
    annealing_epochs: int = 10
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    batch_size: int = 256
    num_workers: int = 4
    prefetch_factor: int = 2
    amp: bool = True
    deterministic: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    recall_k: tuple[int, ...] = (1, 2, 4, 8)
    similarity_chunk_size: int = 1024
    top_n_grid: tuple[int, ...] = (10, 20, 50, 100)
    beta_grid: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75)
    bootstrap_samples: int = 2000
    self_match_exclusion: bool = True


@dataclass(frozen=True)
class OutputConfig:
    root: str = "outputs/e1_evidential"
    feature_cache: str = "outputs/e1_evidential/cache/dinov3_cls.pt"
    schema_version: int = 1


@dataclass(frozen=True)
class E1Config:
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    output: OutputConfig = OutputConfig()


T = TypeVar("T")


def _construct_dataclass(cls: type[T], values: dict[str, Any], path: str) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown configuration key(s) at {path}: {names}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in values:
            continue
        value = values[item.name]
        field_type = hints[item.name]
        if is_dataclass(field_type):
            if not isinstance(value, dict):
                raise TypeError(f"{path}.{item.name} must be a mapping")
            value = _construct_dataclass(field_type, value, f"{path}.{item.name}")
        elif getattr(field_type, "__origin__", None) is tuple:
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{path}.{item.name} must be a sequence")
            value = tuple(value)
        kwargs[item.name] = value
    return cls(**kwargs)


def validate_config(config: E1Config) -> None:
    """Validate invariants that would invalidate an E1 run."""
    if not 0.0 < config.dataset.validation_fraction < 1.0:
        raise ValueError("dataset.validation_fraction must be in (0, 1)")
    if config.dataset.development_classes <= 1:
        raise ValueError("dataset.development_classes must be greater than one")
    if config.dataset.total_classes <= config.dataset.development_classes:
        raise ValueError("dataset.total_classes must exceed development_classes")
    if (
        config.dataset.expected_development_images <= 0
        or config.dataset.expected_test_images <= 0
    ):
        raise ValueError("Expected dataset image counts must be positive")
    if config.model.embedding_dim <= 0 or config.model.register_tokens < 0:
        raise ValueError("model dimensions and token counts must be valid")
    if config.model.evidence_activation != "softplus":
        raise ValueError("E1 supports only softplus evidence")
    if config.model.head not in {"linear", "mlp"}:
        raise ValueError("model.head must be 'linear' or 'mlp'")
    if config.training.batch_size <= 0 or config.training.epochs <= 0:
        raise ValueError("training batch size and epochs must be positive")
    if config.training.annealing_epochs <= 0:
        raise ValueError("training.annealing_epochs must be positive")
    if config.training.num_workers < 0:
        raise ValueError("training.num_workers cannot be negative")
    if config.training.num_workers > 0 and config.training.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive with workers")
    if not config.training.seeds:
        raise ValueError("training.seeds cannot be empty")
    if not config.retrieval.recall_k or min(config.retrieval.recall_k) <= 0:
        raise ValueError("retrieval.recall_k must contain positive values")
    if 1 not in config.retrieval.recall_k:
        raise ValueError("retrieval.recall_k must include the primary Recall@1")
    if config.retrieval.similarity_chunk_size <= 0:
        raise ValueError("similarity_chunk_size must be positive")
    if any(value <= 0 for value in config.retrieval.top_n_grid):
        raise ValueError("top_n_grid values must be positive")
    if any(not 0.0 <= value <= 1.0 for value in config.retrieval.beta_grid):
        raise ValueError("beta_grid values must be in [0, 1]")
    if not config.retrieval.self_match_exclusion:
        raise ValueError("E1 requires self-match exclusion")


def load_config(path: str | Path) -> E1Config:
    """Load a YAML config while rejecting unsupported keys."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise TypeError("Configuration root must be a mapping")
    config = _construct_dataclass(E1Config, raw, "config")
    validate_config(config)
    return config


def save_resolved_config(config: E1Config, path: str | Path) -> None:
    """Persist the exact resolved configuration."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(asdict(config), stream, sort_keys=False)
