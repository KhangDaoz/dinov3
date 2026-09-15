"""Strict, preregistered E2B settings."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .config_e2a import E2ADatasetConfig, _construct


@dataclass(frozen=True)
class PairInputs:
    cache: str = "outputs/e2a_cls/cache/dinov3_cls.pt"
    manifest: str = "outputs/e2a_selection/artifacts/m1/embeddings/manifest.json"
    selection: str = "outputs/e2a_selection/test_selection.json"
    validation_ids: str = "outputs/e2a_cls/m1/split/validation_image_ids.pt"


@dataclass(frozen=True)
class PairTraining:
    seed: int = 42
    epochs: int = 30
    global_batch_size: int = 4096
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0


@dataclass(frozen=True)
class PairRuntime:
    world_size: int = 2
    device: str = "cuda"
    pair_chunk_size: int = 4096
    query_chunk_size: int = 128


@dataclass(frozen=True)
class PairControls:
    enabled: bool = True
    source: str = "controlled_retrain"
    e1_root: str = "outputs/e1_evidential/seed_42"
    e1_cache: str = "outputs/e1_evidential/cache/dinov3_cls.pt"


@dataclass(frozen=True)
class PairEvaluation:
    ranking_depth: int = 100
    lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    uncertainty_top_n: tuple[int, ...] = (10, 20, 50, 100)
    bootstrap_samples: int = 2000


@dataclass(frozen=True)
class E2BConfig:
    dataset: E2ADatasetConfig = E2ADatasetConfig()
    inputs: PairInputs = PairInputs()
    training: PairTraining = PairTraining()
    runtime: PairRuntime = PairRuntime()
    controls: PairControls = PairControls()
    evaluation: PairEvaluation = PairEvaluation()
    output_root: str = "outputs/e2b_pair_confidence/seed_42"


def load_e2b_config(path: str | Path) -> E2BConfig:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("E2B config must be a mapping")
    config = _construct(E2BConfig, values, "e2b")
    if config.dataset != E2ADatasetConfig():
        # Root is configurable; all split/count settings are fixed.
        from dataclasses import replace
        if replace(config.dataset, root=E2ADatasetConfig().root) != E2ADatasetConfig():
            raise ValueError("E2B class split/counts and validation recipe are fixed")
    if config.training != PairTraining():
        raise ValueError("E2B training recipe is preregistered")
    if config.evaluation != PairEvaluation():
        raise ValueError("E2B evaluation recipe is preregistered")
    runtime = config.runtime
    if config.controls.source not in {"controlled_retrain", "reuse_e1"}:
        raise ValueError("Unknown E1 control source")
    if runtime.device not in {"cuda", "cpu"} or runtime.world_size <= 0:
        raise ValueError("Invalid E2B runtime")
    if min(runtime.pair_chunk_size, runtime.query_chunk_size) <= 0:
        raise ValueError("Chunk sizes must be positive")
    if config.training.global_batch_size % runtime.world_size:
        raise ValueError("Global pair batch must divide evenly across ranks")
    return config


def save_e2b_config(config: E2BConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
