from dataclasses import replace
from pathlib import Path

import pytest

from uncertainty_retrieval.config_e2a import load_e2a_config, validate_e2a_config


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "cub_e2a_m1.yaml"


def test_m1_config_has_fixed_protocol() -> None:
    config = load_e2a_config(CONFIG)
    assert config.representation.method == "m1"
    assert config.evaluation.ranking_depth == 100
    assert config.runtime.world_size == 2


def test_m2_config_has_fixed_mean_patch_contract() -> None:
    config = load_e2a_config(CONFIG.with_name("cub_e2a_m2.yaml"))
    assert config.representation.method == "m2"
    assert config.representation.pooling == "mean"
    assert config.model.expected_patch_tokens == 196
    assert config.cache.reuse_candidates == ()


def test_m2_rejects_cls_cache_reuse() -> None:
    config = load_e2a_config(CONFIG.with_name("cub_e2a_m2.yaml"))
    invalid = replace(
        config,
        cache=replace(
            config.cache,
            reuse_candidates=("outputs/e2a_cls/cache/dinov3_cls.pt",),
        ),
    )
    with pytest.raises(ValueError, match="cannot reuse CLS"):
        validate_e2a_config(invalid)


def test_m3_config_locks_training_recipe() -> None:
    config = load_e2a_config(CONFIG.with_name("cub_e2a_m3.yaml"))
    assert config.representation.method == "m3"
    assert config.training is not None
    assert config.training.seed == 42
    assert config.training.classes_per_batch * config.training.images_per_class == 80


def test_m3_rejects_training_recipe_drift() -> None:
    config = load_e2a_config(CONFIG.with_name("cub_e2a_m3.yaml"))
    invalid = replace(
        config,
        training=replace(config.training, proxy_margin=0.2),
    )
    with pytest.raises(ValueError, match="locked plan"):
        validate_e2a_config(invalid)


def test_m1_rejects_patch_pooling() -> None:
    config = load_e2a_config(CONFIG)
    invalid = replace(
        config,
        representation=replace(config.representation, pooling="mean_patch"),
    )
    with pytest.raises(ValueError, match="representation contract"):
        validate_e2a_config(invalid)


def test_m1_rejects_mutable_checkpoint_revision() -> None:
    config = load_e2a_config(CONFIG)
    invalid = replace(config, model=replace(config.model, revision="main"))
    with pytest.raises(ValueError, match="immutable"):
        validate_e2a_config(invalid)
