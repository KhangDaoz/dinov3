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
