from dataclasses import replace

import pytest
import torch

from uncertainty_retrieval.config_e2a import E2AConfig
from uncertainty_retrieval.data.patch_token_cache import (
    fp16_fidelity_statistics,
    validate_fidelity,
)


def test_fp16_roundtrip_fidelity_statistics_are_near_identity() -> None:
    torch.manual_seed(42)
    reference = torch.randn(4, 6, 8, dtype=torch.float32)
    statistics = fp16_fidelity_statistics(reference)
    assert statistics["max_relative_l2"] < 1e-3
    assert statistics["min_patch_cosine"] > 0.99999
    assert statistics["min_mean_cosine"] > 0.99999


def test_fidelity_gate_rejects_threshold_failure() -> None:
    config = E2AConfig()
    config = replace(
        config,
        cache=replace(
            config.cache,
            fidelity_relative_l2_max=1e-3,
            fidelity_patch_cosine_min=0.99999,
            fidelity_mean_cosine_min=0.99999,
        ),
    )
    with pytest.raises(ValueError, match="fidelity failed"):
        validate_fidelity(
            {"max_relative_l2": 0.1, "min_patch_cosine": 0.9,
             "min_mean_cosine": 0.9},
            config,
        )
