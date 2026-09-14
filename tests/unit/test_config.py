from pathlib import Path

import pytest

from uncertainty_retrieval.config import load_config


def test_loads_sequence_fields_as_tuples(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "training:\n  seeds: [42, 43]\n"
        "retrieval:\n  recall_k: [1, 2]\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.training.seeds == (42, 43)
    assert config.retrieval.recall_k == (1, 2)


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  typo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(path)


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        ("training:\n  batch_size: 0\n", "batch size"),
        ("model:\n  evidence_activation: relu\n", "softplus"),
        ("retrieval:\n  beta_grid: [1.1]\n", "beta_grid"),
    ],
)
def test_rejects_invalid_values(
    tmp_path: Path,
    yaml_text: str,
    message: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)

