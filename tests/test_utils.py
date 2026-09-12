from pathlib import Path

import pytest
import torch

from src.utils import (
    atomic_json_save,
    atomic_torch_save,
    load_json,
    public_config,
    resolve_device,
    sha256_file,
    load_config,
)


def test_public_config_redacts_token():
    config = {"model_name": "model", "token": "secret", "seed": 42}
    assert public_config(config) == {"model_name": "model", "seed": 42}


def test_auto_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_auto_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_atomic_artifacts_round_trip(tmp_path):
    tensor_path = tmp_path / "nested" / "value.pt"
    json_path = tmp_path / "nested" / "value.json"
    value = torch.tensor([1.0, 2.0])
    atomic_torch_save(value, tensor_path)
    atomic_json_save({"status": "ok"}, json_path)
    assert torch.equal(
        torch.load(tensor_path, map_location="cpu", weights_only=True), value
    )
    assert load_json(json_path) == {"status": "ok"}
    assert len(sha256_file(tensor_path)) == 64
    assert not list(Path(tensor_path.parent).glob(".*"))


def test_config_inheritance_overrides_representation_and_output(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "\n".join(
            [
                "model_name: model",
                "model_revision: revision",
                "representation: cls",
                "data_root: data",
                "split: [train, eval]",
                "batch_size: 2",
                "num_workers: 0",
                "device: auto",
                "recall_k: [1, 2]",
                "retrieval_chunk_size: 4",
                "output_dir: outputs/cls",
                "seed: 42",
                "num_threads: 1",
                "token: secret",
            ]
        ),
        encoding="utf-8",
    )
    child.write_text(
        "extends: base.yaml\n"
        "representation: mean_patch\n"
        "output_dir: outputs/mean_patch\n",
        encoding="utf-8",
    )
    config = load_config(child)
    assert config["representation"] == "mean_patch"
    assert config["output_dir"] == "outputs/mean_patch"
    assert config["token"] == "secret"


def test_config_inheritance_rejects_cycle(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="vòng lặp"):
        load_config(first)


def test_repository_e1_config_is_valid():
    config = load_config(Path("configs/cub_e1.yaml"))
    assert config["experiment"] == "e1"
    assert config["representation"] == "cls"
    assert config["reranking"]["top_n_grid"] == [8, 16, 32, 64, 128]
    assert config["token"] is None
