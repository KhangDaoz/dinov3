from pathlib import Path

import torch

from src.utils import (
    atomic_json_save,
    atomic_torch_save,
    load_json,
    public_config,
    resolve_device,
    sha256_file,
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
