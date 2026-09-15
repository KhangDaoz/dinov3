"""Regression checks for M4 test-stage checkpoint provenance."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from uncertainty_retrieval.config_e2a import load_e2a_config
from uncertainty_retrieval.data.feature_cache import validation_ids_hash


def test_test_stage_checks_validation_hash_before_inference(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "evaluate_e2a_m4", root / "scripts" / "evaluate_e2a_m4.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = root / "configs" / "cub_e2a_m4.yaml"
    config = load_e2a_config(config_path)
    validation = [SimpleNamespace(image_id=11, split="development")]
    test = [SimpleNamespace(image_id=99, split="test")]
    monkeypatch.setattr("sys.argv", ["evaluate_e2a_m4.py", "--config",
                                    str(config_path), "--stage", "test"])
    monkeypatch.setattr(module, "load_cub_records", lambda *args: validation + test)
    monkeypatch.setattr(module, "split_development_records",
                        lambda *args: ([], validation))
    monkeypatch.setattr(module, "load_patch_token_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "sha256_file", lambda path: "accepted")
    checkpoint = {"provenance": {
        "config_sha256": "accepted",
        "patch_manifest_sha256": "accepted",
        "split_hash": validation_ids_hash([11]),
    }}
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: checkpoint)

    class InferenceReached(Exception):
        pass

    def stop_before_inference():
        raise InferenceReached

    monkeypatch.setattr(module, "resolve_device", stop_before_inference)
    with pytest.raises(InferenceReached):
        module.main()
