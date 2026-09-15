import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_runner():
    path = Path(__file__).resolve().parents[2] / "scripts/run_e2a_test_selection.py"
    spec = importlib.util.spec_from_file_location("test_selection_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", ["m1", "m2", "m3", "m4"])
def test_export_includes_required_method_artifacts(tmp_path, method):
    root = tmp_path / "source"
    files = ["metrics/test.json", "rankings/test_top100.pt", "embeddings/manifest.json"]
    if method in {"m3", "m4"}:
        files.append("embeddings/test.pt")
    if method == "m4":
        files.extend(["attention/test_weights.pt", "attention/test_entropy.pt"])
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture", encoding="utf-8")
    config = SimpleNamespace(representation=SimpleNamespace(method=method),
                             output=SimpleNamespace(root=root))
    destination = tmp_path / "bundle"
    load_runner().export_test_artifacts(config, config_path, destination)
    for name in files + ["config.yaml"]:
        assert (destination / "artifacts" / method / name).is_file()


def test_export_rejects_missing_test_outputs(tmp_path):
    config = SimpleNamespace(representation=SimpleNamespace(method="m1"),
                             output=SimpleNamespace(root=tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="Missing test artifacts"):
        load_runner().export_test_artifacts(config, tmp_path / "config.yaml",
                                           tmp_path / "bundle")
