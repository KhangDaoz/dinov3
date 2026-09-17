"""Tiny synthetic validation->benchmark regression; no CUB/model download."""

from dataclasses import replace

import torch
import pytest

from uncertainty_retrieval.config_e2b import E2BConfig
from uncertainty_retrieval.data.feature_cache import sha256_file
from uncertainty_retrieval.evaluation import e2b_pipeline as pipeline
from uncertainty_retrieval.evaluation.pair_confidence import ranking_metrics
from uncertainty_retrieval.models.pair_confidence import PairConfidenceNetwork
from uncertainty_retrieval.training.representation import atomic_torch_save
from uncertainty_retrieval.utils import write_json


@pytest.mark.parametrize("controls_enabled", [False, True])
def test_synthetic_validation_then_frozen_test_exports_complete_bundle(tmp_path, monkeypatch, controls_enabled):
    torch.manual_seed(42)
    config = E2BConfig()
    config = replace(config, output_root=str(tmp_path),
                     runtime=replace(config.runtime, world_size=1, device="cpu"),
                     controls=replace(config.controls, enabled=controls_enabled),
                     inputs=replace(config.inputs, selection=str(tmp_path / "e2a.json")))
    n = 101
    ids = torch.arange(1000, 1000+n)
    labels = torch.arange(n)//2
    candidates = torch.stack([torch.tensor([j for j in range(n) if j != i]) for i in range(n)])
    cosine = torch.linspace(.9,.1,100).repeat(n,1)
    view = {"features":torch.randn(n,2), "image_ids":ids, "labels":labels,
            "paths":[f"image_{i}.jpg" for i in range(n)],
            "accepted_ranking":{"candidate_image_ids":ids[candidates],"cosine_scores":cosine}}
    provenance = {"cache_sha256":"fixture", "split_hash":"fixture"}
    atomic_torch_save({"image_ids":ids,"candidates":candidates,"cosine":cosine},
                      tmp_path / "pairs/validation_static.pt")
    atomic_torch_save({},tmp_path / "pairs/fit_static.pt")
    checkpoint_provenance = dict(provenance)
    for split in ("fit","validation"):
        checkpoint_provenance[f"{split}_mining_sha256"] = sha256_file(tmp_path / f"pairs/{split}_static.pt")
    atomic_torch_save({"model":PairConfidenceNetwork(2).state_dict(), "epoch":1,
                       "provenance":checkpoint_provenance},tmp_path / "checkpoints/best.pt")
    write_json({"test_hits":{"m1":{k:v for k,v in ranking_metrics(candidates,labels).items()
                                    if k.startswith("hits")}}},tmp_path / "e2a.json")
    (tmp_path / "config_resolved.yaml").write_text("fixture",encoding="utf-8")
    monkeypatch.setattr(pipeline,"load_e2b_config",lambda path: config)
    monkeypatch.setattr(pipeline,"load_pair_inputs",lambda *args: (view,dict(provenance)))
    monkeypatch.setattr(pipeline,"setup",lambda config: (0,1,torch.device("cpu")))
    monkeypatch.setattr(pipeline,"PairConfidenceNetwork",lambda: PairConfidenceNetwork(2))
    cleanup_calls = []
    monkeypatch.setattr(pipeline, "cleanup_distributed", lambda: cleanup_calls.append(True))
    original_export = pipeline.export_e2b

    def checked_export(root, split):
        assert cleanup_calls, "Process group must close before rank-0 export"
        original_export(root, split)

    monkeypatch.setattr(pipeline, "export_e2b", checked_export)
    if controls_enabled:
        alpha = 1 + torch.rand(n, 100)
        uncertainty = 100 / alpha.sum(1)
        monkeypatch.setattr(pipeline, "load_e1_control", lambda *args: (
            alpha, uncertainty, {"source": "synthetic_fixture"}
        ))
        for name in ("controls/edl/tuning/uncertainty/validation.pt",
                     "controls/edl/tuning/best.pt"):
            atomic_torch_save({}, tmp_path / name)
    pipeline.evaluate(tmp_path / "config.yaml","validation")
    before = sha256_file(tmp_path / "checkpoints/best.pt")
    pipeline.evaluate(tmp_path / "config.yaml","test")
    assert sha256_file(tmp_path / "checkpoints/best.pt")==before
    for name in ("scores/test_top100.pt","metrics/test.json","metrics/test_bootstrap.json",
                 "failure_cases/test.json","rankings/test_fusion.pt","selection.json","report.tex",
                 "figures/test_candidate_calibration.svg",
                 "metrics/test_topn_lambda_grid.json",
                 "metrics/test_method_topn.json",
                 "figures/test_topn_lambda_recall1.svg",
                 "figures/failure_cases/test_fusion_n100_still_wrong.png"):
        assert (tmp_path / "export" / name).is_file()
    grid = __import__("json").loads(
        (tmp_path / "metrics/test_topn_lambda_grid.json").read_text()
    )
    assert set(grid) == {"10", "20", "50", "100"}
    assert all(len(row) == len(config.evaluation.lambdas) for row in grid.values())
    if controls_enabled:
        for name in ("controls/test_U1.pt", "controls/test_A1.pt", "controls/test_A2.pt",
                     "controls/test_U1_n10.pt", "controls/test_U1_n100.pt",
                     "controls/test_u1_grid.json"):
            assert (tmp_path / "export" / name).is_file()
