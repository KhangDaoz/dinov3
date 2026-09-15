"""Tiny synthetic validation->benchmark regression; no CUB/model download."""

from dataclasses import replace

import torch

from uncertainty_retrieval.config_e2b import E2BConfig
from uncertainty_retrieval.data.feature_cache import sha256_file
from uncertainty_retrieval.evaluation import e2b_pipeline as pipeline
from uncertainty_retrieval.evaluation.pair_confidence import ranking_metrics
from uncertainty_retrieval.models.pair_confidence import PairConfidenceNetwork
from uncertainty_retrieval.training.representation import atomic_torch_save
from uncertainty_retrieval.utils import write_json


def test_synthetic_validation_then_frozen_test_exports_complete_bundle(tmp_path, monkeypatch):
    torch.manual_seed(42)
    config = E2BConfig()
    config = replace(config, output_root=str(tmp_path),
                     runtime=replace(config.runtime, world_size=1, device="cpu"),
                     controls=replace(config.controls, enabled=False),
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
    pipeline.evaluate(tmp_path / "config.yaml","validation")
    before = sha256_file(tmp_path / "checkpoints/best.pt")
    pipeline.evaluate(tmp_path / "config.yaml","test")
    assert sha256_file(tmp_path / "checkpoints/best.pt")==before
    for name in ("scores/test_top100.pt","metrics/test.json","metrics/test_bootstrap.json",
                 "failure_cases/test.json","rankings/test_fusion.pt","selection.json","report.tex",
                 "figures/test_candidate_calibration.svg"):
        assert (tmp_path / "export" / name).is_file()
