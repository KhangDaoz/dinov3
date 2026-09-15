"""One tiny synthetic epoch, not a real experiment."""

from dataclasses import replace

import torch

from uncertainty_retrieval.config_e2b import E2BConfig
from uncertainty_retrieval.models.pair_confidence import PairConfidenceNetwork
from uncertainty_retrieval.training import pair_confidence as training
from uncertainty_retrieval.training.representation import atomic_torch_save
from uncertainty_retrieval.utils import write_json


def test_one_epoch_writes_fit_only_pairs_and_checkpoint(tmp_path,monkeypatch):
    config=E2BConfig()
    config=replace(config,output_root=str(tmp_path),training=replace(config.training,epochs=1,global_batch_size=32))
    labels=torch.tensor([0,0,1,1,2,2])
    candidates=torch.tensor([[1,2],[0,2],[3,0],[2,0],[5,0],[4,0]])
    fit={"features":torch.randn(6,2),"labels":labels,"image_ids":torch.arange(1,7)}
    validation={"features":torch.randn(6,2),"labels":labels,"image_ids":torch.arange(11,17)}
    provenance={"cache_sha256":"fixture"}
    for split,view in (("fit",fit),("validation",validation)):
        atomic_torch_save({"image_ids":view["image_ids"],"candidates":candidates,
                           "cosine":torch.ones_like(candidates,dtype=torch.float32),
                           "targets":labels[candidates].eq(labels[:,None]),"provenance":provenance},
                          tmp_path/f"pairs/{split}_static.pt")
    write_json({},tmp_path/"environment.json")
    monkeypatch.setattr(training,"load_e2b_config",lambda path:config)
    monkeypatch.setattr(training,"load_pair_inputs",lambda config,path,split:(
        fit if split=="fit" else validation,dict(provenance)))
    monkeypatch.setattr(training,"setup",lambda config:(0,1,torch.device("cpu")))
    monkeypatch.setattr(training,"PairConfidenceNetwork",lambda:PairConfidenceNetwork(2))
    training.train(tmp_path/"config.yaml")
    checkpoint=torch.load(tmp_path/"checkpoints/best.pt",weights_only=True)
    pairs=torch.load(tmp_path/"pairs/epoch_01.pt",weights_only=True)
    assert checkpoint["epoch"]==1 and checkpoint["validation_bce"]>0
    assert set(pairs["image_pairs"].flatten().tolist()).issubset(set(range(1,7)))
