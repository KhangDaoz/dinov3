from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from uncertainty_retrieval.config_e2b import load_e2b_config
from uncertainty_retrieval.data.pair_cache import verify_provenance
from uncertainty_retrieval.evaluation.pair_confidence import (
    fusion_order, raw_alpha_rankings, reliability, select_lambda,
)
from uncertainty_retrieval.models.pair_confidence import PairConfidenceNetwork
from uncertainty_retrieval.sampling.pairs import sample_epoch_pairs, validate_pair_endpoints
from uncertainty_retrieval.training.pair_confidence import merge_shards, score_pairs
from uncertainty_retrieval.training.representation import atomic_torch_save


def test_e2b_config_has_fixed_recipe():
    config = load_e2b_config(Path(__file__).resolve().parents[2] / "configs/cub_e2b.yaml")
    assert config.training.seed == 42
    assert config.runtime.world_size == 2
    assert config.evaluation.ranking_depth == 100


def test_static_sampling_deterministic_balanced_and_split_owned():
    labels = torch.tensor([0,0,1,1,2,2])
    candidates = torch.tensor([[1,2],[0,2],[3,0],[2,0],[5,0],[4,0]])
    original = candidates.clone()
    first = sample_epoch_pairs(labels,candidates,1)
    second = sample_epoch_pairs(labels,candidates,1)
    assert torch.equal(first[0],second[0]) and torch.equal(candidates,original)
    assert first[0].shape==(192,2) and first[2]["positive_fraction"]==0.5
    validate_pair_endpoints(first[0],labels,first[1])


def test_hard_negative_empty_pool_falls_back():
    labels=torch.tensor([0,0,1,1])
    candidates=torch.tensor([[1],[0],[3],[2]])
    _,_,metadata=sample_epoch_pairs(labels,candidates,2)
    assert metadata["hard_negative_fallback_queries"]==4


def test_endpoint_validation_rejects_leakage_and_self():
    with pytest.raises(ValueError,match="outside"):
        validate_pair_endpoints(torch.tensor([[0,4]]),torch.tensor([0,0]),torch.ones(1))
    with pytest.raises(ValueError,match="Self"):
        validate_pair_endpoints(torch.tensor([[0,0]]),torch.tensor([0,0]),torch.ones(1))


def test_pair_network_gradient_and_fixed_orientation():
    torch.manual_seed(42)
    model=PairConfidenceNetwork(4)
    q=torch.randn(2,4,requires_grad=True)
    x=torch.randn(2,4,requires_grad=True)
    logits=model(q,x)
    F.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.])).backward()
    assert q.grad is None and x.grad is None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    features=torch.stack([q[0].detach(),x[0].detach()])
    scored=score_pairs(model,features,torch.tensor([0]),torch.tensor([[1]]),1)
    assert torch.allclose(scored.flatten(),model(features[:1],features[1:]))
    assert not torch.allclose(scored.flatten(),model(features[1:],features[:1]))


def test_fusion_endpoints_ties_and_lambda_tie_rule():
    cosine=torch.tensor([[.9,.8,.7]])
    confidence=torch.tensor([[.2,.8,.8]])
    order,_=fusion_order(cosine,confidence,1)
    assert order.tolist()==[[0,1,2]]
    order,_=fusion_order(cosine,confidence,0)
    assert order.tolist()==[[1,2,0]]
    grid={str(value):{f"hits_at_{k}":1 for k in (1,2,4,8)} for value in (0,0.5,1)}
    assert select_lambda(grid)==1


def test_raw_alpha_l2_not_normalized():
    alpha=torch.tensor([[1.,1.],[2.,2.],[1.,3.]])
    order,distances=raw_alpha_rankings(alpha,torch.tensor([10,11,12]),torch.tensor([0]),1,2)
    assert order.tolist()==[[1,2]]
    assert distances[0,0]==pytest.approx(2**0.5)
    assert distances[0,1]==2


def test_undefined_reliability_not_fabricated():
    result=reliability(torch.tensor([.1,.2]),torch.zeros(2))
    assert result["auroc"]["value"] is None
    assert len(result["reliability_bins"])==10


def test_provenance_mismatch_fails():
    with pytest.raises(ValueError,match="cache"):
        verify_provenance({"cache":"old"},{"cache":"new"})


def test_shard_merge_alignment_and_duplicates(tmp_path):
    atomic_torch_save({"rows":torch.tensor([0,2]),"logits":torch.tensor([[10.],[30.]])},tmp_path/"test_rank0.pt")
    atomic_torch_save({"rows":torch.tensor([1]),"logits":torch.tensor([[20.]])},tmp_path/"test_rank1.pt")
    assert merge_shards(tmp_path,"test",2,["logits"])["logits"].flatten().tolist()==[10,20,30]
    atomic_torch_save({"rows":torch.tensor([0]),"logits":torch.tensor([[20.]])},tmp_path/"test_rank1.pt")
    with pytest.raises(ValueError,match="duplicate"):
        merge_shards(tmp_path,"test",2,["logits"])
