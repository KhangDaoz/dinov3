from dataclasses import replace

import pytest
import torch

from uncertainty_retrieval.config_e2b import E2BConfig
from uncertainty_retrieval.data.e2b_controls import load_e1_control
from uncertainty_retrieval.models.evidential import EvidentialHead
from uncertainty_retrieval.training.evidential import save_evidential_outputs
from uncertainty_retrieval.training.representation import atomic_torch_save


def test_control_validation_uses_tuning_and_retains_full_vectors(tmp_path):
    config = replace(E2BConfig(), output_root=str(tmp_path))
    alpha = torch.full((2,100),2.)
    payload = {"alpha":alpha,"evidence":alpha-1,"uncertainty":100/alpha.sum(1),
               "image_ids":torch.tensor([11,12]),"labels":torch.tensor([0,1]),
               "class_order":list(range(100)),"provenance":{"cache_sha256":"fixture"},
               "probabilities":alpha/alpha.sum(1,keepdim=True),"predictions":torch.zeros(2,dtype=torch.long),
               "schema_version":2,"dtype":"torch.float32"}
    path=tmp_path/"controls/edl/tuning/uncertainty/validation.pt"
    save_evidential_outputs(payload,path)
    view={"image_ids":payload["image_ids"],"labels":payload["labels"]}
    values,uncertainty,metadata=load_e1_control(config,"validation",view,payload["provenance"])
    assert values.shape==(2,100) and torch.equal(values,alpha)
    assert metadata["training_budget"]=="fit_only"
    payload["provenance"]["cache_sha256"]="other"
    with pytest.raises(ValueError,match="provenance"):
        load_e1_control(config,"validation",view,payload["provenance"])


def test_benchmark_control_only_forwards_frozen_fit_only_head(tmp_path):
    config=replace(E2BConfig(),output_root=str(tmp_path))
    provenance={"cache_sha256":"fixture"}
    model=EvidentialHead(768,100)
    atomic_torch_save({"model":model.state_dict(),"provenance":provenance},
                      tmp_path/"controls/edl/tuning/best.pt")
    view={"raw_features":torch.randn(2,768),"image_ids":torch.tensor([51,52]),
          "labels":torch.tensor([100,101])}
    alpha,u,metadata=load_e1_control(config,"test",view,provenance,torch.device("cpu"))
    assert alpha.shape==(2,100) and (u>0).all()
    assert metadata["training_budget"]=="fit_only"
    assert (tmp_path/"controls/edl/tuning/uncertainty/test.pt").is_file()
    # A checkpoint cannot substitute for a missing split-owned validation export.
    with pytest.raises(FileNotFoundError):
        load_e1_control(config,"validation",view,provenance)
