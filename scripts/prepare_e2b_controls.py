#!/usr/bin/env python
"""Controlled E1-style heads on accepted M1 features, before benchmark access."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from uncertainty_retrieval.config_e2b import load_e2b_config
from uncertainty_retrieval.data.pair_cache import load_pair_inputs, verify_provenance
from uncertainty_retrieval.models.evidential import EvidentialHead
from uncertainty_retrieval.training.evidential import (
    FeatureDataset, export_evidential_outputs, make_feature_loader,
    save_evidential_outputs, train_evidential_head,
)
from uncertainty_retrieval.training.pair_confidence import setup
from uncertainty_retrieval.training.representation import atomic_torch_save
from uncertainty_retrieval.utils import cleanup_distributed, distributed_barrier, write_json


def prepare_controls(config_path):
    config=load_e2b_config(config_path)
    rank,world,device=setup(config)
    root=Path(config.output_root)/"controls/edl"
    try:
        fit,provenance=load_pair_inputs(config,config_path,"fit")
        val,_=load_pair_inputs(config,config_path,"validation")
        def dataset(view,indices=None):
            indices=torch.arange(len(view["image_ids"])) if indices is None else indices
            return FeatureDataset(view["raw_features"][indices],view["labels"][indices],view["image_ids"][indices])
        def loader(ds,distributed=False,shuffle=False):
            return make_feature_loader(ds,256,0,2,shuffle,distributed,42)
        tune_path=root/"tuning/best.pt"
        model=EvidentialHead(768,100)
        fit_loader,sampler=loader(dataset(fit),world>1,True)
        val_loader,_=loader(dataset(val,torch.arange(rank,len(val["image_ids"]),world)))
        result=train_evidential_head(model,fit_loader,val_loader,device,30,10,0.001,0.0001,
                                     False,tune_path,rank,sampler)
        distributed_barrier(device)
        state=torch.load(tune_path,map_location="cpu",weights_only=True)
        model.load_state_dict(state["model"])
        if rank==0:
            val_export,_=loader(dataset(val))
            payload=export_evidential_outputs(model,val_export,device,list(range(100)),2)
            payload["provenance"]=provenance
            save_evidential_outputs(payload,root/"tuning/uncertainty/validation.pt")
            state["provenance"]=provenance
            atomic_torch_save(state,tune_path)
            write_json({"selected_epochs":result.best_epoch,"history":result.history,
                        "fit_sampler_padding":(-len(fit["image_ids"]))%world,
                        "procedure":"controlled_E1_retraining_on_raw_M1_CLS"},root/"training.json")
        distributed_barrier(device)
        if rank==0:
            print(f"Controlled fit-only E1 head saved: {root.resolve()}",flush=True)
        distributed_barrier(device)
    finally:
        cleanup_distributed()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True,type=Path)
    prepare_controls(parser.parse_args().config)
