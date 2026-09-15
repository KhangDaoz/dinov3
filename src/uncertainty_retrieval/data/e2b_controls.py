"""Conservative reuse gates for E1 evidential comparison controls."""

from pathlib import Path

import torch

from .feature_cache import sha256_file
from .pair_cache import verify_provenance


def load_e1_control(config, split, view, provenance, device=None):
    # Filename scope is deliberate: final heads have seen validation images.
    scope = "tuning" if split == "validation" else "final"
    if config.controls.source == "controlled_retrain":
        root = Path(config.output_root) / "controls/edl"
        path = root / "tuning/uncertainty/validation.pt"
        if split == "test":
            from uncertainty_retrieval.models.evidential import EvidentialHead
            from uncertainty_retrieval.training.evidential import (
                FeatureDataset, export_evidential_outputs, make_feature_loader,
            )
            checkpoint_path = root / "final/best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            verify_provenance(checkpoint["provenance"], {k:v for k,v in provenance.items()
                                                       if not k.endswith("mining_sha256")})
            model = EvidentialHead(768,100)
            model.load_state_dict(checkpoint["model"])
            loader,_ = make_feature_loader(FeatureDataset(view["raw_features"],view["labels"],view["image_ids"]),
                                          256,0,2,False,False,42)
            payload = export_evidential_outputs(model,loader,device,list(range(100)),2)
            payload["provenance"] = checkpoint["provenance"]
            from uncertainty_retrieval.training.evidential import save_evidential_outputs
            from uncertainty_retrieval.utils import distributed_context
            if distributed_context()[0] == 0:
                save_evidential_outputs(payload, root / "final/uncertainty/test.pt")
            source_hash = sha256_file(checkpoint_path)
            path = checkpoint_path
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            source_hash = sha256_file(path)
        verify_provenance(payload["provenance"], {k:v for k,v in provenance.items()
                                                 if not k.endswith("mining_sha256")})
        cache_hash = provenance["cache_sha256"]
    else:
        payload = None
        source_hash = None
        path = Path(config.controls.e1_root) / scope / "uncertainty" / f"{split}.pt"
        cache_hash = sha256_file(config.controls.e1_cache)
    if cache_hash != provenance["cache_sha256"]:
        raise ValueError(
            "E1/M1 feature cache hashes differ. Stop control reuse; perform a "
            "documented controlled E1 re-export/retraining, or explicitly disable "
            "controls and report them unavailable (never copy historical scores)."
        )
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        source_hash = sha256_file(path)
    if payload.get("class_order") != list(range(100)):
        raise ValueError("E1 control class ordering mismatch")
    ids = payload["image_ids"]
    if len(ids.unique()) != len(ids):
        raise ValueError("Duplicate E1 output IDs")
    positions = {int(value): i for i, value in enumerate(ids)}
    try:
        indices = torch.tensor([positions[int(value)] for value in view["image_ids"]])
    except KeyError as error:
        raise ValueError("Missing E1 control image IDs") from error
    alpha = payload["alpha"][indices].float()
    evidence = payload["evidence"][indices].float()
    uncertainty = payload["uncertainty"][indices].float()
    if alpha.shape != (len(indices), 100) or evidence.shape != alpha.shape:
        raise ValueError("E1 must retain full [N,100] evidence and alpha")
    if not torch.equal(payload["labels"][indices], view["labels"]):
        raise ValueError("E1 control labels mismatch")
    if not all(torch.isfinite(value).all() for value in (alpha, evidence, uncertainty)):
        raise ValueError("Nonfinite E1 control")
    if (evidence < 0).any() or (alpha < 1).any():
        raise ValueError("Invalid evidential values")
    if not torch.allclose(alpha, evidence + 1) or not torch.allclose(
        uncertainty, 100 / alpha.sum(1), rtol=1e-5, atol=1e-7
    ):
        raise ValueError("E1 alpha/evidence/uncertainty equations mismatch")
    return alpha, uncertainty, {
        "scope": scope, "source": str(path), "source_sha256": source_hash,
        "procedure": config.controls.source,
        "feature_cache_sha256": cache_hash,
        "training_budget": "fit_only" if scope == "tuning" else "all_development",
        "notice": "DINOv3 adaptation; A1 follows raw-alpha/L2 representation only.",
    }
