"""Accepted M1 inputs and split-owned pair feature views."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from uncertainty_retrieval.config_e2a import E2AConfig
from uncertainty_retrieval.data.cub import (
    load_cub_records, split_development_records, validate_protocol_counts,
)
from uncertainty_retrieval.data.feature_cache import (
    cub_manifest_hash, sha256_file, validate_representation_cache, validation_ids_hash,
)
from uncertainty_retrieval.data.patch_cache import load_feature_cache
from uncertainty_retrieval.evaluation.representation import (
    hits_from_ranking, select_representation_winner,
)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pair_inputs(config, config_path: Path, split: str) -> tuple[dict, dict]:
    if split not in {"fit", "validation", "test"}:
        raise ValueError("Unknown E2B split")
    selection = read_json(config.inputs.selection)
    winner, _ = select_representation_winner(
        selection["test_hits"], selection["optimized_parameters"]
    )
    if winner != "m1" or selection["winner"] != "m1":
        raise ValueError("E2B requires the accepted M1 winner")
    accepted = Path(config.inputs.selection).parent / "artifacts" / "m1"
    for key, name in (("config", "config.yaml"), ("metrics", "metrics/test.json"),
                      ("ranking", "rankings/test_top100.pt")):
        if sha256_file(accepted / name) != selection["artifact_hashes"]["m1"][key]:
            raise ValueError(f"Accepted M1 {key} hash mismatch")
    manifest = read_json(config.inputs.manifest)
    cache_hash = sha256_file(config.inputs.cache)
    if cache_hash != manifest["cache_sha256"]:
        raise ValueError("M1 feature cache hash differs from accepted manifest")
    records = load_cub_records(config.dataset.root)
    validate_protocol_counts(records, config.dataset.expected_development_images,
                             config.dataset.expected_test_images)
    cache = load_feature_cache(config.inputs.cache)
    reference = E2AConfig(dataset=config.dataset)
    validate_representation_cache(cache, records, reference,
                                  cub_manifest_hash(config.dataset.root))
    if cache["metadata"] != manifest["metadata"]:
        raise ValueError("M1 manifest metadata differs from feature cache")
    fit, validation = split_development_records(records, 0.2, 42)
    reference_ids = torch.load(config.inputs.validation_ids, map_location="cpu",
                               weights_only=True)
    val_ids = torch.tensor([r.image_id for r in validation])
    if not torch.equal(val_ids, reference_ids):
        raise ValueError("E2B validation IDs differ from accepted M1")
    if validation_ids_hash(val_ids.tolist()) != (
        "d9455ac8948ff640f94017339681a51744f6351f1db053732b3153a4c9c958ee"
    ):
        raise ValueError("Canonical validation split hash mismatch")
    selected = {"fit": fit, "validation": validation,
                "test": [r for r in records if r.split == "test"]}[split]
    ids = torch.tensor([r.image_id for r in selected])
    by_id = {int(v): i for i, v in enumerate(cache["image_ids"])}
    positions = torch.tensor([by_id[int(v)] for v in ids])
    features = cache["features"][positions]
    if (features.norm(dim=1) == 0).any():
        raise ValueError("Zero-norm M1 features")
    provenance = {
        "config_sha256": sha256_file(config_path), "cache_sha256": cache_hash,
        "manifest_sha256": sha256_file(config.inputs.manifest),
        "e2a_selection_sha256": sha256_file(config.inputs.selection),
        "split_hash": validation_ids_hash(val_ids.tolist()),
        "fit_hash": validation_ids_hash([r.image_id for r in fit]),
        "orientation": "query_first_no_bidirectional_average",
        "mining": "static_frozen_m1_cosine", "seed": 42,
    }
    view = {
        "features": F.normalize(features.float(), dim=1), "raw_features": features,
        "image_ids": ids,
        "labels": torch.tensor([r.original_label for r in selected]),
        "paths": [r.relative_path for r in selected], "split": split,
    }
    if split == "test":
        ranking = torch.load(accepted / "rankings/test_top100.pt",
                             map_location="cpu", weights_only=True)
        if not torch.equal(ids, ranking["query_image_ids"]):
            raise ValueError("Benchmark IDs differ from accepted M1 ranking")
        if hits_from_ranking(ranking) != selection["test_hits"]["m1"]:
            raise ValueError("Accepted M1 hit counts are inconsistent")
        view["accepted_ranking"] = ranking
    return view, provenance


def verify_provenance(actual: dict, expected: dict) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"E2B provenance mismatch: {key}")


def validate_static_pool(payload: dict, view: dict) -> None:
    if not torch.equal(payload["image_ids"], view["image_ids"]):
        raise ValueError("Static mining IDs differ from permitted split")
    indices = payload["candidates"]
    if indices.ndim != 2 or indices.shape[0] != len(view["image_ids"]):
        raise ValueError("Invalid static mining shape")
    if indices.min() < 0 or indices.max() >= len(view["image_ids"]):
        raise ValueError("Static endpoint outside permitted split")
    if (indices == torch.arange(len(indices))[:,None]).any():
        raise ValueError("Self-match in static mining")
    targets = view["labels"][indices].eq(view["labels"][:,None])
    if not torch.equal(targets, payload["targets"]):
        raise ValueError("Static pair targets disagree with endpoint labels")
