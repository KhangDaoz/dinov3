import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml


REQUIRED_CONFIG_KEYS = {
    "model_name",
    "model_revision",
    "representation",
    "data_root",
    "split",
    "batch_size",
    "num_workers",
    "device",
    "recall_k",
    "retrieval_chunk_size",
    "output_dir",
    "seed",
    "num_threads",
}


def _load_config_mapping(path, seen):
    path = Path(path)
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Phát hiện vòng lặp config extends tại: {path}")
    seen.add(resolved)
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config phải là mapping YAML: {path}")
    parent = config.pop("extends", None)
    if parent is not None:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        inherited = _load_config_mapping(parent_path, seen)
        inherited.update(config)
        config = inherited
    seen.remove(resolved)
    return config


def load_config(path):
    config = _load_config_mapping(Path(path), set())
    missing = sorted(REQUIRED_CONFIG_KEYS - config.keys())
    if missing:
        raise ValueError(f"Config thiếu khóa: {', '.join(missing)}")
    supported_representations = {"cls", "mean_patch", "fusion", "attention_pool"}
    if config["representation"] not in supported_representations:
        supported = ", ".join(sorted(supported_representations))
        raise ValueError(f"representation phải là một trong: {supported}")
    if config["split"] != ["train", "eval"]:
        raise ValueError("E2A-M1 yêu cầu split: [train, eval]")
    for key in ("batch_size", "retrieval_chunk_size", "num_threads"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} phải là số nguyên dương")
    if not isinstance(config["num_workers"], int) or config["num_workers"] < 0:
        raise ValueError("num_workers phải là số nguyên không âm")
    recall_k = config["recall_k"]
    if not recall_k or any(not isinstance(k, int) or k <= 0 for k in recall_k):
        raise ValueError("recall_k phải là danh sách số nguyên dương")
    if config["representation"] in {"fusion", "attention_pool"}:
        representation = config["representation"]
        if representation == "attention_pool":
            hidden_dim = config.get("attention_hidden_dim")
            if not isinstance(hidden_dim, int) or hidden_dim <= 0:
                raise ValueError("attention_pool yêu cầu attention_hidden_dim dương")
            patch_cache = config.get("patch_cache")
            required_cache = {
                "dtype", "shard_size", "max_cached_shards", "prefetch_factor"
            }
            if not isinstance(patch_cache, dict):
                raise ValueError("attention_pool yêu cầu patch_cache config")
            missing_cache = sorted(required_cache - patch_cache.keys())
            if missing_cache:
                raise ValueError(
                    f"patch_cache thiếu khóa: {', '.join(missing_cache)}"
                )
            if patch_cache["dtype"] != "float32":
                raise ValueError("M4 canonical yêu cầu patch_cache.dtype='float32'")
            for key in ("shard_size", "max_cached_shards", "prefetch_factor"):
                if not isinstance(patch_cache[key], int) or patch_cache[key] <= 0:
                    raise ValueError(f"patch_cache.{key} phải là số nguyên dương")
        else:
            if not isinstance(config.get("projection_dim"), int) or config["projection_dim"] <= 0:
                raise ValueError("fusion yêu cầu projection_dim là số nguyên dương")
        training = config.get("training")
        required_training = {
            "loss", "epochs", "classes_per_batch", "samples_per_class",
            "validation_fraction", "alpha", "margin",
            "proxy_lr", "weight_decay", "scheduler_step", "scheduler_gamma",
        }
        learning_rate_key = (
            "projection_lr" if representation == "fusion" else "attention_lr"
        )
        required_training.add(learning_rate_key)
        if not isinstance(training, dict):
            raise ValueError("fusion yêu cầu training config")
        missing_training = sorted(required_training - training.keys())
        if missing_training:
            raise ValueError(
                f"training config thiếu khóa: {', '.join(missing_training)}"
            )
        if training["loss"] != "proxy_anchor":
            raise ValueError("Learned E2A representations yêu cầu proxy_anchor")
        for key in ("epochs", "classes_per_batch", "samples_per_class", "scheduler_step"):
            if not isinstance(training[key], int) or training[key] <= 0:
                raise ValueError(f"training.{key} phải là số nguyên dương")
        if not 0 < training["validation_fraction"] < 1:
            raise ValueError("training.validation_fraction phải nằm trong (0, 1)")
    if config.get("experiment") == "e1":
        _validate_e1_config(config)
    if config.get("experiment") == "e2b":
        _validate_e2b_config(config)
    return config


def _validate_e1_config(config):
    if config["representation"] != "cls":
        raise ValueError("E1 yêu cầu representation='cls'")
    if not isinstance(config.get("source_output_dir"), str):
        raise ValueError("E1 yêu cầu source_output_dir")
    head = config.get("evidential_head")
    training = config.get("training")
    calibration = config.get("calibration")
    reranking = config.get("reranking")
    required_head = {"hidden_dim", "num_classes", "dropout", "evidence_activation"}
    required_training = {
        "epochs", "validation_fraction", "learning_rate", "weight_decay",
        "annealing_epochs", "gradient_clip",
    }
    if not isinstance(head, dict) or required_head - head.keys():
        raise ValueError("E1 evidential_head config không đầy đủ")
    if head["evidence_activation"] != "softplus":
        raise ValueError("E1 hiện chỉ hỗ trợ evidence_activation='softplus'")
    for key in ("hidden_dim", "num_classes"):
        if not isinstance(head[key], int) or head[key] <= 0:
            raise ValueError(f"evidential_head.{key} phải là số nguyên dương")
    if not 0 <= head["dropout"] < 1:
        raise ValueError("evidential_head.dropout phải nằm trong [0, 1)")
    if not isinstance(training, dict) or required_training - training.keys():
        raise ValueError("E1 training config không đầy đủ")
    for key in ("epochs", "annealing_epochs"):
        if not isinstance(training[key], int) or training[key] <= 0:
            raise ValueError(f"training.{key} phải là số nguyên dương")
    if not 0 < training["validation_fraction"] < 1:
        raise ValueError("training.validation_fraction phải nằm trong (0, 1)")
    if not isinstance(calibration, dict) or not isinstance(
        calibration.get("ece_bins"), int
    ) or calibration["ece_bins"] <= 0:
        raise ValueError("E1 calibration.ece_bins phải là số nguyên dương")
    grid = reranking.get("top_n_grid") if isinstance(reranking, dict) else None
    if not isinstance(grid, list) or not grid:
        raise ValueError("E1 reranking.top_n_grid phải là danh sách")
    if any(not isinstance(value, int) or value <= 0 for value in grid):
        raise ValueError("E1 top_n_grid chỉ nhận số nguyên dương")
    if grid != sorted(set(grid)):
        raise ValueError("E1 top_n_grid phải tăng dần và không trùng")


def _validate_e2b_config(config):
    if config["representation"] != "cls":
        raise ValueError("E2B yêu cầu representation='cls'")
    for key in ("source_output_dir", "e1_output_dir"):
        if not isinstance(config.get(key), str):
            raise ValueError(f"E2B yêu cầu {key}")
    network = config.get("pair_network")
    sampling = config.get("pair_sampling")
    training = config.get("training")
    retrieval = config.get("retrieval")
    if not isinstance(network, dict):
        raise ValueError("E2B yêu cầu pair_network")
    if network.get("feature_mode") not in {"full", "ordered", "difference"}:
        raise ValueError("pair_network.feature_mode không hợp lệ")
    hidden = network.get("hidden_dims")
    if not isinstance(hidden, list) or len(hidden) != 2 or min(hidden) <= 0:
        raise ValueError("pair_network.hidden_dims phải có hai số dương")
    if not 0 <= network.get("dropout", -1) < 1:
        raise ValueError("pair_network.dropout không hợp lệ")
    required_sampling = {
        "positive_per_anchor", "negative_per_anchor", "negative_bank_size"
    }
    if not isinstance(sampling, dict) or required_sampling - sampling.keys():
        raise ValueError("E2B pair_sampling không đầy đủ")
    if sampling["positive_per_anchor"] != 2 or sampling["negative_per_anchor"] != 2:
        raise ValueError("Canonical E2B yêu cầu 2 positive và 2 negative")
    required_training = {
        "epochs", "validation_fraction", "anchors_per_batch", "learning_rate",
        "weight_decay", "gradient_clip",
    }
    if not isinstance(training, dict) or required_training - training.keys():
        raise ValueError("E2B training không đầy đủ")
    if not 0 < training["validation_fraction"] < 1:
        raise ValueError("E2B validation_fraction không hợp lệ")
    if not isinstance(retrieval, dict):
        raise ValueError("E2B yêu cầu retrieval config")
    candidates = retrieval.get("candidate_top_n_grid")
    weights = retrieval.get("lambda_grid")
    if not candidates or candidates != sorted(set(candidates)):
        raise ValueError("candidate_top_n_grid không hợp lệ")
    if not weights or any(not 0 <= value <= 1 for value in weights):
        raise ValueError("lambda_grid không hợp lệ")


def public_config(config):
    """Return metadata-safe configuration without changing the source config."""
    return {key: value for key, value in config.items() if key != "token"}


def set_reproducibility(seed, num_threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config yêu cầu CUDA nhưng CUDA không khả dụng")
    return device


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_path(target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    return Path(name)


def atomic_torch_save(value, target):
    target = Path(target)
    temporary = _atomic_path(target)
    try:
        torch.save(value, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(value, target):
    target = Path(target)
    temporary = _atomic_path(target)
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)
