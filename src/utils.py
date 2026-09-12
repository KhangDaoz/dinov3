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


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config phải là mapping YAML: {path}")
    missing = sorted(REQUIRED_CONFIG_KEYS - config.keys())
    if missing:
        raise ValueError(f"Config thiếu khóa: {', '.join(missing)}")
    if config["representation"] != "cls":
        raise ValueError("E2A-M1 chỉ hỗ trợ representation='cls'")
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
    return config


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
