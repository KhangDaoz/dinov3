import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval import recall_at_k
from utils import (
    atomic_json_save,
    load_config,
    load_json,
    public_config,
    resolve_device,
    sha256_file,
)


def artifact_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "train_embeddings.pt": output_dir / "train_embeddings.pt",
        "test_embeddings.pt": output_dir / "test_embeddings.pt",
        "labels.pt": output_dir / "labels.pt",
        "metrics.json": output_dir / "metrics.json",
    }


def verify_hashes(paths, expected_hashes):
    for name in ("train_embeddings.pt", "test_embeddings.pt", "labels.pt"):
        expected = expected_hashes.get(name)
        if not expected:
            raise ValueError(f"Metadata thiếu SHA-256 cho {name}")
        actual = sha256_file(paths[name])
        if actual != expected:
            raise ValueError(f"Cache checksum không khớp: {name}")


def validate_cache(train_embeddings, test_embeddings, labels_payload):
    if not isinstance(labels_payload, dict):
        raise ValueError("labels.pt phải là dictionary")
    for split in ("train", "test"):
        if split not in labels_payload:
            raise ValueError(f"labels.pt thiếu split {split!r}")
        payload = labels_payload[split]
        if set(payload) != {"labels", "paths"}:
            raise ValueError(f"Payload {split!r} phải chứa labels và paths")
    expected = {
        "train": (train_embeddings, 5864),
        "test": (test_embeddings, 5924),
    }
    for split, (embeddings, count) in expected.items():
        labels = labels_payload[split]["labels"]
        paths = labels_payload[split]["paths"]
        if embeddings.shape != (count, 768):
            raise ValueError(
                f"{split} embeddings có shape {tuple(embeddings.shape)}; "
                f"cần {(count, 768)}"
            )
        if labels.shape != (count,) or len(paths) != count:
            raise ValueError(f"{split} labels/paths không khớp embeddings")
        if embeddings.device.type != "cpu" or embeddings.dtype != torch.float32:
            raise ValueError(f"{split} embeddings phải là CPU float32")


def run_evaluation(config_path):
    config = load_config(config_path)
    device = resolve_device(config["device"])
    paths = artifact_paths(config["output_dir"])
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu artifacts: {', '.join(missing)}")

    metrics = load_json(paths["metrics.json"])
    verify_hashes(paths, metrics.get("cache_sha256", {}))
    train_embeddings = torch.load(
        paths["train_embeddings.pt"], map_location="cpu", weights_only=True
    )
    test_embeddings = torch.load(
        paths["test_embeddings.pt"], map_location="cpu", weights_only=True
    )
    labels_payload = torch.load(
        paths["labels.pt"], map_location="cpu", weights_only=True
    )
    validate_cache(train_embeddings, test_embeddings, labels_payload)

    started = time.perf_counter()
    recalls = recall_at_k(
        test_embeddings,
        labels_payload["test"]["labels"],
        config["recall_k"],
        config["retrieval_chunk_size"],
        device=device,
    )
    metrics.update(
        {
            "status": "evaluated",
            "config": public_config(config),
            "evaluation": {
                "metrics": recalls,
                "num_queries": int(test_embeddings.shape[0]),
                "num_gallery": int(test_embeddings.shape[0]),
                "similarity": "cosine",
                "self_exclusion": True,
                "tie_policy": "stable_gallery_index_ascending",
                "chunk_size": config["retrieval_chunk_size"],
                "device": str(device),
                "seconds": time.perf_counter() - started,
            },
        }
    )
    atomic_json_save(metrics, paths["metrics.json"])
    return recalls


def main():
    parser = argparse.ArgumentParser(description="Evaluate E2A-M1 CLS retrieval.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e2a.yaml")
    )
    args = parser.parse_args()
    recalls = run_evaluation(args.config)
    for name, value in recalls.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
