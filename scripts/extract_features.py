import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import CUBirds, collate_pil_batch
from dinov3_backbone import extract_last_hidden_state, load_dinov3
from representations import build_embedding
from utils import (
    atomic_json_save,
    atomic_torch_save,
    load_config,
    public_config,
    resolve_device,
    set_reproducibility,
    sha256_file,
)


EXPECTED_SPLIT_COUNTS = {"train": 5864, "eval": 5924}


def extract_split(model, processor, device, config, split):
    dataset = CUBirds(config["data_root"], mode=split)
    expected_count = EXPECTED_SPLIT_COUNTS[split]
    if len(dataset) != expected_count:
        raise RuntimeError(
            f"Split {split!r} có {len(dataset)} ảnh; cần {expected_count}"
        )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_pil_batch,
        pin_memory=device.type == "cuda",
    )
    embedding_batches = []
    label_batches = []
    paths = []
    offset = 0
    register_count = int(getattr(model.config, "num_register_tokens", 0))
    observed_patch_counts = set()
    for images, labels in tqdm(loader, desc=f"Extract {split}", unit="batch"):
        tokens, _ = extract_last_hidden_state(model, processor, images, device)
        patch_count = tokens.shape[1] - 1 - register_count
        if patch_count <= 0:
            raise RuntimeError("Backbone output không chứa patch token")
        observed_patch_counts.add(patch_count)
        embeddings = build_embedding(
            config["representation"], tokens, register_count
        )
        if embeddings.shape[0] != labels.shape[0]:
            raise RuntimeError("Batch embeddings và labels không khớp")
        embedding_batches.append(embeddings)
        label_batches.append(labels)
        paths.extend(str(path) for path in dataset.im_paths[offset:offset + len(images)])
        offset += len(images)

    embeddings = torch.cat(embedding_batches)
    labels = torch.cat(label_batches)
    if embeddings.shape != (expected_count, model.config.hidden_size):
        raise RuntimeError(f"Embedding shape không hợp lệ: {tuple(embeddings.shape)}")
    if len(paths) != expected_count:
        raise RuntimeError("Số paths không khớp embeddings")
    if len(observed_patch_counts) != 1:
        raise RuntimeError(
            f"Số patch tokens không nhất quán: {sorted(observed_patch_counts)}"
        )
    token_layout = {
        "cls_tokens": 1,
        "register_tokens": register_count,
        "patch_tokens": observed_patch_counts.pop(),
    }
    return embeddings, labels, paths, token_layout


def output_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "train": output_dir / "train_embeddings.pt",
        "eval": output_dir / "test_embeddings.pt",
        "labels": output_dir / "labels.pt",
        "metrics": output_dir / "metrics.json",
    }


def run_extraction(config_path, overwrite=False):
    config = load_config(config_path)
    paths = output_paths(config["output_dir"])
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output đã tồn tại; dùng --overwrite: {names}")

    set_reproducibility(config["seed"], config["num_threads"])
    device = resolve_device(config["device"])
    started = time.perf_counter()
    model, processor, _ = load_dinov3(device=device, config_path=config_path)

    split_results = {}
    split_timings = {}
    for split in config["split"]:
        split_started = time.perf_counter()
        split_results[split] = extract_split(
            model, processor, device, config, split
        )
        split_timings[split] = time.perf_counter() - split_started

    train_embeddings, train_labels, train_paths, train_layout = split_results["train"]
    test_embeddings, test_labels, test_paths, test_layout = split_results["eval"]
    if train_layout != test_layout:
        raise RuntimeError(
            f"Token layout train/eval không khớp: {train_layout} != {test_layout}"
        )
    labels_payload = {
        "train": {"labels": train_labels, "paths": train_paths},
        "test": {"labels": test_labels, "paths": test_paths},
    }
    atomic_torch_save(train_embeddings, paths["train"])
    atomic_torch_save(test_embeddings, paths["eval"])
    atomic_torch_save(labels_payload, paths["labels"])
    hashes = {
        "train_embeddings.pt": sha256_file(paths["train"]),
        "test_embeddings.pt": sha256_file(paths["eval"]),
        "labels.pt": sha256_file(paths["labels"]),
    }
    metrics = {
        "status": "extracted",
        "config": public_config(config),
        "extraction": {
            "model_name": config["model_name"],
            "model_revision": config["model_revision"],
            "processor": processor.to_dict(),
            "device": str(device),
            "dtype": "float32",
            "representation": config["representation"],
            "pooling": (
                "final_layer_cls"
                if config["representation"] == "cls"
                else "final_layer_mean_spatial_patches"
            ),
            "token_layout": train_layout,
            "embedding_dimension": int(train_embeddings.shape[1]),
            "counts": {"train": len(train_labels), "test": len(test_labels)},
            "split_seconds": split_timings,
            "total_seconds": time.perf_counter() - started,
        },
        "cache_sha256": hashes,
    }
    atomic_json_save(metrics, paths["metrics"])
    return paths


def main():
    parser = argparse.ArgumentParser(description="Extract E2A embeddings.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e2a.yaml")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_extraction(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
