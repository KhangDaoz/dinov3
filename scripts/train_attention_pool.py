import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import CUBirds, collate_pil_batch
from dinov3_backbone import extract_last_hidden_state, load_dinov3
from patch_cache import PatchShardDataset, validate_patch_manifest
from representations import AttentionPool, raw_patch_tokens
from retrieval import recall_at_k
from train_projection import (
    BalancedBatchSampler,
    ProxyAnchorLoss,
    select_best_epoch,
    stratified_train_validation_split,
)
from utils import (
    atomic_json_save,
    atomic_torch_save,
    load_config,
    public_config,
    resolve_device,
    set_reproducibility,
    sha256_file,
)

EXPECTED_COUNTS = {"train": 5864, "test": 5924}


def attention_paths(output_dir):
    root = Path(output_dir)
    return {
        "root": root,
        "cache": root / "cache",
        "manifest": root / "cache" / "manifest.json",
        "selection": root / "checkpoints" / "selection_best.pt",
        "final": root / "checkpoints" / "final.pt",
        "history": root / "training_history.json",
        "diagnostics": root / "attention_diagnostics.json",
        "train_embeddings": root / "train_embeddings.pt",
        "test_embeddings": root / "test_embeddings.pt",
        "labels": root / "labels.pt",
        "metrics": root / "metrics.json",
    }


def to_compute_device(tensor, device):
    if device.type != "cuda":
        return tensor.to(device)
    if tensor.device.type == "cpu" and not tensor.is_pinned():
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=True)


def loader_options(config, device, multi_epoch=False):
    workers = config["num_workers"]
    options = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        options["persistent_workers"] = multi_epoch
        options["prefetch_factor"] = config["patch_cache"]["prefetch_factor"]
    return options


def extract_split_shards(model, processor, device, config, split, cache_root):
    dataset_mode = "eval" if split == "test" else "train"
    dataset = CUBirds(config["data_root"], mode=dataset_mode)
    if len(dataset) != EXPECTED_COUNTS[split]:
        raise RuntimeError(
            f"Split {split} có {len(dataset)} ảnh; cần {EXPECTED_COUNTS[split]}"
        )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_pil_batch,
        **loader_options(config, device),
    )
    register_count = int(getattr(model.config, "num_register_tokens", 0))
    shard_size = config["patch_cache"]["shard_size"]
    split_dir = cache_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    buffer = []
    buffer_rows = 0
    shards = []
    start = 0
    patch_count = None

    def flush(rows):
        nonlocal buffer, buffer_rows, start
        joined = torch.cat(buffer)
        current, remainder = joined[:rows], joined[rows:]
        shard_index = len(shards)
        path = split_dir / f"part-{shard_index:05d}.pt"
        atomic_torch_save({"patches": current.contiguous()}, path)
        stop = start + len(current)
        shards.append(
            {
                "path": str(path.relative_to(cache_root)),
                "start": start,
                "stop": stop,
                "shape": list(current.shape),
                "dtype": str(current.dtype).removeprefix("torch."),
                "sha256": sha256_file(path),
            }
        )
        start = stop
        buffer = [remainder] if len(remainder) else []
        buffer_rows = len(remainder)

    for images, _ in tqdm(loader, desc=f"Patch cache {split}", unit="batch"):
        tokens, _ = extract_last_hidden_state(model, processor, images, device)
        patches = raw_patch_tokens(tokens, register_count)
        if patch_count is None:
            patch_count = patches.shape[1]
        elif patches.shape[1] != patch_count:
            raise RuntimeError("Số patch tokens thay đổi giữa các batch")
        buffer.append(patches)
        buffer_rows += len(patches)
        while buffer_rows >= shard_size:
            flush(shard_size)
    if buffer_rows:
        flush(buffer_rows)
    return {
        "count": len(dataset),
        "labels": list(dataset.ys),
        "paths": [str(path) for path in dataset.im_paths],
        "shards": shards,
        "token_layout": {
            "cls_tokens": 1,
            "register_tokens": register_count,
            "patch_tokens": patch_count,
        },
    }


def prepare_patch_cache(config_path, overwrite=False):
    config = load_config(config_path)
    paths = attention_paths(config["output_dir"])
    if paths["manifest"].exists() and not overwrite:
        with paths["manifest"].open(encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("model_revision") != config["model_revision"]:
            raise ValueError("Patch cache dùng model revision khác")
        validate_patch_manifest(manifest, paths["cache"], verify_hashes=True)
        return manifest

    device = resolve_device(config["device"])
    model, processor, _ = load_dinov3(device=device, config_path=config_path)
    started = time.perf_counter()
    splits = {
        split: extract_split_shards(
            model, processor, device, config, split, paths["cache"]
        )
        for split in ("train", "test")
    }
    if splits["train"]["token_layout"] != splits["test"]["token_layout"]:
        raise RuntimeError("Token layout train/test không khớp")
    manifest = {
        "format_version": 1,
        "model_name": config["model_name"],
        "model_revision": config["model_revision"],
        "processor": processor.to_dict(),
        "device": str(device),
        "dtype": config["patch_cache"]["dtype"],
        "token_layout": splits["train"]["token_layout"],
        "config": public_config(config),
        "seconds": time.perf_counter() - started,
        "splits": splits,
    }
    atomic_json_save(manifest, paths["manifest"])
    validate_patch_manifest(manifest, paths["cache"], verify_hashes=True)
    return manifest


def make_components(config, feature_dim, device):
    training = config["training"]
    model = AttentionPool(feature_dim, config["attention_hidden_dim"]).to(device)
    criterion = ProxyAnchorLoss(
        100,
        feature_dim,
        margin=training["margin"],
        alpha=training["alpha"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": training["attention_lr"]},
            {"params": criterion.parameters(), "lr": training["proxy_lr"]},
        ],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training["scheduler_step"],
        gamma=training["scheduler_gamma"],
    )
    return model, criterion, optimizer, scheduler


def train_epoch(model, criterion, optimizer, loader, device):
    model.train()
    criterion.train()
    total = 0.0
    for patches, labels in loader:
        patches = to_compute_device(patches, device)
        labels = to_compute_device(labels, device)
        if next(model.parameters()).device != patches.device:
            raise RuntimeError("Attention model và batch không cùng device")
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(patches), labels)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate_subset(model, criterion, dataset, indices, config, device):
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=config["batch_size"],
        shuffle=False,
        **loader_options(config, device),
    )
    embeddings, labels = [], []
    losses = []
    model.eval()
    criterion.eval()
    with torch.inference_mode():
        for patches, batch_labels in loader:
            patches = to_compute_device(patches, device)
            batch_labels = to_compute_device(batch_labels, device)
            batch_embeddings = model(patches)
            embeddings.append(batch_embeddings.cpu())
            labels.append(batch_labels.cpu())
            losses.append(criterion(batch_embeddings, batch_labels).item())
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    recall = recall_at_k(
        embeddings,
        labels,
        [1],
        config["retrieval_chunk_size"],
        device=device,
    )["recall@1"]
    return sum(losses) / len(losses), recall


def train_stage(config, dataset, train_indices, validation_indices, epochs, device):
    feature_dim = dataset[0][0].shape[-1]
    model, criterion, optimizer, scheduler = make_components(
        config, feature_dim, device
    )
    local_labels = dataset.labels[train_indices]
    sampler = BalancedBatchSampler(
        local_labels,
        config["training"]["classes_per_batch"],
        config["training"]["samples_per_class"],
        seed=config["seed"],
    )
    loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_sampler=sampler,
        **loader_options(config, device, multi_epoch=True),
    )
    history = []
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        row = {
            "epoch": epoch,
            "train_loss": train_epoch(
                model, criterion, optimizer, loader, device
            ),
        }
        if validation_indices is not None:
            loss, recall = evaluate_subset(
                model, criterion, dataset, validation_indices, config, device
            )
            row.update(
                {"validation_loss": loss, "validation_recall@1": recall}
            )
        history.append(row)
        scheduler.step()
    return model, criterion, optimizer, scheduler, history


def checkpoint_payload(stage, config, epoch, manifest_hash):
    model, criterion, optimizer, scheduler, _ = stage
    return {
        "attention_state": model.state_dict(),
        "proxy_state": criterion.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "manifest_sha256": manifest_hash,
        "config": public_config(config),
        "architecture": {
            "feature_dim": model.feature_dim,
            "hidden_dim": model.hidden_dim,
        },
    }


def export_split(model, dataset, config, device):
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        **loader_options(config, device),
    )
    embeddings = []
    entropy_sum = max_weight_sum = effective_sum = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for patches, _ in loader:
            patches = to_compute_device(patches, device)
            batch_embeddings, weights = model(patches, return_weights=True)
            embeddings.append(batch_embeddings.float().cpu())
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
            entropy_sum += entropy.sum().item()
            max_weight_sum += weights.max(dim=1).values.sum().item()
            effective_sum += torch.exp(entropy).sum().item()
            count += len(weights)
    diagnostics = {
        "mean_entropy": entropy_sum / count,
        "mean_max_weight": max_weight_sum / count,
        "mean_effective_patches": effective_sum / count,
    }
    return torch.cat(embeddings), diagnostics


def run_attention_training(config_path, overwrite=False):
    config = load_config(config_path)
    if config["representation"] != "attention_pool":
        raise ValueError("Attention training yêu cầu representation=attention_pool")
    paths = attention_paths(config["output_dir"])
    final_outputs = [
        paths["final"], paths["train_embeddings"], paths["test_embeddings"],
        paths["labels"], paths["metrics"],
    ]
    if any(path.exists() for path in final_outputs) and not overwrite:
        raise FileExistsError("M4 output đã tồn tại; dùng --overwrite để chạy lại")
    set_reproducibility(config["seed"], config["num_threads"])
    device = resolve_device(config["device"])
    manifest = prepare_patch_cache(config_path, overwrite=overwrite)
    manifest_hash = sha256_file(paths["manifest"])
    train_dataset = PatchShardDataset(
        manifest,
        "train",
        paths["cache"],
        config["patch_cache"]["max_cached_shards"],
    )
    labels = train_dataset.labels
    train_indices, validation_indices = stratified_train_validation_split(
        labels, config["training"]["validation_fraction"], config["seed"]
    )
    selection = train_stage(
        config,
        train_dataset,
        train_indices,
        validation_indices,
        config["training"]["epochs"],
        device,
    )
    selected_epoch = select_best_epoch(selection[-1])
    set_reproducibility(config["seed"], config["num_threads"])
    best = train_stage(
        config, train_dataset, train_indices, validation_indices,
        selected_epoch, device,
    )
    atomic_torch_save(
        checkpoint_payload(best, config, selected_epoch, manifest_hash),
        paths["selection"],
    )
    set_reproducibility(config["seed"], config["num_threads"])
    final = train_stage(
        config, train_dataset, torch.arange(len(labels)), None,
        selected_epoch, device,
    )
    atomic_torch_save(
        checkpoint_payload(final, config, selected_epoch, manifest_hash),
        paths["final"],
    )
    train_embeddings, train_diagnostics = export_split(
        final[0], train_dataset, config, device
    )
    test_dataset = PatchShardDataset(
        manifest,
        "test",
        paths["cache"],
        config["patch_cache"]["max_cached_shards"],
    )
    test_embeddings, test_diagnostics = export_split(
        final[0], test_dataset, config, device
    )
    atomic_torch_save(train_embeddings, paths["train_embeddings"])
    atomic_torch_save(test_embeddings, paths["test_embeddings"])
    atomic_torch_save(
        {
            "train": {
                "labels": train_dataset.labels,
                "paths": train_dataset.paths,
            },
            "test": {
                "labels": test_dataset.labels,
                "paths": test_dataset.paths,
            },
        },
        paths["labels"],
    )
    atomic_json_save(
        {
            "selected_epoch": selected_epoch,
            "selection": selection[-1],
            "final": final[-1],
        },
        paths["history"],
    )
    atomic_json_save(
        {"train": train_diagnostics, "test": test_diagnostics},
        paths["diagnostics"],
    )
    hashes = {
        "train_embeddings.pt": sha256_file(paths["train_embeddings"]),
        "test_embeddings.pt": sha256_file(paths["test_embeddings"]),
        "labels.pt": sha256_file(paths["labels"]),
    }
    atomic_json_save(
        {
            "status": "extracted",
            "config": public_config(config),
            "extraction": {
                "representation": "attention_pool",
                "pooling": "learned_additive_attention_over_spatial_patches",
                "embedding_dimension": train_embeddings.shape[1],
                "device": str(device),
                "dtype": "float32",
                "counts": {
                    "train": len(train_embeddings),
                    "test": len(test_embeddings),
                },
                "selected_epoch": selected_epoch,
                "manifest_sha256": manifest_hash,
            },
            "cache_sha256": hashes,
        },
        paths["metrics"],
    )
    return paths


def run_smoke_test(device):
    torch.manual_seed(42)
    model = AttentionPool(4, 3).to(device)
    criterion = ProxyAnchorLoss(2, 4).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), lr=1e-2
    )
    patches = torch.randn(8, 5, 4, device=device)
    labels = torch.tensor([0] * 4 + [1] * 4, device=device)
    losses = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(patches), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if not torch.isfinite(torch.tensor(losses)).all() or losses[-1] >= losses[0]:
        raise RuntimeError("Attention smoke training không giảm finite loss")
    print(
        f"smoke_device={device} initial_loss={losses[0]:.6f} "
        f"final_loss={losses[-1]:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train E2A-M4 attention pooling.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e2a_m4.yaml")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke_test:
        run_smoke_test(resolve_device(config["device"]))
    else:
        run_attention_training(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
