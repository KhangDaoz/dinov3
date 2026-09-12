import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import CUBirds, collate_pil_batch
from dinov3_backbone import extract_last_hidden_state, load_dinov3
from representations import FusionProjection, raw_fusion_features
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

EXPECTED_COUNTS = {"train": 5864, "eval": 5924}


def m3_paths(output_dir):
    root = Path(output_dir)
    return {
        "root": root,
        "raw": root / "cache" / "raw_features.pt",
        "raw_metadata": root / "cache" / "metadata.json",
        "selection": root / "checkpoints" / "selection_best.pt",
        "final": root / "checkpoints" / "final.pt",
        "history": root / "training_history.json",
        "train_embeddings": root / "train_embeddings.pt",
        "test_embeddings": root / "test_embeddings.pt",
        "labels": root / "labels.pt",
        "metrics": root / "metrics.json",
    }


def extract_raw_split(model, processor, device, config, split):
    dataset = CUBirds(config["data_root"], mode=split)
    if len(dataset) != EXPECTED_COUNTS[split]:
        raise RuntimeError(
            f"Split {split} có {len(dataset)} ảnh; cần {EXPECTED_COUNTS[split]}"
        )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_pil_batch,
        pin_memory=device.type == "cuda",
    )
    register_count = int(getattr(model.config, "num_register_tokens", 0))
    cls_batches, mean_batches, label_batches = [], [], []
    patch_counts = set()
    for images, labels in tqdm(loader, desc=f"Raw fusion {split}", unit="batch"):
        tokens, _ = extract_last_hidden_state(model, processor, images, device)
        patch_counts.add(tokens.shape[1] - 1 - register_count)
        cls, mean_patch = raw_fusion_features(tokens, register_count)
        cls_batches.append(cls)
        mean_batches.append(mean_patch)
        label_batches.append(labels)
    if len(patch_counts) != 1 or min(patch_counts) <= 0:
        raise RuntimeError(f"Patch layout không hợp lệ: {sorted(patch_counts)}")
    return {
        "cls": torch.cat(cls_batches),
        "mean_patch": torch.cat(mean_batches),
        "labels": torch.cat(label_batches),
        "paths": [str(path) for path in dataset.im_paths],
        "token_layout": {
            "cls_tokens": 1,
            "register_tokens": register_count,
            "patch_tokens": patch_counts.pop(),
        },
    }


def prepare_raw_cache(config_path, overwrite=False):
    config = load_config(config_path)
    paths = m3_paths(config["output_dir"])
    if paths["raw"].exists() and paths["raw_metadata"].exists() and not overwrite:
        metadata = __import__("json").loads(
            paths["raw_metadata"].read_text(encoding="utf-8")
        )
        if metadata.get("model_revision") != config["model_revision"]:
            raise ValueError("Raw cache dùng model revision khác")
        if metadata.get("sha256") != sha256_file(paths["raw"]):
            raise ValueError("Raw cache checksum không khớp")
        return torch.load(paths["raw"], map_location="cpu", weights_only=True)

    device = resolve_device(config["device"])
    model, processor, _ = load_dinov3(device=device, config_path=config_path)
    started = time.perf_counter()
    payload = {
        split: extract_raw_split(model, processor, device, config, split)
        for split in config["split"]
    }
    if payload["train"]["token_layout"] != payload["eval"]["token_layout"]:
        raise RuntimeError("Token layout train/eval không khớp")
    atomic_torch_save(payload, paths["raw"])
    atomic_json_save(
        {
            "model_name": config["model_name"],
            "model_revision": config["model_revision"],
            "processor": processor.to_dict(),
            "device": str(device),
            "dtype": "float32",
            "token_layout": payload["train"]["token_layout"],
            "counts": {key: len(value["labels"]) for key, value in payload.items()},
            "seconds": time.perf_counter() - started,
            "sha256": sha256_file(paths["raw"]),
        },
        paths["raw_metadata"],
    )
    return payload


def make_components(config, feature_dim, device):
    training = config["training"]
    projection = FusionProjection(feature_dim, config["projection_dim"]).to(device)
    criterion = ProxyAnchorLoss(
        100,
        config["projection_dim"],
        margin=training["margin"],
        alpha=training["alpha"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": projection.parameters(), "lr": training["projection_lr"]},
            {"params": criterion.parameters(), "lr": training["proxy_lr"]},
        ],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training["scheduler_step"],
        gamma=training["scheduler_gamma"],
    )
    return projection, criterion, optimizer, scheduler


def train_one_epoch(projection, criterion, optimizer, loader, device):
    projection.train()
    criterion.train()
    total = 0.0
    for cls, mean_patch, labels in loader:
        cls = cls.to(device)
        mean_patch = mean_patch.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(projection(cls, mean_patch), labels)
        loss.backward()
        torch.nn.utils.clip_grad_value_(projection.parameters(), 10)
        torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate_validation(projection, criterion, features, labels, indices, config, device):
    projection.eval()
    criterion.eval()
    with torch.inference_mode():
        cls = features["cls"][indices].to(device)
        mean_patch = features["mean_patch"][indices].to(device)
        local_labels = labels[indices].to(device)
        embeddings = projection(cls, mean_patch)
        loss = criterion(embeddings, local_labels).item()
    recall = recall_at_k(
        embeddings.cpu(),
        local_labels.cpu(),
        [1],
        config["retrieval_chunk_size"],
        device=device,
    )["recall@1"]
    return loss, recall


def train_stage(config, features, labels, train_indices, validation_indices, epochs, device):
    feature_dim = features["cls"].shape[1]
    projection, criterion, optimizer, scheduler = make_components(
        config, feature_dim, device
    )
    training = config["training"]
    sampler = BalancedBatchSampler(
        labels[train_indices],
        training["classes_per_batch"],
        training["samples_per_class"],
        seed=config["seed"],
    )
    subset = TensorDataset(
        features["cls"][train_indices],
        features["mean_patch"][train_indices],
        labels[train_indices],
    )
    loader = DataLoader(subset, batch_sampler=sampler)
    history = []
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            projection, criterion, optimizer, loader, device
        )
        row = {"epoch": epoch, "train_loss": train_loss}
        if validation_indices is not None:
            validation_loss, recall = evaluate_validation(
                projection,
                criterion,
                features,
                labels,
                validation_indices,
                config,
                device,
            )
            row.update(
                {
                    "validation_loss": validation_loss,
                    "validation_recall@1": recall,
                }
            )
        history.append(row)
        scheduler.step()
    return projection, criterion, optimizer, scheduler, history


def checkpoint_payload(projection, criterion, optimizer, scheduler, config, epoch, raw_hash):
    return {
        "projection_state": projection.state_dict(),
        "proxy_state": criterion.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "raw_cache_sha256": raw_hash,
        "config": public_config(config),
        "architecture": {
            "input_order": ["cls", "mean_patch"],
            "input_dimension": projection.feature_dim * 2,
            "projection_dimension": projection.projection_dim,
        },
    }


def project_all(projection, split, batch_size, device):
    dataset = TensorDataset(split["cls"], split["mean_patch"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batches = []
    projection.eval()
    with torch.inference_mode():
        for cls, mean_patch in loader:
            batches.append(
                projection(cls.to(device), mean_patch.to(device)).float().cpu()
            )
    return torch.cat(batches)


def run_fusion_training(config_path, overwrite=False):
    config = load_config(config_path)
    if config["representation"] != "fusion":
        raise ValueError("train_projection yêu cầu representation='fusion'")
    paths = m3_paths(config["output_dir"])
    final_outputs = [
        paths["final"], paths["train_embeddings"], paths["test_embeddings"],
        paths["labels"], paths["metrics"],
    ]
    if any(path.exists() for path in final_outputs) and not overwrite:
        raise FileExistsError("M3 output đã tồn tại; dùng --overwrite để chạy lại")

    set_reproducibility(config["seed"], config["num_threads"])
    device = resolve_device(config["device"])
    raw = prepare_raw_cache(config_path, overwrite=overwrite)
    train = raw["train"]
    labels = train["labels"].long()
    train_indices, validation_indices = stratified_train_validation_split(
        labels, config["training"]["validation_fraction"], config["seed"]
    )
    selection = train_stage(
        config,
        train,
        labels,
        train_indices,
        validation_indices,
        config["training"]["epochs"],
        device,
    )
    selected_epoch = select_best_epoch(selection[-1])
    raw_hash = sha256_file(paths["raw"])
    set_reproducibility(config["seed"], config["num_threads"])
    selection_best = train_stage(
        config,
        train,
        labels,
        train_indices,
        validation_indices,
        selected_epoch,
        device,
    )
    atomic_torch_save(
        checkpoint_payload(*selection_best[:4], config, selected_epoch, raw_hash),
        paths["selection"],
    )

    all_indices = torch.arange(len(labels))
    set_reproducibility(config["seed"], config["num_threads"])
    final = train_stage(
        config, train, labels, all_indices, None, selected_epoch, device
    )
    atomic_torch_save(
        checkpoint_payload(*final[:4], config, selected_epoch, raw_hash),
        paths["final"],
    )
    train_embeddings = project_all(
        final[0], raw["train"], config["batch_size"], device
    )
    test_embeddings = project_all(
        final[0], raw["eval"], config["batch_size"], device
    )
    labels_payload = {
        "train": {"labels": raw["train"]["labels"], "paths": raw["train"]["paths"]},
        "test": {"labels": raw["eval"]["labels"], "paths": raw["eval"]["paths"]},
    }
    atomic_torch_save(train_embeddings, paths["train_embeddings"])
    atomic_torch_save(test_embeddings, paths["test_embeddings"])
    atomic_torch_save(labels_payload, paths["labels"])
    atomic_json_save(
        {
            "selected_epoch": selected_epoch,
            "selection": selection[-1],
            "final": final[-1],
        },
        paths["history"],
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
                "representation": "fusion",
                "pooling": "raw_cls_concat_raw_mean_patch_linear_projection",
                "embedding_dimension": config["projection_dim"],
                "device": str(device),
                "dtype": "float32",
                "counts": {
                    "train": len(train_embeddings),
                    "test": len(test_embeddings),
                },
                "selected_epoch": selected_epoch,
                "raw_cache_sha256": raw_hash,
            },
            "cache_sha256": hashes,
        },
        paths["metrics"],
    )
    return paths


def run_smoke_test(device):
    torch.manual_seed(42)
    projection = FusionProjection(4, 3).to(device)
    criterion = ProxyAnchorLoss(2, 3).to(device)
    optimizer = torch.optim.AdamW(
        list(projection.parameters()) + list(criterion.parameters()), lr=1e-2
    )
    cls = torch.randn(8, 4, device=device)
    mean_patch = torch.randn(8, 4, device=device)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(projection(cls, mean_patch), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError("Smoke training tạo loss không hữu hạn")
    if losses[-1] >= losses[0]:
        raise RuntimeError("Smoke training không làm giảm loss")
    print(f"smoke_device={device} initial_loss={losses[0]:.6f} final_loss={losses[-1]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train E2A-M3 fusion projection.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e2a_m3.yaml")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke_test:
        run_smoke_test(resolve_device(config["device"]))
    else:
        run_fusion_training(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
