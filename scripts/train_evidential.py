import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidential import (
    EvidentialHead,
    annealing_coefficient,
    classification_metrics,
    dirichlet_statistics,
    edl_mse_loss,
    select_evidential_epoch,
    select_top_n,
)
from retrieval import (
    cosine_top_indices,
    recall_from_indices,
    rerank_top_n_by_uncertainty,
)
from train_projection import stratified_train_validation_split
from utils import (
    atomic_json_save,
    atomic_torch_save,
    load_config,
    load_json,
    public_config,
    resolve_device,
    set_reproducibility,
    sha256_file,
)


def e1_paths(output_dir):
    root = Path(output_dir)
    return {
        "root": root,
        "checkpoint": root / "checkpoint.pt",
        "history": root / "training_history.json",
        "validation": root / "validation_predictions.pt",
        "test_uncertainty": root / "test_uncertainty.pt",
        "metrics": root / "metrics.json",
        "manifest": root / "manifest.json",
        "reranking": root / "reranking_scores.pt",
    }


def _optional_mean(values):
    return values.mean().item() if values.numel() else None


def _validate_cls_cache(train_embeddings, test_embeddings, labels_payload):
    expected = {"train": (train_embeddings, 5864), "test": (test_embeddings, 5924)}
    if not isinstance(labels_payload, dict):
        raise ValueError("CLS labels payload phải là dictionary")
    for split, (embeddings, count) in expected.items():
        payload = labels_payload.get(split)
        if not isinstance(payload, dict) or set(payload) != {"labels", "paths"}:
            raise ValueError(f"CLS labels payload không hợp lệ cho {split}")
        labels, paths = payload["labels"], payload["paths"]
        if embeddings.shape != (count, 768) or embeddings.dtype != torch.float32:
            raise ValueError(f"CLS {split} embeddings không đúng shape/dtype")
        if embeddings.device.type != "cpu":
            raise ValueError("CLS cache phải được lưu trên CPU")
        if labels.shape != (count,) or len(paths) != count:
            raise ValueError(f"CLS {split} labels/paths không đồng bộ")
        if not torch.isfinite(embeddings).all().item():
            raise ValueError(f"CLS {split} embeddings chứa NaN/Inf")
        norms = torch.linalg.vector_norm(embeddings, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
            raise ValueError(f"CLS {split} embeddings chưa chuẩn hóa L2")
    train_labels = labels_payload["train"]["labels"]
    if sorted(train_labels.unique().tolist()) != list(range(100)):
        raise ValueError("E1 yêu cầu train labels cục bộ 0..99")


def load_cls_cache(config):
    root = Path(config["source_output_dir"])
    files = {
        "train": root / "train_embeddings.pt",
        "test": root / "test_embeddings.pt",
        "labels": root / "labels.pt",
        "metrics": root / "metrics.json",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu CLS artifacts: {', '.join(missing)}")
    metadata = load_json(files["metrics"])
    if metadata.get("extraction", {}).get("representation") != "cls":
        raise ValueError("E1 source cache không phải CLS")
    expected_hashes = metadata.get("cache_sha256", {})
    for name, key in (
        ("train_embeddings.pt", "train"),
        ("test_embeddings.pt", "test"),
        ("labels.pt", "labels"),
    ):
        expected = expected_hashes.get(name)
        if not expected or sha256_file(files[key]) != expected:
            raise ValueError(f"CLS cache checksum không khớp: {name}")
    train_embeddings = torch.load(files["train"], map_location="cpu", weights_only=True)
    test_embeddings = torch.load(files["test"], map_location="cpu", weights_only=True)
    labels = torch.load(files["labels"], map_location="cpu", weights_only=True)
    _validate_cls_cache(train_embeddings, test_embeddings, labels)
    return train_embeddings, test_embeddings, labels, {
        name: sha256_file(files[key])
        for name, key in (
            ("train_embeddings.pt", "train"),
            ("test_embeddings.pt", "test"),
            ("labels.pt", "labels"),
        )
    }


def _loader_options(config, device, multi_epoch=False):
    workers = config["num_workers"]
    options = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        options["prefetch_factor"] = 2
        options["persistent_workers"] = multi_epoch
    return options


def make_head(config, input_dim, device):
    head_config = config["evidential_head"]
    return EvidentialHead(
        input_dim=input_dim,
        hidden_dim=head_config["hidden_dim"],
        num_classes=head_config["num_classes"],
        dropout=head_config["dropout"],
    ).to(device)


def _predict(head, dataset, indices, config, device):
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=config["batch_size"],
        shuffle=False,
        **_loader_options(config, device),
    )
    evidence_batches, label_batches = [], []
    head.eval()
    with torch.inference_mode():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device, non_blocking=device.type == "cuda")
            evidence_batches.append(head(embeddings).cpu())
            label_batches.append(labels.cpu())
    evidence = torch.cat(evidence_batches)
    labels = torch.cat(label_batches)
    alpha, _, probabilities, uncertainty = dirichlet_statistics(evidence)
    return {
        "evidence": evidence,
        "alpha": alpha,
        "probabilities": probabilities,
        "uncertainty": uncertainty,
        "labels": labels,
        "indices": indices.cpu(),
    }


def _train_stage(config, dataset, train_indices, validation_indices, epochs, device):
    head = make_head(config, dataset.tensors[0].shape[1], device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        **_loader_options(config, device, multi_epoch=True),
    )
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if next(head.parameters()).device != embeddings.device:
                raise RuntimeError("Evidential Head và batch không cùng device")
            optimizer.zero_grad(set_to_none=True)
            evidence = head(embeddings)
            loss = edl_mse_loss(
                evidence,
                labels,
                annealing_coefficient(epoch, training["annealing_epochs"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), training["gradient_clip"]
            )
            optimizer.step()
            total_loss += loss.item() * len(labels)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_indices),
        }
        if validation_indices is not None:
            prediction = _predict(
                head, dataset, validation_indices, config, device
            )
            validation_loss = edl_mse_loss(
                prediction["evidence"].to(device),
                prediction["labels"].to(device),
                annealing=1.0,
            ).item()
            validation_metrics = classification_metrics(
                prediction["probabilities"],
                prediction["labels"],
                bins=config["calibration"]["ece_bins"],
            )
            row.update(
                {
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_metrics["accuracy"],
                }
            )
        history.append(row)
    return head, optimizer, history


def _choose_top_n(embeddings, labels, uncertainty, config, device):
    grid = config["reranking"]["top_n_grid"]
    max_n = max(grid)
    indices = cosine_top_indices(
        embeddings,
        max_n,
        config["retrieval_chunk_size"],
        device=device,
    )
    results = []
    for top_n in grid:
        reranked = rerank_top_n_by_uncertainty(indices, uncertainty, top_n)
        results.append(
            {
                "top_n": top_n,
                "metrics": recall_from_indices(
                    reranked, labels, config["recall_k"]
                ),
            }
        )
    return select_top_n(results), results


def run_e1_training(config_path, overwrite=False):
    config = load_config(config_path)
    if config.get("experiment") != "e1":
        raise ValueError("train_evidential yêu cầu experiment='e1'")
    paths = e1_paths(config["output_dir"])
    protected = [
        paths["checkpoint"], paths["history"], paths["validation"],
        paths["test_uncertainty"], paths["metrics"], paths["manifest"],
    ]
    if any(path.exists() for path in protected) and not overwrite:
        raise FileExistsError("E1 output đã tồn tại; dùng --overwrite để chạy lại")

    set_reproducibility(config["seed"], config["num_threads"])
    device = resolve_device(config["device"])
    started = time.perf_counter()
    train_embeddings, test_embeddings, labels_payload, source_hashes = (
        load_cls_cache(config)
    )
    train_labels = labels_payload["train"]["labels"].long()
    test_labels = labels_payload["test"]["labels"].long()
    dataset = TensorDataset(train_embeddings, train_labels)
    train_indices, validation_indices = stratified_train_validation_split(
        train_labels,
        config["training"]["validation_fraction"],
        config["seed"],
    )

    selection_head, _, selection_history = _train_stage(
        config,
        dataset,
        train_indices,
        validation_indices,
        config["training"]["epochs"],
        device,
    )
    selected_epoch = select_evidential_epoch(selection_history)
    del selection_head
    set_reproducibility(config["seed"], config["num_threads"])
    best_head, _, best_history = _train_stage(
        config, dataset, train_indices, validation_indices, selected_epoch, device
    )
    validation_prediction = _predict(
        best_head, dataset, validation_indices, config, device
    )
    selected_top_n, top_n_results = _choose_top_n(
        train_embeddings[validation_indices],
        train_labels[validation_indices],
        validation_prediction["uncertainty"],
        config,
        device,
    )
    validation_metrics = classification_metrics(
        validation_prediction["probabilities"],
        validation_prediction["labels"],
        bins=config["calibration"]["ece_bins"],
    )
    validation_correct = validation_prediction["probabilities"].argmax(dim=1).eq(
        validation_prediction["labels"]
    )
    validation_diagnostics = {
        "mean_total_evidence": validation_prediction["evidence"].sum(dim=1).mean().item(),
        "mean_uncertainty": validation_prediction["uncertainty"].mean().item(),
        "mean_uncertainty_correct": _optional_mean(
            validation_prediction["uncertainty"][validation_correct]
        ),
        "mean_uncertainty_incorrect": _optional_mean(
            validation_prediction["uncertainty"][~validation_correct]
        ),
    }
    atomic_torch_save(validation_prediction, paths["validation"])

    all_indices = torch.arange(len(dataset))
    set_reproducibility(config["seed"], config["num_threads"])
    final_head, final_optimizer, final_history = _train_stage(
        config, dataset, all_indices, None, selected_epoch, device
    )
    test_dataset = TensorDataset(test_embeddings, test_labels)
    test_prediction = _predict(
        final_head,
        test_dataset,
        torch.arange(len(test_dataset)),
        config,
        device,
    )
    atomic_torch_save(
        {
            "uncertainty": test_prediction["uncertainty"],
            "evidence": test_prediction["evidence"],
            "paths": labels_payload["test"]["paths"],
        },
        paths["test_uncertainty"],
    )
    atomic_torch_save(
        {
            "head_state": final_head.state_dict(),
            "optimizer_state": final_optimizer.state_dict(),
            "selected_epoch": selected_epoch,
            "selected_top_n": selected_top_n,
            "architecture": {
                "input_dim": final_head.input_dim,
                "hidden_dim": final_head.hidden_dim,
                "num_classes": final_head.num_classes,
                "dropout": final_head.dropout,
                "evidence_activation": "softplus",
            },
            "source_cache_sha256": source_hashes,
            "config": public_config(config),
        },
        paths["checkpoint"],
    )
    atomic_json_save(
        {
            "selected_epoch": selected_epoch,
            "selection": selection_history,
            "selection_refit": best_history,
            "final": final_history,
        },
        paths["history"],
    )
    atomic_json_save(
        {
            "status": "trained",
            "selected_epoch": selected_epoch,
            "selected_top_n": selected_top_n,
            "validation": {
                "classification": validation_metrics,
                "top_n_search": top_n_results,
                "diagnostics": validation_diagnostics,
            },
        },
        paths["metrics"],
    )
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in (
            paths["checkpoint"], paths["history"], paths["validation"],
            paths["test_uncertainty"], paths["metrics"],
        )
    }
    atomic_json_save(
        {
            "format_version": 1,
            "method": "dinov3_frozen_cls_edl_top_n_uncertainty_reranking",
            "lineage": {
                "sources": [
                    {
                        "name": "Evidential Deep Learning",
                        "version": "arXiv:1806.01768v3",
                        "url": "https://arxiv.org/abs/1806.01768v3",
                        "code_used": False,
                        "repository_commit": None,
                        "license": "paper reference only; no source code copied",
                    },
                    {
                        "name": "Evidential Transformers for Improved Image Retrieval",
                        "version": "arXiv:2409.01082v2",
                        "url": "https://arxiv.org/abs/2409.01082v2",
                        "code_used": False,
                        "repository_commit": None,
                        "license": "paper reference only; no author code available",
                    },
                ],
                "equations_used": [
                    "alpha=evidence+1",
                    "uncertainty=num_classes/sum(alpha)",
                    "Bayes-risk MSE with Dirichlet variance",
                    "annealed KL to uniform Dirichlet",
                    "ascending-uncertainty stable reranking of cosine top-N",
                ],
                "deviations": [
                    "frozen DINOv3 CLS instead of a fine-tuned evidential transformer",
                    "standalone MLP evidential head",
                    "Softplus evidence activation",
                    "top-N selected without test access",
                ],
            },
            "config": public_config(config),
            "device": str(device),
            "source_cache_sha256": source_hashes,
            "artifact_sha256": artifact_hashes,
            "seconds": time.perf_counter() - started,
        },
        paths["manifest"],
    )
    return {
        "selected_epoch": selected_epoch,
        "selected_top_n": selected_top_n,
        "validation": validation_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train E1 Evidential Head.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e1.yaml")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_e1_training(args.config, overwrite=args.overwrite)
    print(result)


if __name__ == "__main__":
    main()
