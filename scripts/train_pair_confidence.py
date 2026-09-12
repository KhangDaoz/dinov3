import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidential import binary_ranking_metrics
from pair_confidence import (
    PairConfidenceNetwork,
    pair_bce_loss,
    select_pair_epoch,
    select_retrieval_parameters,
)
from pair_sampling import PairIndexDataset, build_hard_banks, generate_pair_indices
from retrieval import (
    cosine_top_candidates,
    fuse_candidate_rankings,
    recall_from_indices,
)
from train_evidential import load_cls_cache
from train_projection import stratified_train_validation_split
from utils import (
    atomic_json_save,
    atomic_torch_save,
    load_config,
    public_config,
    resolve_device,
    set_reproducibility,
    sha256_file,
)


def e2b_paths(output_dir):
    root = Path(output_dir)
    return {
        "root": root,
        "checkpoint": root / "checkpoint.pt",
        "history": root / "training_history.json",
        "pair_validation": root / "pair_validation.pt",
        "bank_manifest": root / "hard_negative_manifest.json",
        "search": root / "hyperparameter_search.json",
        "rankings": root / "test_rankings.pt",
        "failures": root / "failure_cases.json",
        "metrics": root / "metrics.json",
        "manifest": root / "manifest.json",
    }


def make_model(config, device):
    network = config["pair_network"]
    return PairConfidenceNetwork(
        embedding_dim=768,
        hidden_dims=network["hidden_dims"],
        dropout=network["dropout"],
        feature_mode=network["feature_mode"],
    ).to(device)


def loader_options(config, device, multi_epoch=False):
    workers = config["num_workers"]
    options = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers:
        options["prefetch_factor"] = 2
        options["persistent_workers"] = multi_epoch
    return options


def pair_metrics(logits, targets, bins=15):
    probabilities = torch.sigmoid(logits.float())
    targets = targets.float()
    ranking = binary_ranking_metrics(probabilities, targets.bool())
    confidence = torch.where(probabilities >= 0.5, probabilities, 1 - probabilities)
    correct = (probabilities >= 0.5).eq(targets.bool())
    ece = torch.tensor(0.0)
    boundaries = torch.linspace(0, 1, bins + 1)
    for index in range(bins):
        mask = (confidence > boundaries[index]) & (
            confidence <= boundaries[index + 1]
        )
        if mask.any():
            ece += mask.float().mean() * (
                confidence[mask].mean() - correct[mask].float().mean()
            ).abs()
    return {
        "bce": pair_bce_loss(logits, targets).item(),
        "accuracy": correct.float().mean().item(),
        "ece": ece.item(),
        **ranking,
    }


def make_pair_data(embeddings, labels, indices, config, device, epoch):
    hard_positive, negative_bank = build_hard_banks(
        embeddings,
        labels,
        indices,
        config["pair_sampling"]["negative_bank_size"],
        device=device,
        chunk_size=config["retrieval_chunk_size"],
    )
    pairs = generate_pair_indices(
        labels,
        indices,
        hard_positive,
        negative_bank,
        config["seed"],
        epoch,
    )
    return PairIndexDataset(embeddings, *pairs, bidirectional=True), pairs, negative_bank


def predict_pairs(model, dataset, config, device):
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["anchors_per_batch"] * 8,
        shuffle=False,
        **loader_options(config, device),
    )
    logits, targets = [], []
    model.eval()
    with torch.inference_mode():
        for first, second, target in loader:
            first = first.to(device, non_blocking=device.type == "cuda")
            second = second.to(device, non_blocking=device.type == "cuda")
            logits.append(model(first, second).cpu())
            targets.append(target.cpu())
    return torch.cat(logits), torch.cat(targets)


def train_stage(config, embeddings, labels, indices, validation_dataset, epochs, device):
    model = make_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    history = []
    hard_positive, negative_bank = build_hard_banks(
        embeddings,
        labels,
        indices,
        config["pair_sampling"]["negative_bank_size"],
        device=device,
        chunk_size=config["retrieval_chunk_size"],
    )
    for epoch in range(1, epochs + 1):
        pairs = generate_pair_indices(
            labels,
            indices,
            hard_positive,
            negative_bank,
            config["seed"],
            epoch,
        )
        dataset = PairIndexDataset(embeddings, *pairs, bidirectional=True)
        loader = DataLoader(
            dataset,
            batch_size=config["training"]["anchors_per_batch"] * 8,
            shuffle=True,
            generator=torch.Generator().manual_seed(config["seed"] + epoch),
            **loader_options(config, device),
        )
        model.train()
        loss_sum = 0.0
        for first, second, targets in loader:
            first = first.to(device, non_blocking=device.type == "cuda")
            second = second.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = pair_bce_loss(model(first, second), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"]["gradient_clip"]
            )
            optimizer.step()
            loss_sum += loss.item() * len(targets)
        row = {"epoch": epoch, "train_bce": loss_sum / len(dataset)}
        if validation_dataset is not None:
            logits, targets = predict_pairs(
                model, validation_dataset, config, device
            )
            metrics = pair_metrics(
                logits, targets, config["pair_metrics"]["ece_bins"]
            )
            row.update(
                {
                    "validation_bce": metrics["bce"],
                    "validation_auroc": metrics["auroc"],
                    "validation_auprc": metrics["auprc"],
                    "validation_accuracy": metrics["accuracy"],
                    "validation_ece": metrics["ece"],
                }
            )
        history.append(row)
    return model, optimizer, history, negative_bank


def score_candidates(model, embeddings, indices, config, device):
    rows, columns = indices.shape
    flat_query = torch.arange(rows).repeat_interleave(columns)
    flat_gallery = indices.reshape(-1)
    output = []
    model.eval()
    chunk = config["retrieval"]["pair_chunk_size"]
    with torch.inference_mode():
        for start in range(0, len(flat_query), chunk):
            stop = min(start + chunk, len(flat_gallery))
            first = embeddings[flat_query[start:stop]].to(
                device, non_blocking=device.type == "cuda"
            )
            second = embeddings[flat_gallery[start:stop]].to(
                device, non_blocking=device.type == "cuda"
            )
            output.append(model.symmetric_confidence(first, second).cpu())
    return torch.cat(output).reshape(rows, columns)


def search_retrieval(model, embeddings, labels, config, device):
    max_candidates = max(config["retrieval"]["candidate_top_n_grid"])
    indices, cosine = cosine_top_candidates(
        embeddings,
        max_candidates,
        config["retrieval_chunk_size"],
        device,
    )
    confidence = score_candidates(model, embeddings, indices, config, device)
    results = []
    for candidate_top_n in config["retrieval"]["candidate_top_n_grid"]:
        for weight in config["retrieval"]["lambda_grid"]:
            ranking, _ = fuse_candidate_rankings(
                indices, cosine, confidence, weight, candidate_top_n
            )
            results.append(
                {
                    "candidate_top_n": candidate_top_n,
                    "lambda": weight,
                    "metrics": recall_from_indices(
                        ranking, labels, config["recall_k"]
                    ),
                }
            )
    return select_retrieval_parameters(results), results


def run_e2b_training(config_path, overwrite=False):
    config = load_config(config_path)
    if config.get("experiment") != "e2b":
        raise ValueError("train_pair_confidence yêu cầu experiment='e2b'")
    paths = e2b_paths(config["output_dir"])
    protected = [
        paths["checkpoint"], paths["history"], paths["pair_validation"],
        paths["search"], paths["metrics"], paths["manifest"],
    ]
    if any(path.exists() for path in protected) and not overwrite:
        raise FileExistsError("E2B output đã tồn tại; dùng --overwrite")
    set_reproducibility(config["seed"], config["num_threads"])
    device = resolve_device(config["device"])
    started = time.perf_counter()
    embeddings, _, labels_payload, source_hashes = load_cls_cache(config)
    labels = labels_payload["train"]["labels"].long()
    train_indices, validation_indices = stratified_train_validation_split(
        labels, config["training"]["validation_fraction"], config["seed"]
    )
    validation_dataset, validation_pairs, validation_bank = make_pair_data(
        embeddings, labels, validation_indices, config, device, epoch=0
    )
    selection = train_stage(
        config,
        embeddings,
        labels,
        train_indices,
        validation_dataset,
        config["training"]["epochs"],
        device,
    )
    selected_epoch = select_pair_epoch(selection[2])
    set_reproducibility(config["seed"], config["num_threads"])
    best = train_stage(
        config,
        embeddings,
        labels,
        train_indices,
        validation_dataset,
        selected_epoch,
        device,
    )
    validation_logits, validation_targets = predict_pairs(
        best[0], validation_dataset, config, device
    )
    parameters, search = search_retrieval(
        best[0],
        embeddings[validation_indices],
        labels[validation_indices],
        config,
        device,
    )
    atomic_torch_save(
        {
            "anchors": validation_pairs[0],
            "partners": validation_pairs[1],
            "targets": validation_pairs[2],
            "bidirectional_logits": validation_logits,
            "bidirectional_targets": validation_targets,
        },
        paths["pair_validation"],
    )
    atomic_json_save(search, paths["search"])

    set_reproducibility(config["seed"], config["num_threads"])
    final = train_stage(
        config,
        embeddings,
        labels,
        torch.arange(len(labels)),
        None,
        selected_epoch,
        device,
    )
    atomic_torch_save(
        {
            "model_state": final[0].state_dict(),
            "optimizer_state": final[1].state_dict(),
            "selected_epoch": selected_epoch,
            **parameters,
            "architecture": config["pair_network"],
            "source_cache_sha256": source_hashes,
            "config": public_config(config),
        },
        paths["checkpoint"],
    )
    atomic_json_save(
        {
            "selected_epoch": selected_epoch,
            "selection": selection[2],
            "selection_refit": best[2],
            "final": final[2],
        },
        paths["history"],
    )
    atomic_json_save(
        {
            "negative_bank_size": config["pair_sampling"]["negative_bank_size"],
            "train_anchors": len(labels),
            "validation_anchors": len(validation_indices),
            "validation_bank_entries": len(validation_bank),
            "device": str(device),
        },
        paths["bank_manifest"],
    )
    atomic_json_save(
        {
            "status": "trained",
            "selected_epoch": selected_epoch,
            "selected_parameters": parameters,
            "pair_validation": pair_metrics(
                validation_logits,
                validation_targets,
                config["pair_metrics"]["ece_bins"],
            ),
        },
        paths["metrics"],
    )
    artifact_paths = (
        paths["checkpoint"], paths["history"], paths["pair_validation"],
        paths["bank_manifest"], paths["search"], paths["metrics"],
    )
    atomic_json_save(
        {
            "format_version": 1,
            "method": "frozen_dinov3_cls_symmetric_pair_confidence",
            "lineage": {
                "sources": [
                    {
                        "name": "Introspective Deep Metric Learning",
                        "version": "arXiv:2309.09982v1",
                        "url": "https://arxiv.org/abs/2309.09982v1",
                        "official_code": "https://github.com/wzzheng/IDML",
                        "code_used": False,
                        "repository_commit": None,
                        "license": "paper/code referenced only; no code copied",
                    },
                    {
                        "name": "Proxy Anchor Loss for Deep Metric Learning",
                        "version": "arXiv:2003.13911v1",
                        "url": "https://arxiv.org/abs/2003.13911v1",
                        "official_code": (
                            "https://github.com/sung-yeon-kim/"
                            "Proxy-Anchor-CVPR2020"
                        ),
                        "code_used": False,
                        "repository_commit": None,
                        "license": "paper/code referenced only; no code copied",
                    },
                ],
                "equations_used": [
                    "phi(a,b)=[z_a;z_b;abs(z_a-z_b)]",
                    "symmetric confidence as mean of both pair orders",
                    "binary cross entropy with logits",
                    "lambda-weighted normalized cosine and confidence",
                ],
                "deviations": [
                    "binary pair classifier rather than IDML uncertainty embedding",
                    "frozen DINOv3 CLS",
                    "candidate-pool inference selected on validation",
                ],
            },
            "config": public_config(config),
            "device": str(device),
            "source_cache_sha256": source_hashes,
            "artifact_sha256": {
                path.name: sha256_file(path) for path in artifact_paths
            },
            "seconds": time.perf_counter() - started,
        },
        paths["manifest"],
    )
    return {"selected_epoch": selected_epoch, **parameters}


def main():
    parser = argparse.ArgumentParser(description="Train E2B Pair Confidence.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e2b.yaml")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(run_e2b_training(args.config, args.overwrite))


if __name__ == "__main__":
    main()
