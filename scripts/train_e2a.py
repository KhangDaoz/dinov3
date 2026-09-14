#!/usr/bin/env python
"""Train learned E2A representation pipelines on development-fit only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from uncertainty_retrieval.config_e2a import load_e2a_config, save_e2a_config
from uncertainty_retrieval.data.cub import load_cub_records, split_development_records
from uncertainty_retrieval.data.feature_cache import sha256_file, validation_ids_hash
from uncertainty_retrieval.data.fused_cache import load_fused_feature_cache
from uncertainty_retrieval.evaluation.representation import evaluate_top100
from uncertainty_retrieval.training.representation import (
    DistributedClassBalancedBatchSampler,
    M3TrainingModel,
    atomic_torch_save,
    checkpoint_payload,
    epoch_hit_key,
)
from uncertainty_retrieval.utils import (
    distributed_barrier,
    environment_metadata,
    initialize_distributed,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def _indices_for_ids(all_ids: torch.Tensor, selected: list[int]) -> torch.Tensor:
    lookup = {int(value): index for index, value in enumerate(all_ids)}
    return torch.tensor([lookup[value] for value in selected], dtype=torch.long)


def main() -> None:
    args = parse_args()
    config = load_e2a_config(args.config)
    if config.representation.method != "m3" or config.training is None:
        raise ValueError("train_e2a.py currently requires an M3 config")
    training = config.training
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank, world_size, local_rank, device = initialize_distributed()
    if world_size != config.runtime.world_size:
        raise RuntimeError(f"Expected {config.runtime.world_size} ranks")
    seed_everything(training.seed, deterministic=True)
    fused = load_fused_feature_cache(config)
    records = load_cub_records(
        config.dataset.root,
        config.dataset.development_classes,
        config.dataset.total_classes,
    )
    fit_records, validation_records = split_development_records(
        records, config.dataset.validation_fraction, config.dataset.split_seed
    )
    fit_ids = [record.image_id for record in fit_records]
    validation_ids = [record.image_id for record in validation_records]
    reference_ids = torch.load(
        config.evaluation.reference_validation_ids,
        map_location="cpu",
        weights_only=True,
    )
    if not torch.equal(reference_ids, torch.tensor(validation_ids)):
        raise ValueError("M3 validation IDs differ from accepted M1/M2")
    fit_indices = _indices_for_ids(fused.image_ids, fit_ids)
    validation_indices = _indices_for_ids(fused.image_ids, validation_ids)
    if any(fused.splits[index] != "development" for index in fit_indices.tolist()):
        raise ValueError("Non-development row entered M3 fit data")
    dataset = TensorDataset(
        fused.cls[fit_indices],
        fused.mean_patch[fit_indices],
        fused.labels[fit_indices],
    )
    sampler = DistributedClassBalancedBatchSampler(
        fused.labels[fit_indices],
        training.classes_per_batch,
        training.images_per_class,
        rank,
        world_size,
        training.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.runtime.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.runtime.num_workers > 0,
        prefetch_factor=(
            config.runtime.prefetch_factor
            if config.runtime.num_workers > 0
            else None
        ),
    )
    model = M3TrainingModel(
        config.model.embedding_dim,
        config.dataset.development_classes,
        training.proxy_alpha,
        training.proxy_margin,
    ).to(device)
    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [model.representation.projection.weight],
                "weight_decay": training.weight_decay,
            },
            {
                "params": [model.representation.projection.bias],
                "weight_decay": 0.0,
            },
            {"params": [model.objective.proxies], "weight_decay": 0.0},
        ],
        lr=training.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training.epochs * len(loader)
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=config.runtime.amp and device.type == "cuda",
    )
    root = Path(config.output.root)
    provenance = {
        "config_sha256": sha256_file(args.config),
        "cls_cache_sha256": fused.cls_sha256,
        "mean_patch_cache_sha256": fused.mean_patch_sha256,
        "split_hash": validation_ids_hash(validation_ids),
        "seed": training.seed,
        "concatenation_order": ["cls", "mean_patch"],
        "total_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "deployable_parameters": sum(
            parameter.numel() for parameter in model.representation.parameters()
        ),
    }
    history = []
    best_key: tuple[int, ...] | None = None
    for epoch in range(1, training.epochs + 1):
        sampler.set_epoch(epoch - 1)
        ddp.train()
        loss_sum = 0.0
        for cls, patch, labels in loader:
            cls = cls.to(device, non_blocking=True)
            patch = patch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=config.runtime.amp and device.type == "cuda",
                dtype=torch.float16,
            ):
                loss = ddp(cls, patch, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("M3 gradients are not finite")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach())
        distributed_barrier(device)
        if rank == 0:
            model.eval()
            with torch.inference_mode():
                validation_embeddings = model.representation(
                    fused.cls[validation_indices].to(device),
                    fused.mean_patch[validation_indices].to(device),
                ).cpu()
            validation_labels = fused.labels[validation_indices]
            validation_id_tensor = fused.image_ids[validation_indices]
            metrics, _ = evaluate_top100(
                validation_embeddings,
                validation_id_tensor,
                validation_labels,
                device,
                config.runtime.similarity_chunk_size,
                config.evaluation.ranking_depth,
            )
            hits = {
                k: round(metrics[f"recall_at_{k}"] * len(validation_ids))
                for k in (1, 2, 4, 8)
            }
            row = {
                "epoch": epoch,
                "loss": loss_sum / len(loader),
                "hits": hits,
                "metrics": metrics,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            history.append(row)
            key = epoch_hit_key(hits, epoch)
            if best_key is None or key > best_key:
                best_key = key
                atomic_torch_save(
                    checkpoint_payload(
                        model, optimizer, scheduler, epoch, metrics, hits, provenance
                    ),
                    root / "checkpoints" / "best.pt",
                )
        distributed_barrier(device)
    if rank == 0:
        save_e2a_config(config, root / "config_resolved.yaml")
        write_json(history, root / "training" / "history.json")
        write_json(environment_metadata(sys.argv), root / "environment.json")
        write_json(provenance, root / "metadata.json")
        split_dir = root / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor(fit_ids), split_dir / "fit_image_ids.pt")
        torch.save(torch.tensor(validation_ids), split_dir / "validation_image_ids.pt")
        (split_dir / "manifest_hash.txt").write_text(
            validation_ids_hash(validation_ids) + "\n", encoding="utf-8"
        )
        for source, destination in (
            (config.cache.cls_manifest, root / "inputs" / "cls_manifest.json"),
            (
                config.cache.mean_patch_manifest,
                root / "inputs" / "mean_patch_manifest.json",
            ),
        ):
            write_json(json.loads(Path(source).read_text(encoding="utf-8")), destination)
    distributed_barrier(device)


if __name__ == "__main__":
    main()
