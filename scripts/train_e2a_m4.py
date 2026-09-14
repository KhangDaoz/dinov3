#!/usr/bin/env python
"""Train E2A-M4 attention pooling on development-fit patch tokens."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from uncertainty_retrieval.config_e2a import load_e2a_config, save_e2a_config
from uncertainty_retrieval.data.cub import load_cub_records, split_development_records
from uncertainty_retrieval.data.feature_cache import sha256_file, validation_ids_hash
from uncertainty_retrieval.data.patch_token_cache import load_patch_token_cache
from uncertainty_retrieval.evaluation.representation import evaluate_top100
from uncertainty_retrieval.training.representation import (
    DistributedClassBalancedBatchSampler, M4TrainingModel, atomic_torch_save,
    epoch_hit_key, m4_checkpoint_payload,
)
from uncertainty_retrieval.utils import (
    distributed_barrier, environment_metadata, initialize_distributed,
    seed_everything, write_json,
)


class PatchSubset(Dataset):
    def __init__(self, cache, image_ids: list[int], labels: list[int]) -> None:
        self.cache = cache
        self.image_ids = image_ids
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        return self.cache.batch([self.image_ids[index]])[0], self.labels[index]


def _embed(model, cache, ids: list[int], device, batch_size: int):
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            patches = cache.batch(batch_ids).to(device).float()
            embeddings.append(model.representation(patches).embedding.cpu())
    return torch.cat(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_e2a_config(args.config)
    if config.representation.method != "m4" or config.training is None:
        raise ValueError("train_e2a_m4.py requires an M4 config")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    training = config.training
    rank, world_size, local_rank, device = initialize_distributed()
    if world_size != config.runtime.world_size:
        raise RuntimeError(f"Expected {config.runtime.world_size} ranks")
    seed_everything(training.seed, deterministic=True)
    records = load_cub_records(
        config.dataset.root, config.dataset.development_classes,
        config.dataset.total_classes,
    )
    fit, validation = split_development_records(
        records, config.dataset.validation_fraction, config.dataset.split_seed
    )
    fit_ids = [record.image_id for record in fit]
    val_ids = [record.image_id for record in validation]
    reference_ids = torch.load(
        config.evaluation.reference_validation_ids, map_location="cpu", weights_only=True
    )
    if not torch.equal(reference_ids, torch.tensor(val_ids)):
        raise ValueError("M4 validation IDs differ from accepted M1--M3")
    cache = load_patch_token_cache(config, records)
    dataset = PatchSubset(cache, fit_ids, [record.original_label for record in fit])
    sampler = DistributedClassBalancedBatchSampler(
        dataset.labels, training.classes_per_batch, training.images_per_class,
        rank, world_size, training.seed,
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config.runtime.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.runtime.num_workers > 0,
        prefetch_factor=(config.runtime.prefetch_factor if config.runtime.num_workers else None),
    )
    model = M4TrainingModel(
        config.model.embedding_dim, config.representation.attention_hidden_dim,
        config.model.expected_patch_tokens, config.dataset.development_classes,
        training.proxy_alpha, training.proxy_margin,
    ).to(device)
    ddp = DistributedDataParallel(
        model, device_ids=[local_rank] if device.type == "cuda" else None
    )
    weight_params = [model.representation.hidden.weight, model.representation.score.weight]
    bias_params = [model.representation.hidden.bias, model.representation.score.bias]
    optimizer = torch.optim.AdamW(
        [
            {"params": weight_params, "weight_decay": training.weight_decay},
            {"params": bias_params, "weight_decay": 0.0},
            {"params": [model.objective.proxies], "weight_decay": 0.0},
        ],
        lr=training.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training.epochs * len(loader)
    )
    root = Path(config.output.root)
    provenance = {
        "config_sha256": sha256_file(args.config),
        "patch_manifest_sha256": sha256_file(config.cache.patch_manifest),
        "split_hash": validation_ids_hash(val_ids),
        "seed": training.seed,
        "total_trainable_parameters": sum(p.numel() for p in model.parameters()),
        "deployable_parameters": sum(p.numel() for p in model.representation.parameters()),
    }
    history = []
    best_key = None
    for epoch in range(1, training.epochs + 1):
        sampler.set_epoch(epoch - 1)
        ddp.train()
        loss_sum = 0.0
        for patches, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = ddp(
                patches.to(device, non_blocking=True).float(),
                labels.to(device, non_blocking=True),
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("M4 gradients are not finite")
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach())
        distributed_barrier(device)
        if rank == 0:
            model.eval()
            embeddings = _embed(model, cache, val_ids, device, config.runtime.batch_size)
            labels = torch.tensor([record.original_label for record in validation])
            metrics, _ = evaluate_top100(
                embeddings, torch.tensor(val_ids), labels, device,
                config.runtime.similarity_chunk_size, config.evaluation.ranking_depth,
            )
            hits = {k: round(metrics[f"recall_at_{k}"] * len(val_ids)) for k in (1,2,4,8)}
            row = {"epoch": epoch, "loss": loss_sum / len(loader), "hits": hits,
                   "metrics": metrics, "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            key = epoch_hit_key(hits, epoch)
            if best_key is None or key > best_key:
                best_key = key
                atomic_torch_save(
                    m4_checkpoint_payload(
                        model, optimizer, scheduler, epoch, metrics, hits, provenance
                    ), root / "checkpoints" / "best.pt"
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
        torch.save(torch.tensor(val_ids), split_dir / "validation_image_ids.pt")
        (split_dir / "manifest_hash.txt").write_text(
            validation_ids_hash(val_ids) + "\n", encoding="utf-8"
        )
        write_json(
            json.loads(Path(config.cache.patch_manifest).read_text(encoding="utf-8")),
            root / "inputs" / "patch_token_manifest.json",
        )
    distributed_barrier(device)


if __name__ == "__main__":
    main()
