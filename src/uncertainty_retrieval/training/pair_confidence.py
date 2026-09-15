"""E2B preparation, additive BCE DDP training and frozen scoring."""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from uncertainty_retrieval.config_e2b import load_e2b_config, save_e2b_config
from uncertainty_retrieval.data.pair_cache import load_pair_inputs, verify_provenance, validate_static_pool
from uncertainty_retrieval.data.feature_cache import sha256_file
from uncertainty_retrieval.evaluation.retrieval import cosine_rankings
from uncertainty_retrieval.models.pair_confidence import PairConfidenceNetwork
from uncertainty_retrieval.sampling.pairs import sample_epoch_pairs
from uncertainty_retrieval.training.representation import atomic_torch_save
from uncertainty_retrieval.utils import (
    initialize_distributed, cleanup_distributed, distributed_barrier,
    environment_metadata, seed_everything, write_json,
)


def setup(config):
    if config.runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; set runtime.device=cpu for explicit fallback")
    rank, world, _, device = initialize_distributed()
    if config.runtime.device == "cpu" and device.type != "cpu":
        raise RuntimeError("CPU fallback requires CUDA_VISIBLE_DEVICES='' at launch")
    if world != config.runtime.world_size:
        raise RuntimeError(f"Expected {config.runtime.world_size} torchrun processes")
    seed_everything(42)
    return rank, world, device


def merge_shards(root, name, world, fields):
    shards = [torch.load(root / f"{name}_rank{rank}.pt", map_location="cpu",
                         weights_only=True) for rank in range(world)]
    rows = torch.cat([item["rows"] for item in shards])
    if len(rows.unique()) != len(rows) or not torch.equal(
        rows.sort().values, torch.arange(len(rows))
    ):
        raise ValueError("Missing/duplicate distributed query rows")
    order = rows.argsort()
    return {key: torch.cat([item[key] for item in shards])[order] for key in fields}


def prepare(config_path: Path):
    config = load_e2b_config(config_path)
    rank, world, device = setup(config)
    root = Path(config.output_root)
    try:
        for split in ("fit", "validation"):
            view, provenance = load_pair_inputs(config, config_path, split)
            destination = root / "pairs" / f"{split}_static.pt"
            if destination.is_file():
                payload = torch.load(destination, map_location="cpu", weights_only=True)
                verify_provenance(payload["provenance"], provenance)
                validate_static_pool(payload,view)
                continue
            features = view["features"].to(device)
            ids = view["image_ids"].to(device)
            rows = torch.arange(rank, len(ids), world)
            result = cosine_rankings(features[rows.to(device)], features,
                                     ids[rows.to(device)], ids,
                                     config.runtime.query_chunk_size, 100)
            shards = root / "pairs" / "shards"
            atomic_torch_save({"rows": rows, "candidates": result.indices,
                               "cosine": result.scores},
                              shards / f"{split}_rank{rank}.pt")
            distributed_barrier(device)
            if rank == 0:
                merged = merge_shards(shards, split, world, ("candidates", "cosine"))
                targets = view["labels"][merged["candidates"]].eq(view["labels"][:, None])
                atomic_torch_save({**merged, "image_ids": view["image_ids"],
                                   "targets": targets, "split": split,
                                   "positive_prevalence": float(targets.float().mean()),
                                   "provenance": provenance}, destination)
                atomic_torch_save(view["image_ids"], root / "split" / f"{split}_image_ids.pt")
            distributed_barrier(device)
        if rank == 0:
            save_e2b_config(config, root / "config_resolved.yaml")
            write_json(environment_metadata(sys.argv), root / "environment.json")
            write_json({**provenance,
                        "notice":"Single seed; inherited E2A test-selection bias.",
                        "lineage":{"EDL":"https://arxiv.org/abs/1806.01768v3",
                                   "alpha_control":"https://arxiv.org/html/2409.01082v2",
                                   "alpha_paper_license":"CC BY 4.0",
                                   "pair_head":"repository experimental design; no third-party code copied"}},
                       root / "metadata.json")
            from uncertainty_retrieval.data.pair_cache import read_json
            write_json(read_json(config.inputs.manifest), root / "inputs/m1_manifest.json")
            write_json(read_json(config.inputs.selection), root / "inputs/e2a_selection.json")
            print(f"Static fit/validation mining saved: {root.resolve()}", flush=True)
    finally:
        cleanup_distributed()


@torch.inference_mode()
def score_pairs(model, features, query_rows, candidates, chunk_size):
    """Always query first; no labels, swaps or averaging at inference."""
    width = candidates.shape[1]
    query = query_rows.repeat_interleave(width).to(features.device)
    candidate = candidates.flatten().to(features.device)
    chunks = []
    for start in range(0, len(query), chunk_size):
        end = start + chunk_size
        chunks.append(model(features[query[start:end]], features[candidate[start:end]]))
    return torch.cat(chunks).reshape(len(query_rows), width)


def train(config_path: Path):
    config = load_e2b_config(config_path)
    rank, world, device = setup(config)
    root = Path(config.output_root)
    try:
        fit, provenance = load_pair_inputs(config, config_path, "fit")
        validation, _ = load_pair_inputs(config, config_path, "validation")
        pools = {}
        mining_hashes = {}
        for split, view in (("fit", fit), ("validation", validation)):
            path = root / "pairs" / f"{split}_static.pt"
            pools[split] = torch.load(path, map_location="cpu", weights_only=True)
            verify_provenance(pools[split]["provenance"], provenance)
            validate_static_pool(pools[split],view)
            mining_hashes[f"{split}_mining_sha256"] = sha256_file(path)
        provenance.update(mining_hashes)
        model = PairConfidenceNetwork().to(device)
        wrapped = DistributedDataParallel(model, device_ids=[device.index]
                                          if device.type == "cuda" else None) if world > 1 else model
        recipe = config.training
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=recipe.learning_rate,
                                      weight_decay=recipe.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, recipe.epochs)
        fit_features = fit["features"].to(device)
        val_features = validation["features"].to(device)
        val_rows = torch.arange(rank, len(val_features), world)
        val_targets = pools["validation"]["targets"][val_rows].to(device).float()
        best = math.inf
        history = []
        for epoch in range(1, recipe.epochs + 1):
            pairs, targets, sampling = sample_epoch_pairs(
                fit["labels"], pools["fit"]["candidates"], epoch
            )
            # Pad only to world_size, not global batch; each tail rank has equal weight.
            padding = (-len(pairs)) % world
            if padding:
                pairs = torch.cat((pairs, pairs[:padding]))
                targets = torch.cat((targets, targets[:padding]))
            sampling["ddp_padding_pairs"] = padding
            if rank == 0:
                atomic_torch_save({"image_pairs": fit["image_ids"][pairs],
                                   "targets": targets, "sampling": sampling},
                                  root / "pairs" / f"epoch_{epoch:02d}.pt")
            model.train()
            total = torch.zeros(2, dtype=torch.float64, device=device)
            for start in range(0, len(pairs), recipe.global_batch_size):
                end = start + recipe.global_batch_size
                indices = pairs[start:end][rank::world].to(device)
                target = targets[start:end][rank::world].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = wrapped(fit_features[indices[:, 0]], fit_features[indices[:, 1]])
                loss = F.binary_cross_entropy_with_logits(logits, target)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise FloatingPointError(f"E2B nonfinite loss/gradient epoch={epoch} batch={start//recipe.global_batch_size}")
                optimizer.step()
                total[0] += loss.detach().double() * len(indices)
                total[1] += len(indices)
            model.eval()
            with torch.inference_mode():
                logits = score_pairs(model, val_features, val_rows,
                                     pools["validation"]["candidates"][val_rows],
                                     config.runtime.pair_chunk_size)
                losses = F.binary_cross_entropy_with_logits(logits, val_targets, reduction="none")
                validation_sum = torch.stack((losses.double().sum(),
                                              torch.tensor(losses.numel(), dtype=torch.float64, device=device)))
                fixed_scores = 0.5*pools["validation"]["cosine"][val_rows].to(device) + 0.5*logits.sigmoid()
                fixed_order = fixed_scores.argsort(dim=1,descending=True,stable=True)
                relevant = val_targets.gather(1,fixed_order).bool()
                validation_hits = torch.stack([relevant[:,:k].any(1).double().sum() for k in (1,2,4,8)])
            if world > 1:
                torch.distributed.all_reduce(total)
                torch.distributed.all_reduce(validation_sum)
                torch.distributed.all_reduce(validation_hits)
            validation_bce = float(validation_sum[0] / validation_sum[1])
            if not math.isfinite(validation_bce):
                raise FloatingPointError(f"E2B nonfinite validation BCE epoch={epoch}")
            scheduler.step()
            history.append({"epoch": epoch, "balanced_training_bce": float(total[0]/total[1]),
                            "natural_candidate_validation_bce": validation_bce,
                            "secondary_validation_fusion_lambda_05": {
                                f"recall_at_{k}":float(validation_hits[i]/len(val_features))
                                for i,k in enumerate((1,2,4,8))},
                            "sampling": sampling})
            if rank == 0:
                if validation_bce < best:
                    best = validation_bce
                    atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                       "scheduler": scheduler.state_dict(), "epoch": epoch,
                                       "validation_bce": best, "provenance": provenance,
                                       "environment": environment_metadata(sys.argv),
                                       "parameter_count": sum(p.numel() for p in model.parameters())},
                                      root / "checkpoints/best.pt")
                write_json(history, root / "metrics/training_history.json")
                print(f"epoch={epoch} validation_bce={validation_bce:.8f}", flush=True)
            distributed_barrier(device)
        if rank == 0:
            from uncertainty_retrieval.data.pair_cache import read_json
            environment = read_json(root / "environment.json")
            write_json({"preparation": environment, "training": environment_metadata(sys.argv)},
                       root / "environment.json")
    finally:
        cleanup_distributed()
