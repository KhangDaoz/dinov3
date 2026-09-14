"""DDP training utilities for the learned E2A-M3 representation."""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import BatchSampler

from uncertainty_retrieval.models.metric_learning import ProxyAnchorLoss
from uncertainty_retrieval.models.representations import (
    AttentionPatchPooling,
    CLSMeanPatchProjection,
)


class DistributedClassBalancedBatchSampler(BatchSampler):
    """Generate deterministic P-by-K global batches and shard each by rank."""

    def __init__(
        self,
        labels: Tensor,
        classes_per_batch: int,
        images_per_class: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError("Sampler labels must be a non-empty vector")
        global_size = classes_per_batch * images_per_class
        if global_size % world_size:
            raise ValueError("Global balanced batch must divide across ranks")
        self.by_class: dict[int, list[int]] = {}
        for index, label in enumerate(labels.tolist()):
            self.by_class.setdefault(int(label), []).append(index)
        if classes_per_batch > len(self.by_class):
            raise ValueError("classes_per_batch exceeds available classes")
        if any(len(rows) < images_per_class for rows in self.by_class.values()):
            raise ValueError("A class has fewer rows than images_per_class")
        self.classes_per_batch = classes_per_batch
        self.images_per_class = images_per_class
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.global_size = global_size
        self.local_size = global_size // world_size
        self.batches = max(1, len(labels) // global_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        classes = sorted(self.by_class)
        for _ in range(self.batches):
            selected = generator.sample(classes, self.classes_per_batch)
            global_batch = []
            for label in selected:
                global_batch.extend(
                    generator.sample(
                        self.by_class[label], self.images_per_class
                    )
                )
            generator.shuffle(global_batch)
            start = self.rank * self.local_size
            yield global_batch[start : start + self.local_size]


def differentiable_global_gather(values: Tensor) -> Tensor:
    """Gather tensors in rank order while preserving autograd semantics."""
    if not torch.distributed.is_initialized():
        return values
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(values), dim=0)


def global_gather_labels(labels: Tensor) -> Tensor:
    if not torch.distributed.is_initialized():
        return labels
    gathered = [torch.empty_like(labels) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, labels)
    return torch.cat(gathered, dim=0)


class M3TrainingModel(nn.Module):
    """Projection and training-only proxies in one DDP-safe forward graph."""

    def __init__(
        self,
        embedding_dim: int,
        classes: int,
        alpha: float,
        margin: float,
    ) -> None:
        super().__init__()
        self.representation = CLSMeanPatchProjection(embedding_dim)
        self.objective = ProxyAnchorLoss(classes, embedding_dim, alpha, margin)

    def forward(self, cls: Tensor, mean_patch: Tensor, labels: Tensor) -> Tensor:
        local_embeddings = self.representation(cls, mean_patch)
        embeddings = differentiable_global_gather(local_embeddings)
        global_labels = global_gather_labels(labels)
        return self.objective(embeddings, global_labels)


class M4TrainingModel(nn.Module):
    """Attention pooling and training-only proxies in a DDP-safe graph."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        patch_tokens: int,
        classes: int,
        alpha: float,
        margin: float,
    ) -> None:
        super().__init__()
        self.representation = AttentionPatchPooling(
            embedding_dim, hidden_dim, patch_tokens
        )
        self.objective = ProxyAnchorLoss(classes, embedding_dim, alpha, margin)

    def forward(self, patches: Tensor, labels: Tensor) -> Tensor:
        local_embeddings = self.representation(patches).embedding
        embeddings = differentiable_global_gather(local_embeddings)
        global_labels = global_gather_labels(labels)
        return self.objective(embeddings, global_labels)


def epoch_hit_key(hit_counts: dict[int, int], epoch: int) -> tuple[int, ...]:
    """Hits@1/2/4/8 followed by preference for the earliest epoch."""
    return tuple(hit_counts[k] for k in (1, 2, 4, 8)) + (-epoch,)


def checkpoint_payload(
    model: M3TrainingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    hit_counts: dict[int, int],
    provenance: dict,
) -> dict:
    def cpu_state(state: dict) -> dict:
        return {
            key: value.detach().cpu() if isinstance(value, Tensor) else value
            for key, value in state.items()
        }

    return {
        "schema_version": 1,
        "method": "m3",
        "epoch": epoch,
        "projection": cpu_state(model.representation.state_dict()),
        "proxies": cpu_state(model.objective.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "validation_metrics": metrics,
        "validation_hit_counts": hit_counts,
        "provenance": provenance,
    }


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def m4_checkpoint_payload(
    model: M4TrainingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    hit_counts: dict[int, int],
    provenance: dict,
) -> dict:
    state = {
        key: value.detach().cpu()
        for key, value in model.representation.state_dict().items()
    }
    proxies = {
        key: value.detach().cpu()
        for key, value in model.objective.state_dict().items()
    }
    return {
        "schema_version": 1,
        "method": "m4",
        "epoch": epoch,
        "attention": state,
        "proxies": proxies,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "validation_metrics": metrics,
        "validation_hit_counts": hit_counts,
        "provenance": provenance,
    }
