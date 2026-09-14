"""Training and full-output export for the E1 Evidential Head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from uncertainty_retrieval.models.evidential import (
    EvidentialHead,
    edl_mse_loss,
)
from uncertainty_retrieval.utils import distributed_barrier


class FeatureDataset(Dataset):
    def __init__(self, features: Tensor, labels: Tensor, image_ids: Tensor) -> None:
        if not (len(features) == len(labels) == len(image_ids)):
            raise ValueError("Feature dataset fields must have equal lengths")
        self.features = features.float().cpu()
        self.labels = labels.long().cpu()
        self.image_ids = image_ids.long().cpu()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.features[index], self.labels[index], self.image_ids[index]


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float]]


def train_evidential_fixed_epochs(
    model: EvidentialHead,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    annealing_epochs: int,
    learning_rate: float,
    weight_decay: float,
    amp: bool,
    checkpoint_path: str | Path,
    rank: int = 0,
    sampler: DistributedSampler | None = None,
) -> list[dict[str, float]]:
    """Retrain on all development data for a validation-selected epoch count."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    model.to(device)
    wrapped: nn.Module = model
    if torch.distributed.is_initialized():
        device_ids = [device.index] if device.type == "cuda" else None
        wrapped = DistributedDataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.AdamW(
        wrapped.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history = []
    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loss = _run_epoch(
            wrapped,
            loader,
            device,
            epoch,
            annealing_epochs,
            amp,
            optimizer,
        )
        history.append({"epoch": float(epoch), "development_loss": loss})
    if rank == 0:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "epoch": epochs},
            checkpoint,
        )
    distributed_barrier(device)
    return history


def make_feature_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool,
    distributed: bool,
    seed: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
    }
    if num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(dataset, **kwargs), sampler


def _reduce_mean(value_sum: float, count: int, device: torch.device) -> float:
    values = torch.tensor([value_sum, count], dtype=torch.float64, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(values)
    return (values[0] / values[1].clamp_min(1)).item()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    annealing_epochs: int,
    amp: bool,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    item_count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for features, labels, _ in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
                dtype=torch.float16,
            ):
                output = model(features)
            loss, _ = edl_mse_loss(
                output.alpha,
                labels,
                epoch=epoch,
                annealing_epochs=annealing_epochs,
            )
            if training:
                loss.backward()
                optimizer.step()
            batch_size = len(features)
            loss_sum += loss.detach().item() * batch_size
            item_count += batch_size
    return _reduce_mean(loss_sum, item_count, device)


def train_evidential_head(
    model: EvidentialHead,
    fit_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    annealing_epochs: int,
    learning_rate: float,
    weight_decay: float,
    amp: bool,
    checkpoint_path: str | Path,
    rank: int = 0,
    fit_sampler: DistributedSampler | None = None,
) -> TrainingResult:
    """Train a head and select the lowest validation EDL-loss checkpoint."""
    model.to(device)
    wrapped: nn.Module = model
    if torch.distributed.is_initialized():
        device_ids = [device.index] if device.type == "cuda" else None
        wrapped = DistributedDataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.AdamW(
        wrapped.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_loss = float("inf")
    best_epoch = -1
    history = []
    checkpoint = Path(checkpoint_path)
    for epoch in range(1, epochs + 1):
        if fit_sampler is not None:
            fit_sampler.set_epoch(epoch)
        fit_loss = _run_epoch(
            wrapped,
            fit_loader,
            device,
            epoch,
            annealing_epochs,
            amp,
            optimizer,
        )
        validation_loss = _run_epoch(
            wrapped,
            validation_loader,
            device,
            epoch,
            annealing_epochs,
            amp,
            optimizer=None,
        )
        history.append(
            {
                "epoch": float(epoch),
                "fit_loss": fit_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            if rank == 0:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                    },
                    checkpoint,
                )
        distributed_barrier(device)
    return TrainingResult(best_epoch, best_loss, history)


@torch.inference_mode()
def export_evidential_outputs(
    model: EvidentialHead,
    loader: DataLoader,
    device: torch.device,
    class_order: list[int],
    schema_version: int,
) -> dict[str, Any]:
    """Export full e, alpha, probabilities, and u on CPU for E2."""
    model.eval().to(device)
    chunks: dict[str, list[Tensor]] = {
        "image_ids": [],
        "labels": [],
        "evidence": [],
        "alpha": [],
        "probabilities": [],
        "uncertainty": [],
        "predictions": [],
    }
    for features, labels, image_ids in loader:
        output = model(features.to(device, non_blocking=True))
        chunks["image_ids"].append(image_ids.cpu())
        chunks["labels"].append(labels.cpu())
        chunks["evidence"].append(output.evidence.cpu())
        chunks["alpha"].append(output.alpha.cpu())
        chunks["probabilities"].append(output.probabilities.cpu())
        chunks["uncertainty"].append(output.uncertainty.cpu())
        chunks["predictions"].append(output.probabilities.argmax(dim=-1).cpu())
    payload = {key: torch.cat(value) for key, value in chunks.items()}
    order = torch.argsort(payload["image_ids"], stable=True)
    payload = {key: value[order].contiguous() for key, value in payload.items()}
    payload.update(
        {
            "class_order": class_order,
            "dtype": str(payload["alpha"].dtype),
            "schema_version": schema_version,
        }
    )
    return payload


def save_evidential_outputs(payload: dict[str, Any], path: str | Path) -> None:
    """Atomically persist full evidential tensors."""
    required = {
        "image_ids",
        "labels",
        "evidence",
        "alpha",
        "probabilities",
        "uncertainty",
        "predictions",
        "class_order",
        "dtype",
        "schema_version",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Evidential output missing keys: {sorted(missing)}")
    size = len(payload["image_ids"])
    if any(
        len(payload[key]) != size
        for key in (
            "labels",
            "evidence",
            "alpha",
            "probabilities",
            "uncertainty",
            "predictions",
        )
    ):
        raise ValueError("Evidential output fields have inconsistent lengths")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_evidential_outputs(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)
