#!/usr/bin/env python
"""Train the E1 Evidential Head on cached frozen CLS features."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from uncertainty_retrieval.config import load_config, save_resolved_config
from uncertainty_retrieval.data.cub import (
    load_cub_records,
    split_development_records,
    validate_protocol_counts,
)
from uncertainty_retrieval.data.patch_cache import load_feature_cache
from uncertainty_retrieval.models.evidential import EvidentialHead
from uncertainty_retrieval.training.evidential import (
    FeatureDataset,
    export_evidential_outputs,
    make_feature_loader,
    save_evidential_outputs,
    train_evidential_head,
    train_evidential_fixed_epochs,
)
from uncertainty_retrieval.utils import (
    environment_metadata,
    distributed_barrier,
    initialize_distributed,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage", choices=("tune", "final"), default="tune")
    return parser.parse_args()


def _select(cache: dict, ids: set[int]) -> FeatureDataset:
    indices = [
        index
        for index, image_id in enumerate(cache["image_ids"])
        if image_id in ids
    ]
    return FeatureDataset(
        cache["features"][indices],
        torch.tensor(cache["labels"])[indices],
        torch.tensor(cache["image_ids"])[indices],
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config.training.seed if args.seed is None else args.seed
    rank, world_size, _, device = initialize_distributed()
    seed_everything(seed, config.training.deterministic)
    cache = load_feature_cache(config.output.feature_cache)
    records = load_cub_records(
        config.dataset.root,
        config.dataset.development_classes,
        config.dataset.total_classes,
    )
    validate_protocol_counts(
        records,
        config.dataset.expected_development_images,
        config.dataset.expected_test_images,
    )
    fit_records, validation_records = split_development_records(
        records,
        config.dataset.validation_fraction,
        config.training.seed,
    )
    development_dataset = _select(
        cache,
        {
            record.image_id
            for record in records
            if record.split == "development"
        },
    )
    fit_dataset = _select(cache, {record.image_id for record in fit_records})
    validation_dataset = _select(
        cache,
        {record.image_id for record in validation_records},
    )
    test_dataset = _select(
        cache,
        {record.image_id for record in records if record.split == "test"},
    )
    training_dataset = (
        fit_dataset if args.stage == "tune" else development_dataset
    )
    fit_loader, sampler = make_feature_loader(
        training_dataset,
        config.training.batch_size,
        config.training.num_workers,
        config.training.prefetch_factor,
        shuffle=True,
        distributed=world_size > 1,
        seed=seed,
    )
    validation_loader, _ = make_feature_loader(
        validation_dataset,
        config.training.batch_size,
        config.training.num_workers,
        config.training.prefetch_factor,
        shuffle=False,
        distributed=world_size > 1,
        seed=seed,
    )
    seed_root = Path(config.output.root) / f"seed_{seed}"
    stage_directory = "tuning" if args.stage == "tune" else "final"
    output_root = seed_root / stage_directory
    checkpoint = output_root / "checkpoints" / "best.pt"
    model = EvidentialHead(
        config.model.embedding_dim,
        config.dataset.development_classes,
        config.model.head,
        config.model.hidden_dim,
    )
    if args.stage == "tune":
        result = train_evidential_head(
            model,
            fit_loader,
            validation_loader,
            device,
            config.training.epochs,
            config.training.annealing_epochs,
            config.training.learning_rate,
            config.training.weight_decay,
            config.training.amp,
            checkpoint,
            rank,
            sampler,
        )
        selected_epochs = result.best_epoch
        training_metrics = {
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "history": result.history,
        }
    else:
        selection_path = seed_root / "tuning" / "metrics" / "training.json"
        import json

        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_epochs = int(selection["best_epoch"])
        history = train_evidential_fixed_epochs(
            model,
            fit_loader,
            device,
            selected_epochs,
            config.training.annealing_epochs,
            config.training.learning_rate,
            config.training.weight_decay,
            config.training.amp,
            checkpoint,
            rank,
            sampler,
        )
        training_metrics = {
            "selected_epochs": selected_epochs,
            "selection_source": str(selection_path),
            "history": history,
        }
    if rank == 0:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        export_dataset = (
            validation_dataset
            if args.stage == "tune"
            else development_dataset
        )
        export_name = (
            "validation.pt" if args.stage == "tune" else "development.pt"
        )
        export_loader, _ = make_feature_loader(
            export_dataset,
            config.training.batch_size,
            config.training.num_workers,
            config.training.prefetch_factor,
            shuffle=False,
            distributed=False,
            seed=seed,
        )
        test_loader, _ = make_feature_loader(
            test_dataset,
            config.training.batch_size,
            config.training.num_workers,
            config.training.prefetch_factor,
            shuffle=False,
            distributed=False,
            seed=seed,
        )
        class_order = list(range(config.dataset.development_classes))
        save_evidential_outputs(
            export_evidential_outputs(
                model,
                export_loader,
                device,
                class_order,
                config.output.schema_version,
            ),
            output_root / "uncertainty" / export_name,
        )
        save_evidential_outputs(
            export_evidential_outputs(
                model,
                test_loader,
                device,
                class_order,
                config.output.schema_version,
            ),
            output_root / "uncertainty" / "test.pt",
        )
        save_resolved_config(config, output_root / "config_resolved.yaml")
        write_json(
            training_metrics,
            output_root / "metrics" / "training.json",
        )
        write_json(
            environment_metadata(),
            output_root / "environment.json",
        )
    distributed_barrier(device)


if __name__ == "__main__":
    main()
