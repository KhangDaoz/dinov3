#!/usr/bin/env python
"""Prepare and evaluate an E2A representation pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from uncertainty_retrieval.config_e2a import load_e2a_config
from uncertainty_retrieval.data.cub import load_cub_records
from uncertainty_retrieval.data.feature_cache import find_reusable_m1_cache
from uncertainty_retrieval.evaluation.representation import verify_selection_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    config = load_e2a_config(args.config)
    if args.stage == "test":
        lock = verify_selection_lock(config.output.selection_lock)
        if lock["winner"] != config.representation.method:
            raise PermissionError("M1 is not the locked E2A winner")

    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))
    kernel_cache = Path("/tmp/uncertainty_retrieval_torch_kernels")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(kernel_cache))
    records = load_cub_records(
        config.dataset.root,
        config.dataset.development_classes,
        config.dataset.total_classes,
    )
    cache_path, diagnostics = find_reusable_m1_cache(config, records)
    if cache_path is None:
        print(f"No reusable M1 cache; extracting ({diagnostics})")
        subprocess.run(
            [
                "torchrun", "--standalone",
                f"--nproc-per-node={config.runtime.world_size}",
                "scripts/extract_e2a_features.py", "--config", str(args.config),
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, "scripts/evaluate_e2a.py", "--config", str(args.config),
         "--stage", args.stage],
        check=True,
    )


if __name__ == "__main__":
    main()
