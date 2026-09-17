#!/usr/bin/env python
"""Run the reproducible multi-seed E1 pipeline."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from uncertainty_retrieval.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    os.environ.setdefault(
        "OMP_NUM_THREADS",
        str(max(1, (os.cpu_count() or 2) // 2)),
    )
    kernel_cache = Path("/tmp/uncertainty_retrieval_torch_kernels")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(kernel_cache))
    base = [sys.executable]
    config_args = ["--config", str(args.config)]
    subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc-per-node=2",
            "scripts/extract_features.py",
            *config_args,
        ],
        check=True,
    )
    for seed in config.training.seeds:
        seed_args = ["--seed", str(seed)]
        subprocess.run(
            [
                "torchrun",
                "--standalone",
                "--nproc-per-node=2",
                "scripts/train_evidential.py",
                *config_args,
                *seed_args,
                "--stage",
                "tune",
            ],
            check=True,
        )
        subprocess.run(
            [
                *base,
                "scripts/evaluate.py",
                *config_args,
                *seed_args,
                "--stage",
                "tune",
            ],
            check=True,
        )
        if not args.test:
            continue
        selection_path = (
            Path(config.output.root)
            / f"seed_{seed}"
            / "tuning"
            / "metrics"
            / "selection.json"
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_top_n = (
            args.top_n
            if args.top_n is not None
            else int(selection["r1"]["top_n"])
        )
        selected_beta = (
            args.beta
            if args.beta is not None
            else float(selection["r2"]["beta"])
        )
        subprocess.run(
            [
                "torchrun",
                "--standalone",
                "--nproc-per-node=2",
                "scripts/train_evidential.py",
                *config_args,
                *seed_args,
                "--stage",
                "final",
            ],
            check=True,
        )
        subprocess.run(
            [
                *base,
                "scripts/evaluate.py",
                *config_args,
                *seed_args,
                "--stage",
                "test",
                "--top-n",
                str(selected_top_n),
                "--beta",
                str(selected_beta),
            ],
            check=True,
        )
    if not args.test:
        return
    test_metrics = []
    for seed in config.training.seeds:
        metrics_path = (
            Path(config.output.root)
            / f"seed_{seed}"
            / "final"
            / "metrics"
            / "test.json"
        )
        test_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    summary = {
        "seeds": list(config.training.seeds),
        "baseline": {},
        "r1": {},
        "r1_topn_grid": {},
    }
    for method in ("baseline", "r1"):
        for metric in test_metrics[0][method]:
            values = [run[method][metric] for run in test_metrics]
            summary[method][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    for top_n in config.retrieval.top_n_grid:
        key = str(top_n)
        summary["r1_topn_grid"][key] = {}
        for metric in test_metrics[0]["r1_topn_grid"][key]:
            values = [run["r1_topn_grid"][key][metric] for run in test_metrics]
            summary["r1_topn_grid"][key][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    output = Path(config.output.root) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


if __name__ == "__main__":
    main()
