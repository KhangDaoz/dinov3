#!/usr/bin/env python
"""Evaluate M1--M4 on test and select the E2A winner."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def export_test_artifacts(config, config_path: Path, destination: Path) -> None:
    """Bundle required test outputs so selection-folder downloads are complete."""
    method = config.representation.method
    root = Path(config.output.root)
    required = ["metrics/test.json", "rankings/test_top100.pt",
                "embeddings/manifest.json"]
    if method in {"m3", "m4"}:
        required.append("embeddings/test.pt")
    if method == "m4":
        required.extend(["attention/test_weights.pt", "attention/test_entropy.pt"])
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing test artifacts:\n" + "\n".join(missing))
    target = destination / "artifacts" / method
    for name in required + ["config_resolved.yaml", "environment.json", "metadata.json"]:
        source = root / name
        if source.is_file():
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
    shutil.copy2(config_path, target / "config.yaml")
    print(f"{method.upper()} test artifacts exported to: {target.resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs=4, required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e2a_selection"))
    parser.add_argument("--export-only", action="store_true",
                        help="Reuse existing test outputs without evaluating again")
    args = parser.parse_args()
    from uncertainty_retrieval.config_e2a import load_e2a_config

    configs = [load_e2a_config(path) for path in args.configs]
    if {config.representation.method for config in configs} != {"m1", "m2", "m3", "m4"}:
        raise ValueError("Exactly one config for each of M1--M4 is required")
    for config_path, config in zip(args.configs, configs, strict=True):
        if not args.export_only:
            print(f"Running {config.representation.method.upper()} test; "
                  f"output: {Path(config.output.root).resolve()}", flush=True)
            subprocess.run(
                [sys.executable, "scripts/run_e2a.py", "--config", str(config_path),
                 "--stage", "test"], check=True,
            )
        export_test_artifacts(config, config_path, args.output_dir)
    subprocess.run(
        [sys.executable, "scripts/select_e2a_winner.py", "--configs",
         *(str(path) for path in args.configs),
         "--output", str(args.output_dir / "test_selection.json")], check=True,
    )
    print(f"Complete test bundle: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
