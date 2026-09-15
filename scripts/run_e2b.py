#!/usr/bin/env python
"""Run E2B validation/training or frozen benchmark scoring."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
from uncertainty_retrieval.config_e2b import load_e2b_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_e2b_config(config_path)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    launch = [sys.executable, "-m", "torch.distributed.run", "--standalone",
              f"--nproc-per-node={config.runtime.world_size}"]
    scripts = ["prepare_e2b_pairs.py", "train_e2b.py"] if args.stage == "validation" else []
    if args.stage == "validation" and config.controls.enabled and config.controls.source == "controlled_retrain":
        scripts.insert(1, "prepare_e2b_controls.py")
    scripts.append("evaluate_e2b.py")
    for script in scripts:
        command = [*launch, str(REPOSITORY / "scripts" / script),
                   "--config", str(config_path)]
        if script == "evaluate_e2b.py":
            command.extend(("--stage", args.stage))
        subprocess.run(command, cwd=REPOSITORY, check=True)


if __name__ == "__main__":
    main()
