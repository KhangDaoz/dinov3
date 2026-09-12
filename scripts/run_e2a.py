import argparse
import importlib.util
from pathlib import Path

from evaluate import run_evaluation
from extract_features import run_extraction

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import load_config


def load_learned_runner(script_name, function_name):
    script_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(
        f"e2a_{script_path.stem}_script", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Không load được training script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def main():
    parser = argparse.ArgumentParser(description="Run E2A extraction and evaluation.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e2a.yaml")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["representation"] == "fusion":
        load_learned_runner(
            "train_projection.py", "run_fusion_training"
        )(args.config, overwrite=args.overwrite)
    elif config["representation"] == "attention_pool":
        load_learned_runner(
            "train_attention_pool.py", "run_attention_training"
        )(args.config, overwrite=args.overwrite)
    else:
        run_extraction(args.config, overwrite=args.overwrite)
    recalls = run_evaluation(args.config)
    for name, value in recalls.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
