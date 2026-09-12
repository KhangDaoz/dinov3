import argparse
from pathlib import Path

from evaluate_e1 import run_e1_evaluation
from train_evidential import run_e1_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate E1.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e1.yaml")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_e1_training(args.config, overwrite=args.overwrite)
    result = run_e1_evaluation(args.config)
    print(result)


if __name__ == "__main__":
    main()
