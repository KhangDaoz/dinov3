import argparse
from pathlib import Path

from evaluate_e2b import run_e2b_evaluation
from train_pair_confidence import run_e2b_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate E2B.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e2b.yaml")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_e2b_training(args.config, args.overwrite)
    print(run_e2b_evaluation(args.config))


if __name__ == "__main__":
    main()
