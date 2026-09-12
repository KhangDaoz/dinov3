import argparse
from pathlib import Path

from evaluate import run_evaluation
from extract_features import run_extraction


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run E2A extraction and evaluation.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e2a.yaml")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_extraction(args.config, overwrite=args.overwrite)
    recalls = run_evaluation(args.config)
    for name, value in recalls.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
