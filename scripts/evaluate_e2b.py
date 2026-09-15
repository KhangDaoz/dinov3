#!/usr/bin/env python
"""Validate or benchmark a frozen E2B checkpoint; launch with torchrun."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uncertainty_retrieval.evaluation.e2b_pipeline import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    evaluate(args.config, args.stage)
