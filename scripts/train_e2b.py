#!/usr/bin/env python
"""Train only the E2B pair head; launch with torchrun."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uncertainty_retrieval.training.pair_confidence import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    train(parser.parse_args().config)
