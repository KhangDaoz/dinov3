#!/usr/bin/env python
"""Evaluate M1--M4 on test and select the E2A winner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs=4, required=True, type=Path)
    args = parser.parse_args()
    for config in args.configs:
        subprocess.run(
            [sys.executable, "scripts/run_e2a.py", "--config", str(config),
             "--stage", "test"], check=True,
        )
    subprocess.run(
        [sys.executable, "scripts/select_e2a_winner.py", "--configs",
         *(str(path) for path in args.configs)], check=True,
    )


if __name__ == "__main__":
    main()
