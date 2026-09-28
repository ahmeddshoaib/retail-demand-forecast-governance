"""Apply the pre-specified model-complexity gate without subjective overrides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_selection import build_paired_robustness_table, evaluate_model_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    decision, sensitivity = evaluate_model_selection(metrics)
    robustness = build_paired_robustness_table(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    decision.to_csv(args.output, index=False)
    sensitivity_path = args.output.with_name("model_selection_sensitivity.csv")
    robustness_path = args.output.with_name("paired_model_robustness.csv")
    sensitivity.to_csv(sensitivity_path, index=False)
    robustness.to_csv(robustness_path, index=False)
    print(decision.to_string(index=False))
    print("\nSelection sensitivity checks:")
    print(sensitivity.to_string(index=False))
    print("\nPaired post-selection robustness checks:")
    print(robustness.to_string(index=False))


if __name__ == "__main__":
    main()
