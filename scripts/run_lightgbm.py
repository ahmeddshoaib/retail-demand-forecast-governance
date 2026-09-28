"""Run recursive LightGBM, leakage checks and combined model evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import build_metric_table  # noqa: E402
from src.lightgbm_model import (  # noqa: E402
    prepare_calendar,
    recursive_lightgbm_forecasts,
    verify_post_origin_masking,
)
from src.project_paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "m5_category_store_panel.csv",
    )
    parser.add_argument("--calendar", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--verify-leakage", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(PROJECT_ROOT)
    calendar_path = args.calendar or paths.m5_file("calendar.csv")
    panel = pd.read_csv(args.panel, parse_dates=["date"])
    calendar = prepare_calendar(calendar_path)
    lightgbm_forecasts, feature_importance = recursive_lightgbm_forecasts(panel, calendar)

    baseline_path = args.output_dir / "powerbi" / "fact_forecasts_baselines.csv"
    scales_path = args.output_dir / "metrics" / "rmsse_scales.csv"
    baselines = pd.read_csv(baseline_path)
    scales = pd.read_csv(scales_path)
    all_forecasts = pd.concat([baselines, lightgbm_forecasts], ignore_index=True)
    metrics = build_metric_table(all_forecasts, scales)

    powerbi_dir = args.output_dir / "powerbi"
    metrics_dir = args.output_dir / "metrics"
    powerbi_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    all_forecasts.to_csv(powerbi_dir / "fact_forecasts_all_models.csv", index=False)
    lightgbm_forecasts.to_csv(
        powerbi_dir / "fact_forecasts_lightgbm.csv", index=False
    )
    metrics.to_csv(metrics_dir / "forecast_metrics_all_models.csv", index=False)
    feature_importance.to_csv(metrics_dir / "lightgbm_feature_importance.csv", index=False)

    if args.verify_leakage:
        leakage = verify_post_origin_masking(panel, calendar, lightgbm_forecasts)
        leakage.to_csv(metrics_dir / "lightgbm_leakage_test.csv", index=False)
        print(leakage.to_string(index=False))

    overall = metrics.loc[metrics["scope"] == "overall"]
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
