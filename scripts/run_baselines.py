"""Build the category-store panel and execute the validated baseline pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import holt_winters_forecasts, seasonal_naive_forecasts  # noqa: E402
from src.data_pipeline import build_category_store_panel  # noqa: E402
from src.evaluation import build_metric_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    metrics_dir = args.output_dir / "metrics"
    powerbi_dir = args.output_dir / "powerbi"
    for directory in (processed_dir, metrics_dir, powerbi_dir):
        directory.mkdir(parents=True, exist_ok=True)

    panel = build_category_store_panel(args.sales, args.calendar)
    seasonal_forecasts, scales = seasonal_naive_forecasts(panel)
    holt_winters = holt_winters_forecasts(panel)
    forecasts = pd.concat([seasonal_forecasts, holt_winters], ignore_index=True)
    metrics = build_metric_table(forecasts, scales)

    panel.to_csv(processed_dir / "m5_category_store_panel.csv", index=False)
    seasonal_forecasts.to_csv(
        powerbi_dir / "fact_forecasts_seasonal_naive.csv", index=False
    )
    forecasts.to_csv(powerbi_dir / "fact_forecasts_baselines.csv", index=False)
    scales.to_csv(metrics_dir / "rmsse_scales.csv", index=False)
    metrics.to_csv(metrics_dir / "forecast_metrics_baselines.csv", index=False)
    metrics.loc[metrics["model"].eq("seasonal_naive_7")].to_csv(
        metrics_dir / "forecast_metrics_seasonal_naive.csv", index=False
    )

    overall = metrics.loc[metrics["scope"] == "overall"].copy()
    pd.set_option("display.max_columns", None)
    print(f"Panel rows: {len(panel):,}")
    print(f"Forecast rows: {len(forecasts):,}")
    print(f"Series: {panel['series_id'].nunique()}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
