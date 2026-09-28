"""Create validated, dashboard-ready files from final analytical outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_METADATA = pd.DataFrame(
    [
        {
            "model": "seasonal_naive_7",
            "display_name": "Seasonal naive",
            "model_family": "Benchmark",
            "complexity_order": 1,
        },
        {
            "model": "holt_winters_add_damped_7",
            "display_name": "Holt-Winters",
            "model_family": "Statistical",
            "complexity_order": 2,
        },
        {
            "model": "global_lightgbm_recursive",
            "display_name": "Global LightGBM",
            "model_family": "Machine learning",
            "complexity_order": 3,
        },
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = root / "outputs" / "powerbi"
    output.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(root / "data" / "processed" / "m5_category_store_panel.csv")
    forecasts = pd.read_csv(output / "fact_forecasts_all_models.csv")
    metrics = pd.read_csv(root / "outputs" / "metrics" / "forecast_metrics_all_models.csv")
    metrics["fold_id"] = metrics["fold_id"].fillna("ALL")
    metrics["series_id"] = metrics["series_id"].fillna("ALL")
    metrics["horizon_band"] = metrics["horizon_band"].fillna("ALL")
    selection = pd.read_csv(root / "outputs" / "metrics" / "model_selection.csv")
    importance = pd.read_csv(root / "outputs" / "metrics" / "lightgbm_feature_importance.csv")
    leakage = pd.read_csv(root / "outputs" / "metrics" / "lightgbm_leakage_test.csv")
    calendar = pd.read_csv(args.calendar)
    forward_path = output / "fact_forward_forecast.csv"
    category_forward_path = output / "fact_category_forward_forecast.csv"
    monitoring_path = output / "fact_forecast_monitoring.csv"
    forward = pd.read_csv(forward_path) if forward_path.exists() else None
    category_forward = (
        pd.read_csv(category_forward_path) if category_forward_path.exists() else None
    )
    monitoring = pd.read_csv(monitoring_path) if monitoring_path.exists() else None

    # Use one date-only representation across dimensions and facts. Baseline
    # exports were date strings while recursive LightGBM exports included a
    # midnight time component, which would otherwise break Power BI joins.
    panel["date"] = pd.to_datetime(panel["date"], format="mixed").dt.strftime("%Y-%m-%d")
    forecasts["origin_date"] = pd.to_datetime(
        forecasts["origin_date"], format="mixed"
    ).dt.strftime("%Y-%m-%d")
    forecasts["forecast_date"] = pd.to_datetime(
        forecasts["forecast_date"], format="mixed"
    ).dt.strftime("%Y-%m-%d")
    calendar["date"] = pd.to_datetime(calendar["date"], format="mixed").dt.strftime(
        "%Y-%m-%d"
    )
    if forward is not None:
        forward["origin_date"] = pd.to_datetime(
            forward["origin_date"], format="mixed"
        ).dt.strftime("%Y-%m-%d")
        forward["forecast_date"] = pd.to_datetime(
            forward["forecast_date"], format="mixed"
        ).dt.strftime("%Y-%m-%d")
    if category_forward is not None:
        category_forward["origin_date"] = pd.to_datetime(
            category_forward["origin_date"], format="mixed"
        ).dt.strftime("%Y-%m-%d")
        category_forward["forecast_date"] = pd.to_datetime(
            category_forward["forecast_date"], format="mixed"
        ).dt.strftime("%Y-%m-%d")

    dim_series = panel[["series_id", "category", "store", "state"]].drop_duplicates()
    calendar["day_number"] = calendar["d"].str.split("_").str[1].astype(int)
    dim_date = calendar.loc[
        calendar["day_number"].between(1, 1969),
        [
            "date",
            "d",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ],
    ].copy()
    dim_date["event_name_1"] = dim_date["event_name_1"].fillna("No event")
    dim_date["event_type_1"] = dim_date["event_type_1"].fillna("No event")
    dim_date["event_name_2"] = dim_date["event_name_2"].fillna("No event")
    dim_date["event_type_2"] = dim_date["event_type_2"].fillna("No event")

    actual = panel[["series_id", "date", "d", "day_number", "actual"]].copy()
    forecast_fact = forecasts[
        [
            "model",
            "fold_id",
            "origin_day",
            "origin_date",
            "series_id",
            "forecast_day",
            "forecast_date",
            "horizon_day",
            "horizon_band",
            "actual",
            "forecast",
            "error",
            "absolute_error",
            "squared_error",
        ]
    ].copy()
    forecast_fact.insert(0, "forecast_type", "Backtest")

    files = {
        "dim_date.csv": dim_date,
        "dim_series.csv": dim_series,
        "dim_model.csv": MODEL_METADATA,
        "fact_actual_demand.csv": actual,
        "fact_forecast_results.csv": forecast_fact,
        "fact_forecast_metrics.csv": metrics,
        "fact_model_selection.csv": selection,
        "fact_lightgbm_feature_importance.csv": importance,
        "fact_leakage_validation.csv": leakage,
    }
    if forward is not None and category_forward is not None and monitoring is not None:
        files.update(
            {
                "fact_forward_forecast.csv": forward,
                "fact_category_forward_forecast.csv": category_forward,
                "fact_forecast_monitoring.csv": monitoring,
            }
        )

    manifest_rows = []
    for filename, frame in files.items():
        if frame.empty:
            raise ValueError(f"Refusing to export empty Power BI table: {filename}")
        path = output / filename
        frame.to_csv(path, index=False)
        manifest_rows.append(
            {
                "file_name": filename,
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "missing_cells": int(frame.isna().sum().sum()),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output / "refresh_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
