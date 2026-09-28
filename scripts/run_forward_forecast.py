"""Generate the final full-history 28-day forecast and governance outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast_governance import (  # noqa: E402
    SELECTED_MODEL,
    attach_historical_error_bands,
    build_category_forward_forecast,
    build_series_monitoring_profile,
)
from src.lightgbm_model import fit_predict_full_history, prepare_calendar  # noqa: E402
from src.project_paths import ProjectPaths  # noqa: E402


def main() -> None:
    paths = ProjectPaths(PROJECT_ROOT)
    paths.ensure_output_directories()

    selection = pd.read_csv(paths.metrics / "model_selection.csv")
    selected_model = str(selection.loc[0, "selected_model"])
    if selected_model != SELECTED_MODEL or not bool(selection.loc[0, "lightgbm_selected"]):
        raise RuntimeError(
            "The frozen selection output does not authorise a full-history LightGBM run"
        )

    panel = pd.read_csv(
        paths.processed / "m5_category_store_panel.csv",
        parse_dates=["date"],
    )
    calendar = prepare_calendar(paths.m5_file("calendar.csv"))
    backtests = pd.read_csv(
        paths.powerbi / "fact_forecasts_all_models.csv",
        parse_dates=["origin_date", "forecast_date"],
    )
    metrics = pd.read_csv(paths.metrics / "forecast_metrics_all_models.csv")

    forward, importance = fit_predict_full_history(panel, calendar)
    profile = build_series_monitoring_profile(backtests, metrics)
    forward = attach_historical_error_bands(forward, backtests, profile)
    category = build_category_forward_forecast(forward, backtests)

    expected_rows = 30 * 28
    if len(forward) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} forward rows; found {len(forward)}")
    if forward.duplicated(["series_id", "forecast_day"]).any():
        raise AssertionError("Duplicate category-store forward forecast keys found")
    if forward["forecast"].isna().any() or forward["forecast"].lt(0).any():
        raise AssertionError("Forward forecasts must be complete and non-negative")
    if len(category) != 3 * 28:
        raise AssertionError(f"Expected 84 category rows; found {len(category)}")
    if forward["forecast_day"].min() != 1942 or forward["forecast_day"].max() != 1969:
        raise AssertionError("Forward forecast dates do not cover d_1942-d_1969")

    forward.to_csv(paths.powerbi / "fact_forward_forecast.csv", index=False)
    category.to_csv(paths.powerbi / "fact_category_forward_forecast.csv", index=False)
    profile.to_csv(paths.powerbi / "fact_forecast_monitoring.csv", index=False)
    importance.to_csv(paths.metrics / "full_history_lightgbm_feature_importance.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "output": "fact_forward_forecast.csv",
                "rows": len(forward),
            },
            {
                "output": "fact_category_forward_forecast.csv",
                "rows": len(category),
            },
            {
                "output": "fact_forecast_monitoring.csv",
                "rows": len(profile),
            },
        ]
    )
    manifest.to_csv(paths.powerbi / "forward_forecast_manifest.csv", index=False)

    print(f"Selected model: {selected_model}")
    print(f"Category-store forward rows: {len(forward):,}")
    print(f"Category-total forward rows: {len(category):,}")
    print(
        "Forecast period: "
        f"{forward['forecast_date'].min().date()} to {forward['forecast_date'].max().date()}"
    )
    print("Forecast type: Forward forecast")
    print(category.groupby("category", observed=True)["category_forecast"].sum().to_string())


if __name__ == "__main__":
    main()
