"""Leakage-safe baseline forecast generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .config import FOLDS, HORIZON, SEASONAL_PERIOD, horizon_band
from .evaluation import rmsse_scale


def seasonal_naive_forecasts(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the final observed weekly pattern for every 28-day fold."""
    forecast_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []

    for fold in FOLDS:
        for series_id, series in panel.groupby("series_id", sort=True, observed=True):
            series = series.sort_values("day_number")
            train = series.loc[series["day_number"] <= fold.origin_day]
            test = series.loc[
                series["day_number"].between(fold.test_start_day, fold.test_end_day)
            ]
            if len(test) != HORIZON:
                raise ValueError(f"{fold.fold_id} {series_id} does not contain {HORIZON} test days")
            if len(train) < SEASONAL_PERIOD:
                raise ValueError(f"Insufficient seasonal history for {series_id}")

            weekly_pattern = train["actual"].tail(SEASONAL_PERIOD).to_numpy(dtype=float)
            predicted = np.resize(weekly_pattern, HORIZON)
            scale_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "series_id": series_id,
                    "rmsse_scale": rmsse_scale(train["actual"].to_numpy()),
                }
            )

            metadata = test.iloc[0][["category", "store", "state"]].to_dict()
            for horizon_day, (row, forecast) in enumerate(
                zip(test.itertuples(index=False), predicted, strict=True),
                start=1,
            ):
                forecast_rows.append(
                    {
                        "model": "seasonal_naive_7",
                        "fold_id": fold.fold_id,
                        "origin_day": fold.origin_day,
                        "origin_date": train.iloc[-1]["date"],
                        "series_id": series_id,
                        **metadata,
                        "forecast_day": row.day_number,
                        "forecast_date": row.date,
                        "horizon_day": horizon_day,
                        "horizon_band": horizon_band(horizon_day),
                        "actual": float(row.actual),
                        "forecast": float(forecast),
                    }
                )

    forecasts = pd.DataFrame(forecast_rows)
    forecasts["error"] = forecasts["actual"] - forecasts["forecast"]
    forecasts["absolute_error"] = forecasts["error"].abs()
    forecasts["squared_error"] = forecasts["error"] ** 2
    scales = pd.DataFrame(scale_rows)
    return forecasts, scales


def _holt_winters_point_forecast(training_actual: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Fit the frozen additive, damped Holt-Winters specification."""
    model = ExponentialSmoothing(
        np.asarray(training_actual, dtype=float),
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True, remove_bias=False)
    return np.maximum(np.asarray(fitted.forecast(horizon), dtype=float), 0.0)


def holt_winters_forecasts(panel: pd.DataFrame) -> pd.DataFrame:
    """Generate fixed-specification local Holt-Winters forecasts for all folds."""
    forecast_rows: list[dict[str, object]] = []

    for fold in FOLDS:
        for series_id, series in panel.groupby("series_id", sort=True, observed=True):
            series = series.sort_values("day_number")
            train = series.loc[series["day_number"] <= fold.origin_day]
            test = series.loc[
                series["day_number"].between(fold.test_start_day, fold.test_end_day)
            ]
            if len(test) != HORIZON:
                raise ValueError(f"{fold.fold_id} {series_id} does not contain {HORIZON} test days")

            predicted = _holt_winters_point_forecast(train["actual"].to_numpy())
            metadata = test.iloc[0][["category", "store", "state"]].to_dict()
            for horizon_day, (row, forecast) in enumerate(
                zip(test.itertuples(index=False), predicted, strict=True),
                start=1,
            ):
                forecast_rows.append(
                    {
                        "model": "holt_winters_add_damped_7",
                        "fold_id": fold.fold_id,
                        "origin_day": fold.origin_day,
                        "origin_date": train.iloc[-1]["date"],
                        "series_id": series_id,
                        **metadata,
                        "forecast_day": row.day_number,
                        "forecast_date": row.date,
                        "horizon_day": horizon_day,
                        "horizon_band": horizon_band(horizon_day),
                        "actual": float(row.actual),
                        "forecast": float(forecast),
                    }
                )

    forecasts = pd.DataFrame(forecast_rows)
    forecasts["error"] = forecasts["actual"] - forecasts["forecast"]
    forecasts["absolute_error"] = forecasts["error"].abs()
    forecasts["squared_error"] = forecasts["error"] ** 2
    return forecasts
