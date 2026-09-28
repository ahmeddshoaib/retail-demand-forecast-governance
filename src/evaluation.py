"""Forecast evaluation using pre-specified dissertation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmsse_scale(training_actual: np.ndarray) -> float:
    """Return the M5-style one-step squared-difference scale from training data."""
    values = np.asarray(training_actual, dtype=float)
    nonzero = np.flatnonzero(values != 0)
    if len(nonzero) == 0:
        raise ValueError("RMSSE is undefined for an all-zero training series")
    trimmed = values[nonzero[0] :]
    if len(trimmed) < 2:
        raise ValueError("RMSSE requires at least two training observations")
    scale = float(np.mean(np.diff(trimmed) ** 2))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("RMSSE scale must be finite and positive")
    return scale


def metric_values(actual: np.ndarray, forecast: np.ndarray, scale: float) -> dict[str, float]:
    """Calculate RMSSE, WAPE, MAE and RMSE for one evaluation slice."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape or actual.size == 0:
        raise ValueError("Actual and forecast arrays must have the same non-zero length")
    error = actual - forecast
    absolute_error = np.abs(error)
    denominator = float(np.sum(np.abs(actual)))
    return {
        "rmsse": float(np.sqrt(np.mean(error**2) / scale)),
        "wape": float(np.sum(absolute_error) / denominator) if denominator else np.nan,
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def build_metric_table(forecasts: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    """Create series-fold, fold, horizon-band and overall metric summaries."""
    required = {
        "model",
        "fold_id",
        "series_id",
        "horizon_band",
        "actual",
        "forecast",
    }
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Forecast table missing columns: {sorted(missing)}")

    scale_lookup = scales.set_index(["fold_id", "series_id"])["rmsse_scale"]
    rows: list[dict[str, object]] = []

    def add_group(scope: str, keys: tuple[str, ...]) -> None:
        grouped = forecasts.groupby(list(keys), observed=True, sort=True)
        for key, frame in grouped:
            key = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(keys, key, strict=True))
            series_scales = frame[["fold_id", "series_id"]].drop_duplicates()
            squared_scaled = []
            for item in series_scales.itertuples(index=False):
                subset = frame.loc[
                    (frame["fold_id"] == item.fold_id)
                    & (frame["series_id"] == item.series_id)
                ]
                scale = float(scale_lookup.loc[(item.fold_id, item.series_id)])
                squared_scaled.append(
                    float(np.mean((subset["actual"] - subset["forecast"]) ** 2) / scale)
                )
            values = metric_values(
                frame["actual"].to_numpy(),
                frame["forecast"].to_numpy(),
                scale=1.0,
            )
            values["rmsse"] = float(np.mean(np.sqrt(squared_scaled)))
            rows.append(
                {
                    "scope": scope,
                    "model": labels.get("model"),
                    "fold_id": labels.get("fold_id"),
                    "series_id": labels.get("series_id"),
                    "horizon_band": labels.get("horizon_band"),
                    "n_forecasts": len(frame),
                    **values,
                }
            )

    add_group("series_fold", ("model", "fold_id", "series_id"))
    add_group("fold", ("model", "fold_id"))
    add_group("horizon_band", ("model", "horizon_band"))
    add_group("overall", ("model",))
    return pd.DataFrame(rows)

