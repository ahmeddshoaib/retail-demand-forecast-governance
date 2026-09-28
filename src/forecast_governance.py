"""Historical-error monitoring outputs for the final forward forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd


SELECTED_MODEL = "global_lightgbm_recursive"


def build_series_monitoring_profile(
    backtests: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise selected-model reliability without inventing business thresholds."""
    selected = backtests.loc[backtests["model"].eq(SELECTED_MODEL)].copy()
    if selected.empty:
        raise ValueError("No selected-model backtests were supplied")

    series_fold = metrics.loc[
        metrics["scope"].eq("series_fold") & metrics["model"].eq(SELECTED_MODEL)
    ].copy()
    if series_fold.empty:
        raise ValueError("No selected-model series-fold metrics were supplied")

    accuracy = (
        series_fold.groupby("series_id", observed=True)
        .agg(
            historical_mean_rmsse=("rmsse", "mean"),
            historical_max_fold_rmsse=("rmsse", "max"),
            evaluated_folds=("fold_id", "nunique"),
        )
        .reset_index()
    )
    errors = (
        selected.groupby(["series_id", "category", "store", "state"], observed=True)
        .agg(
            evaluated_forecasts=("forecast", "size"),
            historical_absolute_error=("absolute_error", "sum"),
            historical_actual_units=("actual", "sum"),
            historical_bias_units=("error", "mean"),
        )
        .reset_index()
    )
    errors["historical_wape"] = (
        errors["historical_absolute_error"] / errors["historical_actual_units"]
    )
    profile = errors.merge(accuracy, on="series_id", how="left", validate="one_to_one")

    ranked = profile["historical_mean_rmsse"].rank(method="first")
    profile["monitoring_priority"] = pd.qcut(
        ranked,
        q=3,
        labels=["Standard", "Review", "High review"],
    ).astype(str)
    profile["threshold_basis"] = "Relative tercile of historical mean RMSSE"
    return profile.sort_values("historical_mean_rmsse", ascending=False).reset_index(drop=True)


def attach_historical_error_bands(
    forward: pd.DataFrame,
    backtests: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """Attach descriptive 10th-90th percentile residual bands to series forecasts."""
    selected = backtests.loc[backtests["model"].eq(SELECTED_MODEL)].copy()
    residuals = (
        selected.groupby(["series_id", "horizon_band"], observed=True)["error"]
        .quantile([0.10, 0.90])
        .unstack()
        .rename(columns={0.10: "historical_error_q10", 0.90: "historical_error_q90"})
        .reset_index()
    )
    profile_columns = [
        "series_id",
        "historical_mean_rmsse",
        "historical_max_fold_rmsse",
        "historical_wape",
        "historical_bias_units",
        "monitoring_priority",
    ]
    result = forward.merge(
        residuals,
        on=["series_id", "horizon_band"],
        how="left",
        validate="many_to_one",
    ).merge(
        profile[profile_columns],
        on="series_id",
        how="left",
        validate="many_to_one",
    )
    if result[["historical_error_q10", "historical_error_q90"]].isna().any().any():
        raise ValueError("Historical residual bands could not be assigned to every forecast")

    result["historical_band_lower"] = np.maximum(
        result["forecast"] + result["historical_error_q10"], 0.0
    )
    result["historical_band_upper"] = np.maximum(
        result["forecast"] + result["historical_error_q90"],
        result["historical_band_lower"],
    )
    result["uncertainty_method"] = (
        "Descriptive 10th-90th percentile backtest residual band by series and horizon band"
    )
    return result


def build_category_forward_forecast(
    forward: pd.DataFrame,
    backtests: pd.DataFrame,
) -> pd.DataFrame:
    """Create executive category totals with category-level historical residual bands."""
    totals = (
        forward.groupby(
            [
                "forecast_type",
                "model",
                "origin_day",
                "origin_date",
                "category",
                "forecast_day",
                "forecast_date",
                "horizon_day",
                "horizon_band",
            ],
            observed=True,
            as_index=False,
        )["forecast"]
        .sum()
        .rename(columns={"forecast": "category_forecast"})
    )

    selected = backtests.loc[backtests["model"].eq(SELECTED_MODEL)].copy()
    category_history = (
        selected.groupby(
            ["fold_id", "category", "forecast_date", "horizon_day", "horizon_band"],
            observed=True,
            as_index=False,
        )[["actual", "forecast"]]
        .sum()
    )
    category_history["error"] = (
        category_history["actual"] - category_history["forecast"]
    )
    residuals = (
        category_history.groupby(["category", "horizon_band"], observed=True)["error"]
        .quantile([0.10, 0.90])
        .unstack()
        .rename(columns={0.10: "historical_error_q10", 0.90: "historical_error_q90"})
        .reset_index()
    )
    result = totals.merge(
        residuals,
        on=["category", "horizon_band"],
        how="left",
        validate="many_to_one",
    )
    result["historical_band_lower"] = np.maximum(
        result["category_forecast"] + result["historical_error_q10"], 0.0
    )
    result["historical_band_upper"] = np.maximum(
        result["category_forecast"] + result["historical_error_q90"],
        result["historical_band_lower"],
    )
    result["uncertainty_method"] = (
        "Descriptive 10th-90th percentile category backtest residual band by horizon band"
    )
    return result
