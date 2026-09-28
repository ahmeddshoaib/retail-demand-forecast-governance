"""Leakage-safe recursive global LightGBM forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .config import FOLDS, HORIZON, LIGHTGBM_PARAMS, ForecastFold, horizon_band


LAG_DAYS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 28)
CATEGORICAL_FEATURES = (
    "category",
    "store",
    "state",
    "weekday",
    "month",
    "event_type_1",
    "event_type_2",
)
NUMERIC_FEATURES = (
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "snap",
)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_calendar(calendar_path: str | Path) -> pd.DataFrame:
    """Load only calendar fields known independently of future demand."""
    columns = [
        "date",
        "d",
        "weekday",
        "month",
        "event_type_1",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    calendar = pd.read_csv(calendar_path, usecols=columns)
    calendar["day_number"] = calendar["d"].str.split("_").str[1].astype(int)
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["month"] = calendar["month"].astype(str)
    calendar["event_type_1"] = calendar["event_type_1"].fillna("No event")
    calendar["event_type_2"] = calendar["event_type_2"].fillna("No event")
    return calendar


def _add_snap_indicator(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["snap"] = np.select(
        [result["state"].eq("CA"), result["state"].eq("TX"), result["state"].eq("WI")],
        [result["snap_CA"], result["snap_TX"], result["snap_WI"]],
        default=0,
    ).astype(float)
    return result.drop(columns=["snap_CA", "snap_TX", "snap_WI"])


def build_training_frame(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    origin_day: int,
) -> pd.DataFrame:
    """Construct shifted training features using origin-day history only."""
    history = panel.loc[panel["day_number"] <= origin_day].copy()
    history = history.sort_values(["series_id", "day_number"])
    grouped = history.groupby("series_id", observed=True, sort=False)["actual"]
    for lag in LAG_DAYS:
        history[f"lag_{lag}"] = grouped.shift(lag)
    shifted = grouped.shift(1)
    for window in ROLLING_WINDOWS:
        history[f"rolling_mean_{window}"] = shifted.groupby(
            history["series_id"], observed=True, sort=False
        ).transform(lambda values: values.rolling(window, min_periods=window).mean())

    calendar_fields = calendar.drop(columns=["date", "d"])
    history = history.merge(calendar_fields, on="day_number", how="left", validate="many_to_one")
    history = _add_snap_indicator(history)
    history = history.dropna(subset=list(NUMERIC_FEATURES)).reset_index(drop=True)
    return history


def _categorical_schema(training: pd.DataFrame) -> dict[str, list[str]]:
    return {
        column: sorted(training[column].astype(str).unique().tolist())
        for column in CATEGORICAL_FEATURES
    }


def _cast_categoricals(
    frame: pd.DataFrame,
    schema: dict[str, list[str]],
) -> pd.DataFrame:
    result = frame.copy()
    for column, categories in schema.items():
        result[column] = pd.Categorical(result[column].astype(str), categories=categories)
    return result


def _future_feature_rows(
    series_metadata: pd.DataFrame,
    histories: dict[str, list[float]],
    calendar_row: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for meta in series_metadata.itertuples(index=False):
        values = histories[meta.series_id]
        row: dict[str, object] = {
            "series_id": meta.series_id,
            "category": meta.category,
            "store": meta.store,
            "state": meta.state,
            "weekday": calendar_row["weekday"],
            "month": str(calendar_row["month"]),
            "event_type_1": calendar_row["event_type_1"],
            "event_type_2": calendar_row["event_type_2"],
        }
        for lag in LAG_DAYS:
            row[f"lag_{lag}"] = float(values[-lag])
        for window in ROLLING_WINDOWS:
            row[f"rolling_mean_{window}"] = float(np.mean(values[-window:]))
        snap_column = f"snap_{meta.state}"
        row["snap"] = float(calendar_row[snap_column])
        rows.append(row)
    return pd.DataFrame(rows)


def fit_predict_recursive_fold(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    fold: ForecastFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one global model and recursively predict one 28-day outer fold."""
    training = build_training_frame(panel, calendar, fold.origin_day)
    schema = _categorical_schema(training)
    training = _cast_categoricals(training, schema)
    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(
        training.loc[:, FEATURE_COLUMNS],
        training["actual"].astype(float),
        categorical_feature=list(CATEGORICAL_FEATURES),
    )

    series_metadata = (
        panel[["series_id", "category", "store", "state"]]
        .drop_duplicates()
        .sort_values("series_id")
        .reset_index(drop=True)
    )
    histories = {
        series_id: group.loc[group["day_number"] <= fold.origin_day]
        .sort_values("day_number")["actual"]
        .astype(float)
        .tolist()
        for series_id, group in panel.groupby("series_id", observed=True, sort=True)
    }
    calendar_lookup = calendar.set_index("day_number")
    prediction_rows: list[dict[str, object]] = []

    for horizon_day, forecast_day in enumerate(
        range(fold.test_start_day, fold.test_end_day + 1),
        start=1,
    ):
        calendar_row = calendar_lookup.loc[forecast_day]
        feature_rows = _future_feature_rows(series_metadata, histories, calendar_row)
        prediction_features = _cast_categoricals(feature_rows, schema)
        predicted = np.maximum(
            model.predict(prediction_features.loc[:, FEATURE_COLUMNS]),
            0.0,
        )
        for meta, forecast in zip(
            series_metadata.itertuples(index=False), predicted, strict=True
        ):
            histories[meta.series_id].append(float(forecast))
            prediction_rows.append(
                {
                    "model": "global_lightgbm_recursive",
                    "fold_id": fold.fold_id,
                    "origin_day": fold.origin_day,
                    "origin_date": calendar_lookup.loc[fold.origin_day, "date"],
                    "series_id": meta.series_id,
                    "category": meta.category,
                    "store": meta.store,
                    "state": meta.state,
                    "forecast_day": forecast_day,
                    "forecast_date": calendar_row["date"],
                    "horizon_day": horizon_day,
                    "horizon_band": horizon_band(horizon_day),
                    "forecast": float(forecast),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    actuals = panel[["series_id", "day_number", "actual"]].rename(
        columns={"day_number": "forecast_day"}
    )
    predictions = predictions.merge(
        actuals,
        on=["series_id", "forecast_day"],
        how="left",
        validate="one_to_one",
    )
    predictions["error"] = predictions["actual"] - predictions["forecast"]
    predictions["absolute_error"] = predictions["error"].abs()
    predictions["squared_error"] = predictions["error"] ** 2

    importance = pd.DataFrame(
        {
            "fold_id": fold.fold_id,
            "feature": model.feature_name_,
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
        }
    )
    gain_total = importance["gain_importance"].sum()
    importance["gain_share"] = (
        importance["gain_importance"] / gain_total if gain_total else 0.0
    )
    return predictions, importance


def recursive_lightgbm_forecasts(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the frozen recursive model across all four outer folds."""
    forecasts = []
    importances = []
    for fold in FOLDS:
        fold_forecasts, fold_importance = fit_predict_recursive_fold(panel, calendar, fold)
        forecasts.append(fold_forecasts)
        importances.append(fold_importance)
    return pd.concat(forecasts, ignore_index=True), pd.concat(importances, ignore_index=True)


def fit_predict_full_history(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    origin_day: int = 1941,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the selected model through the final observed day and forecast forward."""
    if horizon != HORIZON:
        raise ValueError(f"The dissertation forecast horizon is frozen at {HORIZON} days")
    if int(panel["day_number"].max()) != origin_day:
        raise ValueError(
            f"Expected demand history to end at d_{origin_day}; "
            f"found d_{int(panel['day_number'].max())}"
        )

    forecast_days = list(range(origin_day + 1, origin_day + horizon + 1))
    calendar_days = set(calendar["day_number"].astype(int))
    missing_calendar_days = [day for day in forecast_days if day not in calendar_days]
    if missing_calendar_days:
        raise ValueError(f"Calendar is missing forward days: {missing_calendar_days}")

    training = build_training_frame(panel, calendar, origin_day)
    schema = _categorical_schema(training)
    training = _cast_categoricals(training, schema)
    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(
        training.loc[:, FEATURE_COLUMNS],
        training["actual"].astype(float),
        categorical_feature=list(CATEGORICAL_FEATURES),
    )

    series_metadata = (
        panel[["series_id", "category", "store", "state"]]
        .drop_duplicates()
        .sort_values("series_id")
        .reset_index(drop=True)
    )
    histories = {
        series_id: group.sort_values("day_number")["actual"].astype(float).tolist()
        for series_id, group in panel.groupby("series_id", observed=True, sort=True)
    }
    calendar_lookup = calendar.set_index("day_number")
    origin_date = calendar_lookup.loc[origin_day, "date"]
    prediction_rows: list[dict[str, object]] = []

    for horizon_day, forecast_day in enumerate(forecast_days, start=1):
        calendar_row = calendar_lookup.loc[forecast_day]
        feature_rows = _future_feature_rows(series_metadata, histories, calendar_row)
        prediction_features = _cast_categoricals(feature_rows, schema)
        predicted = np.maximum(
            model.predict(prediction_features.loc[:, FEATURE_COLUMNS]),
            0.0,
        )
        for meta, forecast in zip(
            series_metadata.itertuples(index=False), predicted, strict=True
        ):
            histories[meta.series_id].append(float(forecast))
            prediction_rows.append(
                {
                    "forecast_type": "Forward forecast",
                    "model": "global_lightgbm_recursive",
                    "origin_day": origin_day,
                    "origin_date": origin_date,
                    "series_id": meta.series_id,
                    "category": meta.category,
                    "store": meta.store,
                    "state": meta.state,
                    "forecast_day": forecast_day,
                    "forecast_date": calendar_row["date"],
                    "horizon_day": horizon_day,
                    "horizon_band": horizon_band(horizon_day),
                    "forecast": float(forecast),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    importance = pd.DataFrame(
        {
            "model_run": "full_history_d_1941",
            "feature": model.feature_name_,
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
        }
    )
    gain_total = importance["gain_importance"].sum()
    importance["gain_share"] = (
        importance["gain_importance"] / gain_total if gain_total else 0.0
    )
    return predictions, importance


def verify_post_origin_masking(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    reference_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Test whether post-origin demand changes the implemented forecast pipeline."""
    results = []
    for fold in FOLDS:
        masked = panel.copy()
        masked.loc[masked["day_number"] > fold.origin_day, "actual"] = np.nan
        masked_forecasts, _ = fit_predict_recursive_fold(masked, calendar, fold)
        expected = reference_forecasts.loc[reference_forecasts["fold_id"] == fold.fold_id].sort_values(
            ["series_id", "forecast_day"]
        )
        observed = masked_forecasts.sort_values(["series_id", "forecast_day"])
        difference = np.abs(expected["forecast"].to_numpy() - observed["forecast"].to_numpy())
        maximum_difference = float(np.max(difference))
        passed = bool(np.allclose(difference, 0.0, atol=1e-10, rtol=0.0))
        results.append(
            {
                "fold_id": fold.fold_id,
                "origin_day": fold.origin_day,
                "n_predictions": len(difference),
                "maximum_absolute_difference": maximum_difference,
                "passed": passed,
            }
        )
        if not passed:
            raise AssertionError(f"Post-origin masking leakage test failed for {fold.fold_id}")
    return pd.DataFrame(results)
