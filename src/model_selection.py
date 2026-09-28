"""Shared model-selection and post-selection robustness calculations."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from .config import RANDOM_STATE


LIGHTGBM = "global_lightgbm_recursive"
SEASONAL_NAIVE = "seasonal_naive_7"
HOLT_WINTERS = "holt_winters_add_damped_7"
REQUIRED_SERIES_WINS = 16


def _model_metric_views(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return validated overall, series and fold RMSSE views."""
    required_columns = {"scope", "model", "rmsse"}
    missing_columns = required_columns - set(metrics.columns)
    if missing_columns:
        raise ValueError(f"Metrics table missing columns: {sorted(missing_columns)}")

    required_models = {LIGHTGBM, SEASONAL_NAIVE, HOLT_WINTERS}
    overall = metrics.loc[metrics["scope"] == "overall"].set_index("model")
    missing_models = required_models - set(overall.index)
    if missing_models:
        raise ValueError(f"Missing overall model results: {sorted(missing_models)}")

    series_fold = metrics.loc[metrics["scope"] == "series_fold"].pivot(
        index=["fold_id", "series_id"], columns="model", values="rmsse"
    )
    fold = metrics.loc[metrics["scope"] == "fold"].pivot(
        index="fold_id", columns="model", values="rmsse"
    )
    for name, view in (("series-fold", series_fold), ("fold", fold)):
        missing_models = required_models - set(view.columns)
        if missing_models:
            raise ValueError(f"Missing {name} model results: {sorted(missing_models)}")
    return overall, series_fold.groupby("series_id").mean(), fold


def evaluate_model_selection(
    metrics: pd.DataFrame,
    required_series_wins: int = REQUIRED_SERIES_WINS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the original gate once and report stricter sensitivity conditions."""
    overall, series_mean, fold = _model_metric_views(metrics)
    lower_than_both = bool(
        overall.loc[LIGHTGBM, "rmsse"] < overall.loc[SEASONAL_NAIVE, "rmsse"]
        and overall.loc[LIGHTGBM, "rmsse"] < overall.loc[HOLT_WINTERS, "rmsse"]
    )
    wins_vs_naive = int((series_mean[LIGHTGBM] < series_mean[SEASONAL_NAIVE]).sum())
    wins_vs_holt_winters = int(
        (series_mean[LIGHTGBM] < series_mean[HOLT_WINTERS]).sum()
    )
    folds_won_vs_holt_winters = int((fold[LIGHTGBM] < fold[HOLT_WINTERS]).sum())
    lightgbm_selected = bool(
        lower_than_both and wins_vs_naive >= required_series_wins
    )

    if lightgbm_selected:
        selected_model = LIGHTGBM
        rationale = (
            "LightGBM passed the pre-specified complexity gate: lower overall RMSSE than "
            "both baselines and lower mean fold RMSSE than seasonal-naive on at least "
            f"{required_series_wins} series."
        )
    else:
        baseline_rmsse = overall.loc[[SEASONAL_NAIVE, HOLT_WINTERS], "rmsse"]
        selected_model = str(baseline_rmsse.idxmin())
        rationale = (
            "LightGBM did not pass the pre-specified complexity gate; the baseline with "
            "the lower overall RMSSE was retained."
        )

    decision = pd.DataFrame(
        [
            {
                "selected_model": selected_model,
                "lightgbm_lower_rmsse_than_both_baselines": lower_than_both,
                "lightgbm_series_wins_vs_seasonal_naive": wins_vs_naive,
                "lightgbm_series_wins_vs_holt_winters": wins_vs_holt_winters,
                "required_series_wins": required_series_wins,
                "lightgbm_selected": lightgbm_selected,
                "rationale": rationale,
            }
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "condition": "Overall RMSSE lower than both baselines",
                "observed": lower_than_both,
                "required": True,
                "passed": lower_than_both,
                "role": "Original gate",
            },
            {
                "condition": "Series wins versus seasonal-naive",
                "observed": wins_vs_naive,
                "required": required_series_wins,
                "passed": wins_vs_naive >= required_series_wins,
                "role": "Original gate",
            },
            {
                "condition": "Series wins versus Holt-Winters",
                "observed": wins_vs_holt_winters,
                "required": required_series_wins,
                "passed": wins_vs_holt_winters >= required_series_wins,
                "role": "Stricter sensitivity check",
            },
            {
                "condition": "Folds won versus Holt-Winters",
                "observed": folds_won_vs_holt_winters,
                "required": len(fold),
                "passed": folds_won_vs_holt_winters == len(fold),
                "role": "Stricter sensitivity check",
            },
        ]
    )
    return decision, sensitivity


def _exact_two_sided_sign_pvalue(differences: np.ndarray) -> float:
    """Calculate an exact two-sided sign-test p-value after excluding ties."""
    values = np.asarray(differences, dtype=float)
    values = values[values != 0]
    n = len(values)
    if n == 0:
        return 1.0
    wins = int((values < 0).sum())
    extreme = max(wins, n - wins)
    upper_tail = sum(comb(n, k) for k in range(extreme, n + 1)) / (2**n)
    return float(min(1.0, 2 * upper_tail))


def build_paired_robustness_table(
    metrics: pd.DataFrame,
    *,
    n_bootstrap: int = 20_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compare paired series RMSSE differences as a post-selection sensitivity check."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    _, series_mean, _ = _model_metric_views(metrics)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []

    for baseline in (SEASONAL_NAIVE, HOLT_WINTERS):
        differences = (series_mean[LIGHTGBM] - series_mean[baseline]).to_numpy()
        bootstrap_means = np.empty(n_bootstrap, dtype=float)
        for index in range(n_bootstrap):
            bootstrap_means[index] = rng.choice(
                differences, size=len(differences), replace=True
            ).mean()
        interval_low, interval_high = np.quantile(bootstrap_means, [0.025, 0.975])
        rows.append(
            {
                "candidate_model": LIGHTGBM,
                "baseline_model": baseline,
                "n_series": len(differences),
                "candidate_series_wins": int((differences < 0).sum()),
                "mean_rmsse_difference": float(differences.mean()),
                "median_rmsse_difference": float(np.median(differences)),
                "bootstrap_interval_low": float(interval_low),
                "bootstrap_interval_high": float(interval_high),
                "exact_sign_test_pvalue": _exact_two_sided_sign_pvalue(differences),
                "interpretation": "Negative RMSSE differences favour LightGBM",
            }
        )
    return pd.DataFrame(rows)
