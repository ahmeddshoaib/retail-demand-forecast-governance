"""Frozen modelling configuration agreed after first supervision."""

from dataclasses import dataclass


RANDOM_STATE = 42
HORIZON = 28
SEASONAL_PERIOD = 7
EXPECTED_SERIES = 30
EXPECTED_DAYS = 1941

LIGHTGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.90,
    "subsample_freq": 1,
    "colsample_bytree": 0.90,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "n_jobs": -1,
}


@dataclass(frozen=True)
class ForecastFold:
    fold_id: str
    origin_day: int
    test_start_day: int
    test_end_day: int


FOLDS = (
    ForecastFold("F1", 1829, 1830, 1857),
    ForecastFold("F2", 1857, 1858, 1885),
    ForecastFold("F3", 1885, 1886, 1913),
    ForecastFold("F4", 1913, 1914, 1941),
)


def horizon_band(horizon_day: int) -> str:
    """Return the pre-specified seven-day horizon band."""
    if not 1 <= horizon_day <= HORIZON:
        raise ValueError(f"horizon_day must be 1-{HORIZON}; got {horizon_day}")
    lower = ((horizon_day - 1) // 7) * 7 + 1
    upper = lower + 6
    return f"D{lower:02d}-D{upper:02d}"
