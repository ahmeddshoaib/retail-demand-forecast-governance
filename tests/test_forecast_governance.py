import unittest

import numpy as np
import pandas as pd

from src.forecast_governance import (
    SELECTED_MODEL,
    attach_historical_error_bands,
    build_category_forward_forecast,
    build_series_monitoring_profile,
)


def _backtests() -> pd.DataFrame:
    rows = []
    for fold_number in range(1, 5):
        for series_number, category in enumerate(["FOODS", "HOBBIES", "HOUSEHOLD"], start=1):
            for horizon_day in range(1, 29):
                actual = 100.0 + series_number * 10 + horizon_day
                error = float((horizon_day % 5) - 2 + series_number)
                forecast = actual - error
                rows.append(
                    {
                        "model": SELECTED_MODEL,
                        "fold_id": f"F{fold_number}",
                        "series_id": f"{category}_S1",
                        "category": category,
                        "store": "S1",
                        "state": "X",
                        "forecast_date": pd.Timestamp("2016-01-01")
                        + pd.Timedelta(days=horizon_day),
                        "horizon_day": horizon_day,
                        "horizon_band": f"D{((horizon_day - 1) // 7) * 7 + 1:02d}-D{((horizon_day - 1) // 7) * 7 + 7:02d}",
                        "actual": actual,
                        "forecast": forecast,
                        "error": error,
                        "absolute_error": abs(error),
                    }
                )
    return pd.DataFrame(rows)


def _metrics() -> pd.DataFrame:
    rows = []
    for fold_number in range(1, 5):
        for series_number, category in enumerate(["FOODS", "HOBBIES", "HOUSEHOLD"], start=1):
            rows.append(
                {
                    "scope": "series_fold",
                    "model": SELECTED_MODEL,
                    "fold_id": f"F{fold_number}",
                    "series_id": f"{category}_S1",
                    "rmsse": 0.4 + series_number * 0.1,
                }
            )
    return pd.DataFrame(rows)


def _forward() -> pd.DataFrame:
    rows = []
    for series_number, category in enumerate(["FOODS", "HOBBIES", "HOUSEHOLD"], start=1):
        for horizon_day in range(1, 29):
            rows.append(
                {
                    "forecast_type": "Forward forecast",
                    "model": SELECTED_MODEL,
                    "origin_day": 1941,
                    "origin_date": pd.Timestamp("2016-05-22"),
                    "series_id": f"{category}_S1",
                    "category": category,
                    "store": "S1",
                    "state": "X",
                    "forecast_day": 1941 + horizon_day,
                    "forecast_date": pd.Timestamp("2016-05-22")
                    + pd.Timedelta(days=horizon_day),
                    "horizon_day": horizon_day,
                    "horizon_band": f"D{((horizon_day - 1) // 7) * 7 + 1:02d}-D{((horizon_day - 1) // 7) * 7 + 7:02d}",
                    "forecast": 120.0 + series_number,
                }
            )
    return pd.DataFrame(rows)


class ForecastGovernanceTests(unittest.TestCase):
    def test_monitoring_profile_covers_each_series(self) -> None:
        profile = build_series_monitoring_profile(_backtests(), _metrics())
        self.assertEqual(len(profile), 3)
        self.assertEqual(profile["series_id"].nunique(), 3)
        self.assertTrue(profile["historical_wape"].between(0, 1).all())

    def test_historical_bands_are_complete_and_ordered(self) -> None:
        profile = build_series_monitoring_profile(_backtests(), _metrics())
        result = attach_historical_error_bands(_forward(), _backtests(), profile)
        self.assertEqual(len(result), 3 * 28)
        self.assertFalse(result[["historical_band_lower", "historical_band_upper"]].isna().any().any())
        self.assertTrue(
            np.less_equal(result["historical_band_lower"], result["historical_band_upper"]).all()
        )

    def test_category_totals_have_three_complete_series(self) -> None:
        result = build_category_forward_forecast(_forward(), _backtests())
        self.assertEqual(len(result), 3 * 28)
        self.assertEqual(result["category"].nunique(), 3)
        self.assertTrue(result["category_forecast"].gt(0).all())


if __name__ == "__main__":
    unittest.main()
