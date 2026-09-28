import unittest

import numpy as np
import pandas as pd

from src.baselines import _holt_winters_point_forecast, seasonal_naive_forecasts
from src.config import EXPECTED_DAYS, EXPECTED_SERIES, FOLDS, horizon_band
from src.evaluation import metric_values, rmsse_scale


def _synthetic_panel() -> pd.DataFrame:
    rows = []
    categories = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    stores = [f"S_{number}" for number in range(1, 11)]
    for category in categories:
        for store in stores:
            values = np.tile(np.arange(1, 8), EXPECTED_DAYS // 7 + 1)[:EXPECTED_DAYS]
            for day, value in enumerate(values, start=1):
                rows.append(
                    {
                        "series_id": f"{category}_{store}",
                        "category": category,
                        "store": store,
                        "state": "X",
                        "day_number": day,
                        "d": f"d_{day}",
                        "date": pd.Timestamp("2011-01-01") + pd.Timedelta(days=day - 1),
                        "actual": float(value),
                    }
                )
    panel = pd.DataFrame(rows)
    assert panel["series_id"].nunique() == EXPECTED_SERIES
    return panel


class BaselinePipelineTests(unittest.TestCase):
    def test_horizon_bands_are_fixed(self) -> None:
        self.assertEqual(horizon_band(1), "D01-D07")
        self.assertEqual(horizon_band(8), "D08-D14")
        self.assertEqual(horizon_band(28), "D22-D28")

    def test_metrics_have_expected_values(self) -> None:
        actual = np.array([1.0, 2.0, 3.0])
        forecast = np.array([1.0, 1.0, 5.0])
        values = metric_values(actual, forecast, scale=2.0)
        self.assertTrue(np.isclose(values["mae"], 1.0))
        self.assertTrue(np.isclose(values["wape"], 0.5))
        self.assertTrue(np.isclose(values["rmse"], np.sqrt(5 / 3)))
        self.assertTrue(np.isclose(values["rmsse"], np.sqrt((5 / 3) / 2)))

    def test_rmsse_scale_ignores_leading_zero_period(self) -> None:
        self.assertTrue(np.isclose(rmsse_scale(np.array([0, 0, 2, 4, 6])), 4.0))

    def test_holt_winters_returns_nonnegative_horizon(self) -> None:
        training = np.tile(np.array([10, 12, 14, 16, 18, 20, 22], dtype=float), 20)
        forecast = _holt_winters_point_forecast(training)
        self.assertEqual(len(forecast), 28)
        self.assertTrue(np.isfinite(forecast).all())
        self.assertTrue((forecast >= 0).all())

    def test_seasonal_naive_does_not_use_test_actuals(self) -> None:
        panel = _synthetic_panel()
        original, _ = seasonal_naive_forecasts(panel)
        mutated = panel.copy()
        mutated.loc[mutated["day_number"] > FOLDS[0].origin_day, "actual"] = 999999.0
        changed, _ = seasonal_naive_forecasts(mutated)
        original_f1 = original.loc[original["fold_id"] == "F1", "forecast"].to_numpy()
        changed_f1 = changed.loc[changed["fold_id"] == "F1", "forecast"].to_numpy()
        self.assertTrue(np.array_equal(original_f1, changed_f1))


if __name__ == "__main__":
    unittest.main()
