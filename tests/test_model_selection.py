import unittest

import pandas as pd

from src.model_selection import (
    HOLT_WINTERS,
    LIGHTGBM,
    SEASONAL_NAIVE,
    build_paired_robustness_table,
    evaluate_model_selection,
)


class ModelSelectionTests(unittest.TestCase):
    @staticmethod
    def _metrics() -> pd.DataFrame:
        series_values = {
            LIGHTGBM: {"S1": 0.50, "S2": 0.60, "S3": 0.90},
            SEASONAL_NAIVE: {"S1": 0.80, "S2": 0.70, "S3": 0.80},
            HOLT_WINTERS: {"S1": 0.70, "S2": 0.65, "S3": 0.85},
        }
        rows = []
        for model, values in series_values.items():
            rows.append(
                {
                    "scope": "overall",
                    "model": model,
                    "fold_id": None,
                    "series_id": None,
                    "rmsse": sum(values.values()) / len(values),
                }
            )
            for fold_id in ("F1", "F2"):
                fold_adjustment = 0.01 if fold_id == "F2" else 0.0
                rows.append(
                    {
                        "scope": "fold",
                        "model": model,
                        "fold_id": fold_id,
                        "series_id": None,
                        "rmsse": sum(values.values()) / len(values) + fold_adjustment,
                    }
                )
                for series_id, value in values.items():
                    rows.append(
                        {
                            "scope": "series_fold",
                            "model": model,
                            "fold_id": fold_id,
                            "series_id": series_id,
                            "rmsse": value + fold_adjustment,
                        }
                    )
        return pd.DataFrame(rows)

    def test_shared_gate_selects_lightgbm_and_reports_both_baselines(self) -> None:
        decision, sensitivity = evaluate_model_selection(
            self._metrics(), required_series_wins=2
        )
        self.assertEqual(decision.loc[0, "selected_model"], LIGHTGBM)
        self.assertTrue(bool(decision.loc[0, "lightgbm_selected"]))
        self.assertEqual(
            int(decision.loc[0, "lightgbm_series_wins_vs_seasonal_naive"]), 2
        )
        self.assertEqual(
            int(decision.loc[0, "lightgbm_series_wins_vs_holt_winters"]), 2
        )
        self.assertEqual(len(sensitivity), 4)
        self.assertTrue(sensitivity["passed"].all())

    def test_robustness_table_is_deterministic(self) -> None:
        first = build_paired_robustness_table(
            self._metrics(), n_bootstrap=500, random_state=42
        )
        second = build_paired_robustness_table(
            self._metrics(), n_bootstrap=500, random_state=42
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["baseline_model"]), {SEASONAL_NAIVE, HOLT_WINTERS})


if __name__ == "__main__":
    unittest.main()
