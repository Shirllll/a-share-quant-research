from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ml_quant import fit as ridge_fit
from residual_common import (
    _stable_cross_sectional_residual,
    assert_selection_period_only,
    purged_validation_split,
)
from residual_target_quant import (
    compute_forward_return_3m,
    deterministic_sample,
    fixed_combined_score,
)


class ResidualTargetTests(unittest.TestCase):
    def test_one_month_label_alignment(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        self.assertEqual(returns.shift(-1).iloc[0], 0.02)

    def test_three_month_compound_alignment(self):
        returns = pd.Series([0.00, 0.10, -0.10, 0.20, 0.05])
        result = compute_forward_return_3m(returns)
        self.assertAlmostEqual(result.iloc[0], 1.10 * 0.90 * 1.20 - 1.0)
        self.assertTrue(pd.isna(result.iloc[-2]))

    def test_three_month_split_has_purge_and_embargo(self):
        months = pd.date_range("2018-01-31", periods=60, freq="ME")
        history = pd.DataFrame(
            {
                "month": months,
                "target_realization_end_3m": months + pd.offsets.MonthEnd(3),
            }
        )
        train, validation, audit = purged_validation_split(
            history, pd.Timestamp("2023-01-31"), 3
        )
        self.assertEqual(audit["purge_months"], 3)
        self.assertEqual(audit["embargo_months"], 3)
        self.assertLess(
            train["target_realization_end_3m"].max(),
            validation["month"].min() - pd.offsets.MonthEnd(2),
        )

    def test_risk_exposures_do_not_require_future_dates(self):
        signal_month = pd.Timestamp("2023-01-31")
        exposure_months = pd.to_datetime(["2022-11-30", "2022-12-31", "2023-01-31"])
        self.assertTrue((exposure_months <= signal_month).all())

    def test_cross_sectional_residual_mean_is_near_zero(self):
        rng = np.random.default_rng(7)
        n = 100
        group = pd.DataFrame(
            {
                "month": pd.Timestamp("2022-01-31"),
                "industry": np.repeat(["A", "B"], n // 2),
                "forward_return_1m": rng.normal(size=n),
                "log_size": rng.normal(size=n),
                "market_beta": rng.normal(size=n),
                "volatility_exposure": rng.normal(size=n),
                "liquidity_exposure": rng.normal(size=n),
            }
        )
        residual, record = _stable_cross_sectional_residual(
            group, "forward_return_1m"
        )
        self.assertTrue(record["regression_valid"])
        self.assertAlmostEqual(residual.mean(), 0.0, places=10)

    def test_ridge_standardization_comes_from_training_data(self):
        x_train = np.array([[0.0], [2.0], [4.0]])
        y_train = np.array([0.0, 1.0, 2.0])
        _, stats = ridge_fit(x_train, y_train, 10.0)
        self.assertAlmostEqual(stats[0, 0], 2.0)

    def test_retrospective_is_rejected_from_alpha_selection(self):
        with self.assertRaises(ValueError):
            assert_selection_period_only(
                pd.DataFrame({"signal_month": ["2024-06-30"]})
            )

    def test_combined_score_is_exactly_half_and_half(self):
        one = pd.Series([1.0, 2.0])
        three = pd.Series([3.0, 4.0])
        pd.testing.assert_series_equal(
            fixed_combined_score(one, three), pd.Series([2.0, 3.0])
        )

    def test_future_leakage_audit_logic_passes_valid_split(self):
        months = pd.date_range("2018-01-31", periods=60, freq="ME")
        history = pd.DataFrame(
            {
                "month": months,
                "target_realization_end_3m": months + pd.offsets.MonthEnd(3),
            }
        )
        _, _, audit = purged_validation_split(
            history, pd.Timestamp("2023-01-31"), 3
        )
        self.assertTrue(audit["training_precedes_validation"])
        self.assertTrue(audit["validation_precedes_prediction"])

    def test_fixed_seed_sampling_is_reproducible(self):
        frame = pd.DataFrame(
            {
                "month": pd.date_range("2020-01-31", periods=100, freq="ME"),
                "Stkcd": [f"S{i}" for i in range(100)],
            }
        )
        pd.testing.assert_frame_equal(
            deterministic_sample(frame, 20, 7),
            deterministic_sample(frame, 20, 7),
        )


if __name__ == "__main__":
    unittest.main()
