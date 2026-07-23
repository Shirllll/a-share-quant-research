from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model_comparison_quant import fit_huber, fit_ols, future_leakage_audit, predict_linear


class ModelComparisonTests(unittest.TestCase):
    def test_ordinary_ols_recovers_linear_relationship(self):
        x = np.arange(20, dtype=float).reshape(-1, 1)
        y = 1.5 + 2.0 * x[:, 0]
        prediction = predict_linear(x, fit_ols(x, y))
        np.testing.assert_allclose(prediction, y, atol=1e-10)

    def test_huber_predictions_are_finite_with_outlier(self):
        x = np.arange(30, dtype=float).reshape(-1, 1)
        y = 0.5 + x[:, 0]
        y[-1] = 10_000
        prediction = predict_linear(x, fit_huber(x, y))
        self.assertTrue(np.isfinite(prediction).all())

    def test_fixed_data_produces_reproducible_ols(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(100, 4))
        y = rng.normal(size=100)
        first = predict_linear(x, fit_ols(x, y))
        second = predict_linear(x, fit_ols(x, y))
        np.testing.assert_array_equal(first, second)

    def test_leakage_audit_requires_missing_future_returns_to_remain_scored(self):
        monthly = pd.DataFrame({
            "signal_month": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "maximum_feature_month": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "scored_stocks": [10, 11], "missing_forward_return_stocks": [1, 2],
        })
        log = pd.DataFrame({
            "refit_month": pd.to_datetime(["2024-01-31"]),
            "validation_end": pd.to_datetime(["2023-12-31"]),
        })
        audit = future_leakage_audit(monthly, log).set_index("check")
        self.assertTrue(bool(audit.loc["scoring_ignores_future_return_availability", "passed"]))


if __name__ == "__main__":
    unittest.main()
