from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from advanced_quant import (
    Standardizer,
    TemporalRetriever,
    _select_blend,
    industry_balanced_diagnostic,
    leakage_audit,
    leg_turnover,
    monthly_rank_ic,
)


class AdvancedModelTests(unittest.TestCase):
    def test_standardizer_is_finite_and_centered(self):
        x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], dtype=np.float32)
        z = Standardizer().fit(x).transform(x)
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue(np.allclose(z.mean(axis=0), 0.0, atol=1e-6))

    def test_monthly_rank_ic_respects_month_groups(self):
        pred = np.array([1, 2, 2, 1], dtype=float)
        target = np.array([1, 2, 2, 1], dtype=float)
        months = np.array(["2024-01", "2024-01", "2024-02", "2024-02"])
        self.assertAlmostEqual(monthly_rank_ic(pred, target, months), 1.0, places=12)

    def test_temporal_retriever_returns_finite_predictions(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(100, 6)).astype(np.float32)
        y = rng.normal(size=100).astype(np.float32)
        months = pd.date_range("2020-01-01", periods=100, freq="D").to_numpy()
        model = TemporalRetriever(max_samples=50, neighbors=8, seed=7).fit(x, y, months)
        prediction = model.predict(x[:5])
        self.assertEqual(prediction.shape, (5,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_nonlinear_gate_falls_back_to_linear_without_material_gain(self):
        target = np.array([0.0, 1.0, 0.0, 1.0])
        months = np.array(["2020-01", "2020-01", "2020-02", "2020-02"])
        linear = target.copy()
        neural = target.copy()
        retrieval = target.copy()
        weights, _, _, enabled = _select_blend([linear, neural, retrieval], target, months, 0.005)
        self.assertFalse(enabled)
        self.assertTrue(np.array_equal(weights, np.array([1.0, 0.0, 0.0])))

    def test_industry_diagnostic_is_exactly_balanced(self):
        frame = pd.DataFrame({
            "industry": ["A"] * 10 + ["B"] * 20,
            "prediction": np.arange(30),
        })
        long_weights, short_weights = industry_balanced_diagnostic(frame)
        self.assertAlmostEqual(long_weights.sum(), 1.0)
        self.assertAlmostEqual(short_weights.sum(), 1.0)
        self.assertAlmostEqual(long_weights[frame.loc[long_weights.index, "industry"].eq("A")].sum(), 0.5)
        self.assertAlmostEqual(short_weights[frame.loc[short_weights.index, "industry"].eq("B")].sum(), 0.5)

    def test_leg_turnover_uses_weights(self):
        self.assertAlmostEqual(leg_turnover({"A": 0.5, "B": 0.5}, {}), 1.0)
        self.assertAlmostEqual(
            leg_turnover({"A": 0.5, "C": 0.5}, {"A": 0.5, "B": 0.5}),
            0.5,
        )

    def test_leakage_audit_requires_realised_training_labels(self):
        valid = pd.DataFrame({
            "refit_month": ["2024-03-31"],
            "prediction_month": ["2024-03-31"],
            "history_start": ["2018-01-31"],
            "history_end": ["2024-02-29"],
            "max_label_realization_month": ["2024-03-31"],
            "future_leakage_pass": [True],
        })
        self.assertTrue(leakage_audit(valid)["future_leakage_pass"].all())
        invalid = valid.copy()
        invalid["max_label_realization_month"] = "2024-04-30"
        with self.assertRaises(RuntimeError):
            leakage_audit(invalid)


if __name__ == "__main__":
    unittest.main()
