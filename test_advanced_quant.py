from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from advanced_quant import Standardizer, TemporalRetriever, monthly_rank_ic


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


if __name__ == "__main__":
    unittest.main()
