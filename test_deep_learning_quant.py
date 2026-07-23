from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from deep_learning_quant import MLP, cost_stress, data_quality_report, metric, rank_corr


class DeepLearningQuantTests(unittest.TestCase):
    def test_rank_corr_is_one_for_equal_rankings(self):
        values = np.array([3.0, 1.0, 2.0])
        self.assertAlmostEqual(rank_corr(values, values), 1.0)

    def test_mlp_is_reproducible_with_fixed_seed(self):
        x = np.arange(40, dtype=np.float32).reshape(10, 4) / 40
        first = MLP(4, seed=7).predict(x)
        second = MLP(4, seed=7).predict(x)
        self.assertTrue(np.array_equal(first, second))

    def test_cost_stress_uses_turnover_and_is_ordered(self):
        result = pd.DataFrame({
            "gross_return": [0.01, 0.02, -0.01],
            "turnover": [1.0, 0.5, 0.25],
        })
        stress = cost_stress(result)
        self.assertEqual(stress["cost_bps"].tolist(), [0, 20, 50, 100, 150, 200])
        self.assertTrue(stress["annual_return"].is_monotonic_decreasing)

    def test_metric_has_finite_output(self):
        output = metric(pd.Series([0.01, -0.02, 0.03]))
        self.assertTrue(all(np.isfinite(value) for value in output.values()))

    def test_data_quality_report_counts_missing_flags(self):
        panel = pd.DataFrame({"eligible": [True, False]})
        for feature in [
            "momentum", "value_pe", "value_pb", "value_ps", "low_volatility",
            "small_size", "reversal", "liquidity", "roe", "roa", "gross_margin",
            "cash_quality", "low_leverage",
        ]:
            panel[f"{feature}_missing"] = [0.0, 1.0]
        report = data_quality_report(panel)
        eligible = report.loc[report["rule"].eq("eligible_rows")].iloc[0]
        self.assertEqual(eligible["affected_rows"], 1)
        self.assertAlmostEqual(eligible["affected_share"], 0.5)


if __name__ == "__main__":
    unittest.main()
