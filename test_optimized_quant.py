from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from optimized_quant import (
    ResearchConfig,
    build_portfolio,
    choose_blend,
    select_parameters,
    smooth_scores,
    volatility_control,
)


class OptimizedQuantTests(unittest.TestCase):
    def _signals(self) -> pd.DataFrame:
        rows = []
        months = pd.date_range("2018-01-31", "2023-12-31", freq="ME")
        for month_no, month in enumerate(months):
            for stock_no in range(20):
                rows.append({
                    "month": month,
                    "realization_month": month + pd.offsets.MonthEnd(1),
                    "Stkcd": f"{stock_no:06d}",
                    "industry": "A",
                    "forward_return": (0.01 + month_no / 10000) if stock_no >= 18 else -month_no / 20000,
                    "raw_score": stock_no + month_no / 100,
                })
        result = pd.DataFrame(rows)
        result["ridge_score"] = result["raw_score"]
        result["mlp_score"] = result["raw_score"]
        return result

    def test_smoothing_is_causal(self):
        signals = self._signals()
        before = smooth_scores(signals, 0.5)
        changed = signals.copy()
        last = changed["month"].max()
        changed.loc[changed["month"].eq(last), "raw_score"] += 10_000
        after = smooth_scores(changed, 0.5)
        earlier = before[before["month"].lt(last)]["smoothed_score"].to_numpy()
        comparison = after[after["month"].lt(last)]["smoothed_score"].to_numpy()
        np.testing.assert_array_equal(earlier, comparison)

    def test_buffer_retains_existing_holding(self):
        signals = pd.DataFrame([
            {"month": "2022-01-31", "Stkcd": "A", "forward_return": 0.0, "raw_score": 3.0},
            {"month": "2022-01-31", "Stkcd": "B", "forward_return": 0.0, "raw_score": 2.0},
            {"month": "2022-01-31", "Stkcd": "C", "forward_return": 0.0, "raw_score": 1.0},
            {"month": "2022-02-28", "Stkcd": "A", "forward_return": 0.0, "raw_score": 2.0},
            {"month": "2022-02-28", "Stkcd": "B", "forward_return": 0.0, "raw_score": 3.0},
            {"month": "2022-02-28", "Stkcd": "C", "forward_return": 0.0, "raw_score": 1.0},
        ])
        signals["month"] = pd.to_datetime(signals["month"])
        result = build_portfolio(signals, 1 / 3, 2 / 3, 1.0, 0)
        self.assertEqual(result.iloc[1]["turnover"], 0.0)

    def test_true_weight_turnover(self):
        signals = self._signals().query("month <= '2018-02-28'")
        result = build_portfolio(signals, 0.10, 0.20, 1.0, 0)
        self.assertEqual(result.iloc[0]["turnover"], 1.0)
        self.assertEqual(result.iloc[1]["turnover"], 0.0)

    def test_parameter_selection_has_no_retrospective_columns(self):
        chosen, table = select_parameters(self._signals(), ResearchConfig())
        self.assertIn(chosen["smoothing_weight"], (1.0, 0.75, 0.5))
        self.assertIn(chosen["signal_model"], ("ridge", "mlp", "dynamic_blend"))
        self.assertFalse(any("retrospective" in column for column in table.columns))

    def test_volatility_control_uses_lagged_returns(self):
        result = pd.DataFrame({
            "gross_return": [0.01] * 7 + [0.50],
            "turnover": [0.1] * 8,
            "net_return": [0.01] * 8,
        })
        first = volatility_control(result, 0.15, 0.5, 20)
        changed = result.copy()
        changed.loc[7, "gross_return"] = -0.50
        second = volatility_control(changed, 0.15, 0.5, 20)
        self.assertEqual(first.loc[7, "exposure"], second.loc[7, "exposure"])

    def test_monthly_ic_blend_is_finite(self):
        frame = pd.DataFrame({
            "month": pd.to_datetime(["2022-01-31"] * 40 + ["2022-02-28"] * 40),
            "target": np.tile(np.arange(40), 2),
        })
        ridge = frame["target"].to_numpy(float)
        mlp = -ridge
        weight, score = choose_blend(frame, ridge, mlp, (0.0, 0.25, 0.5))
        self.assertEqual(weight, 0.0)
        self.assertTrue(np.isfinite(score))


if __name__ == "__main__":
    unittest.main()
