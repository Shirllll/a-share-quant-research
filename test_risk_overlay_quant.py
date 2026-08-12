from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from risk_overlay_quant import OverlayConfig, apply_overlay, causal_beta


class RiskOverlayTests(unittest.TestCase):
    def test_beta_is_causal(self):
        portfolio = pd.Series(np.linspace(-0.04, 0.05, 15))
        index = pd.Series(np.linspace(-0.03, 0.04, 15))
        first = causal_beta(portfolio, index, 12, 12, 0.0, 1.5)
        changed = portfolio.copy()
        changed.iloc[-1] = 10.0
        second = causal_beta(changed, index, 12, 12, 0.0, 1.5)
        self.assertEqual(first.iloc[-1], second.iloc[-1])

    def test_hedge_turnover_uses_notional_change(self):
        frame = pd.DataFrame({
            "month": pd.date_range("2020-01-31", periods=15, freq="ME"),
            "gross_return": np.linspace(-0.04, 0.05, 15),
            "net_return": np.linspace(-0.041, 0.049, 15),
            "hedge_index_return": np.linspace(-0.03, 0.04, 15),
        })
        result = apply_overlay(frame, OverlayConfig())
        expected = result["hedge_notional"].diff().abs()
        expected.iloc[0] = abs(result.iloc[0]["hedge_notional"])
        np.testing.assert_allclose(result["hedge_turnover"], expected)

    def test_costs_are_finite_and_nonnegative(self):
        frame = pd.DataFrame({
            "month": pd.date_range("2020-01-31", periods=15, freq="ME"),
            "gross_return": np.linspace(-0.04, 0.05, 15),
            "net_return": np.linspace(-0.041, 0.049, 15),
            "hedge_index_return": np.linspace(-0.03, 0.04, 15),
        })
        result = apply_overlay(frame, OverlayConfig())
        self.assertTrue(np.isfinite(result[["hedge_trade_cost", "hedge_roll_cost"]]).all().all())
        self.assertTrue(result[["hedge_trade_cost", "hedge_roll_cost"]].ge(0).all().all())


if __name__ == "__main__":
    unittest.main()
