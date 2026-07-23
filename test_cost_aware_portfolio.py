from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cost_aware_portfolio import (
    PortfolioConfig,
    build_target_weights,
    estimate_order_cost,
)
from residual_common import true_weight_turnover


def synthetic_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 180
    frame = pd.DataFrame(
        {
            "stock_code": [f"{i:06d}" for i in range(n)],
            "industry": np.repeat([f"I{i}" for i in range(12)], 15),
            "original_1m_score": np.linspace(-1, 1, n),
            "original_1m_alpha": np.linspace(-0.01, 0.02, n),
            "market_beta": rng.normal(1.0, 0.15, n),
            "log_size": rng.normal(10.0, 1.0, n),
            "volatility_exposure": rng.uniform(0.10, 0.50, n),
            "liquidity_exposure": rng.normal(18.0, 1.0, n),
            "amount_clean": rng.uniform(20_000_000, 500_000_000, n),
            "forward_return_1m": rng.normal(0.01, 0.05, n),
        }
    )
    return frame


class CostAwarePortfolioTests(unittest.TestCase):
    def test_weights_sum_to_one_and_are_non_negative(self):
        weights, _ = build_target_weights(
            synthetic_frame(), {}, PortfolioConfig()
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreaterEqual(min(weights.values()), 0.0)

    def test_single_stock_cap(self):
        weights, _ = build_target_weights(
            synthetic_frame(), {}, PortfolioConfig(max_stock_weight=0.02)
        )
        self.assertLessEqual(max(weights.values()), 0.02000001)

    def test_industry_constraint(self):
        frame = synthetic_frame()
        weights, _ = build_target_weights(frame, {}, PortfolioConfig())
        local = frame.set_index("stock_code")
        series = pd.Series(weights)
        industry_weight = series.groupby(local.loc[series.index, "industry"]).sum()
        self.assertLessEqual(industry_weight.max(), 0.200001)

    def test_style_exposure_is_bounded(self):
        frame = synthetic_frame()
        weights, _ = build_target_weights(frame, {}, PortfolioConfig())
        local = frame.set_index("stock_code")
        values = local.loc[list(weights), "market_beta"]
        z = (values - frame["market_beta"].mean()) / frame["market_beta"].std(ddof=0)
        exposure = float((pd.Series(weights) * z).sum())
        self.assertLessEqual(abs(exposure), 0.50)

    def test_higher_turnover_penalty_does_not_increase_turnover(self):
        frame = synthetic_frame()
        previous, _ = build_target_weights(frame, {}, PortfolioConfig())
        changed = frame.copy()
        changed["original_1m_score"] = -changed["original_1m_score"]
        changed["original_1m_alpha"] = -changed["original_1m_alpha"]
        low, _ = build_target_weights(
            changed, previous, PortfolioConfig(lambda_turnover=0.5)
        )
        high, _ = build_target_weights(
            changed, previous, PortfolioConfig(lambda_turnover=2.0)
        )
        self.assertLessEqual(
            true_weight_turnover(high, previous),
            true_weight_turnover(low, previous) + 1e-8,
        )

    def test_low_liquidity_cost_is_higher(self):
        config = PortfolioConfig()
        low = estimate_order_cost(0.01, 1_000_000, 0.30, 50_000_000, config)
        high = estimate_order_cost(
            0.01, 100_000_000, 0.30, 50_000_000, config
        )
        self.assertGreaterEqual(
            low["estimated_total_cost"], high["estimated_total_cost"]
        )

    def test_cash_ratio_is_not_used_to_raise_sharpe(self):
        weights, _ = build_target_weights(
            synthetic_frame(), {}, PortfolioConfig()
        )
        self.assertLessEqual(1.0 - sum(weights.values()), 1e-10)

    def test_adv_participation_formula(self):
        result = estimate_order_cost(
            0.02, 10_000_000, 0.20, 50_000_000, PortfolioConfig()
        )
        self.assertAlmostEqual(result["ADV_participation_5d"], 0.02)

    def test_target_weights_do_not_use_future_return(self):
        frame = synthetic_frame()
        first, _ = build_target_weights(frame, {}, PortfolioConfig())
        frame["forward_return_1m"] = np.arange(len(frame)) * 1000
        second, _ = build_target_weights(frame, {}, PortfolioConfig())
        self.assertEqual(first, second)

    def test_holdings_count_is_within_declared_range(self):
        weights, _ = build_target_weights(
            synthetic_frame(), {}, PortfolioConfig()
        )
        self.assertGreaterEqual(len(weights), 80)
        self.assertLessEqual(len(weights), 200)


if __name__ == "__main__":
    unittest.main()
