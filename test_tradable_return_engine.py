from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tradable_return_engine import (
    classify_execution_state,
    execute_target_orders,
    portfolio_return_without_imputation,
)


def state_frame(buyable: bool = True, sellable: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["A", "B"],
            "buyable_first_5d": [buyable, True],
            "sellable_first_5d": [sellable, True],
            "one_price_limit_up_first_5d": [False, False],
            "one_price_limit_down_first_5d": [False, False],
        }
    )


class TradableReturnEngineTests(unittest.TestCase):
    def test_suspended_holding_is_not_falsely_sold(self):
        actual, log = execute_target_orders(
            {"A": 1.0},
            {},
            state_frame(sellable=False),
            pd.Timestamp("2022-01-31"),
            pd.Timestamp("2022-02-28"),
        )
        self.assertEqual(actual, {"A": 1.0})
        self.assertEqual(log[0]["execution_status"], "unfilled_sell_carried")

    def test_suspended_stock_is_not_falsely_bought(self):
        actual, log = execute_target_orders(
            {},
            {"A": 1.0},
            state_frame(buyable=False),
            pd.Timestamp("2022-01-31"),
            pd.Timestamp("2022-02-28"),
        )
        self.assertEqual(actual, {})
        self.assertEqual(log[0]["execution_status"], "unfilled_buy")

    def test_unfilled_order_is_carried(self):
        actual, log = execute_target_orders(
            {"A": 0.5},
            {"B": 1.0},
            state_frame(sellable=False),
            pd.Timestamp("2022-01-31"),
            pd.Timestamp("2022-02-28"),
        )
        self.assertIn("A", actual)
        self.assertGreater(sum(row["unfilled_weight"] for row in log), 0)

    def test_missing_return_is_not_zero_imputed(self):
        value, unresolved = portfolio_return_without_imputation(
            {"A": 0.5, "B": 0.5}, pd.Series({"A": 0.02, "B": np.nan})
        )
        self.assertTrue(np.isnan(value))
        self.assertAlmostEqual(unresolved, 0.5)

    def test_unresolved_weight_is_exact(self):
        _, unresolved = portfolio_return_without_imputation(
            {"A": 0.3, "B": 0.7}, pd.Series({"A": np.nan, "B": 0.01})
        )
        self.assertAlmostEqual(unresolved, 0.3)

    def test_resumed_trading_is_identified(self):
        row = pd.Series({"buyable_first_5d": True, "sellable_first_5d": True})
        self.assertEqual(
            classify_execution_state(row, "A", "A", previous_had_data=False),
            "resumed_trading",
        )

    def test_theoretical_and_tradable_returns_can_differ(self):
        theoretical, _ = portfolio_return_without_imputation(
            {"A": 1.0}, pd.Series({"A": 0.10})
        )
        tradable, _ = portfolio_return_without_imputation(
            {"B": 1.0}, pd.Series({"B": -0.05})
        )
        self.assertNotEqual(theoretical, tradable)

    def test_limit_support_flag_is_not_faked(self):
        _, log = execute_target_orders(
            {},
            {"A": 1.0},
            state_frame(),
            pd.Timestamp("2022-01-31"),
            pd.Timestamp("2022-02-28"),
        )
        self.assertFalse(log[0]["unsupported_limit_execution"])


if __name__ == "__main__":
    unittest.main()
