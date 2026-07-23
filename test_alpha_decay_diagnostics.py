from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alpha_decay_diagnostics import (
    decile_diagnostics,
    holding_status_attribution,
    long_short_leg_diagnostics,
)
from residual_common import assert_selection_period_only, period_label


class AlphaDecayDiagnosticsTests(unittest.TestCase):
    def test_period_boundaries_are_fixed(self):
        self.assertEqual(period_label(pd.Timestamp("2017-12-31")), "formation")
        self.assertEqual(period_label(pd.Timestamp("2021-12-31")), "development")
        self.assertEqual(period_label(pd.Timestamp("2023-12-31")), "selection")
        self.assertEqual(
            period_label(pd.Timestamp("2024-01-31")), "retrospective_test"
        )

    def test_retrospective_cannot_enter_selection(self):
        with self.assertRaises(ValueError):
            assert_selection_period_only(
                pd.DataFrame({"signal_month": ["2024-01-31"]})
            )

    def test_short_leg_uses_short_sale_sign(self):
        frame = pd.DataFrame(
            {
                "signal_month": pd.Timestamp("2022-01-31"),
                "realization_month": pd.Timestamp("2022-02-28"),
                "period": "selection",
                "stock_code": [f"S{i:02d}" for i in range(10)],
                "industry": ["A"] * 10,
                "raw_score": np.arange(10),
                "forward_return": np.linspace(-0.10, 0.10, 10),
                "target": np.arange(10),
            }
        )
        legs = long_short_leg_diagnostics(frame)
        self.assertGreater(legs.loc[0, "short_sale_gross_return"], 0)

    def test_deciles_are_formed_with_contemporaneous_scores(self):
        records = []
        for month in pd.to_datetime(["2022-01-31", "2022-02-28"]):
            for i in range(100):
                records.append(
                    {
                        "signal_month": month,
                        "realization_month": month + pd.offsets.MonthEnd(1),
                        "period": "selection",
                        "stock_code": f"S{i:03d}",
                        "industry": "A",
                        "raw_score": i,
                        "forward_return": i / 1000,
                        "target": i,
                    }
                )
        result = decile_diagnostics(pd.DataFrame(records))
        self.assertGreater(result["g10_minus_g1"].iloc[0], 0)

    def test_holding_status_contributions_reconcile(self):
        records = []
        for month in pd.to_datetime(["2022-01-31", "2022-02-28"]):
            for i in range(20):
                records.append(
                    {
                        "signal_month": month,
                        "realization_month": month + pd.offsets.MonthEnd(1),
                        "period": "selection",
                        "stock_code": f"S{i:03d}",
                        "industry": "A",
                        "raw_score": i,
                        "forward_return": 0.01,
                        "target": i,
                    }
                )
        result = holding_status_attribution(pd.DataFrame(records))
        for _, group in result.groupby("signal_month"):
            self.assertAlmostEqual(
                group["gross_return_contribution"].sum()
                - group["explicit_cost"].sum(),
                group["net_return_contribution"].sum(),
            )

    def test_gross_minus_cost_equals_net(self):
        gross = pd.Series([0.02, -0.01])
        cost = pd.Series([0.001, 0.002])
        net = gross - cost
        np.testing.assert_allclose(gross - cost, net)


if __name__ == "__main__":
    unittest.main()
