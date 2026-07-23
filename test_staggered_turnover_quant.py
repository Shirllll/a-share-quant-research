from __future__ import annotations

import unittest

import pandas as pd

from advanced_quant import OUT
from staggered_turnover_quant import carry_available, combine_sleeves
from staggered_turnover_quant import StaggeredConfig


class StaggeredTurnoverTests(unittest.TestCase):
    def test_equal_sleeve_capital_is_preserved(self):
        combined = combine_sleeves([{"A": 1.0}, {"B": 1.0}])
        self.assertEqual(combined, {"A": 0.5, "B": 0.5})

    def test_overlapping_names_are_aggregated(self):
        combined = combine_sleeves([{"A": 0.5, "B": 0.5}, {"A": 1.0}])
        self.assertAlmostEqual(combined["A"], 0.75)
        self.assertAlmostEqual(combined["B"], 0.25)

    def test_inactive_sleeve_only_drops_unavailable_names(self):
        carried = carry_available({"A": 0.4, "B": 0.6}, {"B", "C"})
        self.assertEqual(carried, {"B": 0.6})

    def test_combination_is_deterministic(self):
        sleeves = [{"A": 0.2, "B": 0.8}, {"C": 1.0}, {"A": 1.0}]
        self.assertEqual(combine_sleeves(sleeves), combine_sleeves(sleeves))

    def test_hybrid_configuration_records_both_caps(self):
        config = StaggeredConfig(
            sleeves=2,
            long_monthly_turnover_cap=0.15,
            diagnostic_leg_monthly_turnover_cap=0.175,
        )
        self.assertEqual(config.sleeves, 2)
        self.assertEqual(config.long_monthly_turnover_cap, 0.15)
        self.assertEqual(config.diagnostic_leg_monthly_turnover_cap, 0.175)

    def test_generated_output_has_no_future_features(self):
        path = OUT / "staggered_turnover_backtest.csv"
        if not path.exists():
            self.skipTest("full staggered run has not been generated")
        output = pd.read_csv(path, parse_dates=["signal_month", "maximum_feature_month"])
        self.assertFalse((output["maximum_feature_month"] > output["signal_month"]).any())

    def test_retrospective_is_not_used_for_parameter_selection(self):
        path = OUT / "staggered_turnover_parameter_selection.csv"
        if not path.exists():
            self.skipTest("full staggered run has not been generated")
        selection = pd.read_csv(path)
        self.assertFalse(selection["uses_retrospective_test"].astype(bool).any())


if __name__ == "__main__":
    unittest.main()
