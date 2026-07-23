from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from advanced_quant import OUT
from turnover_aware_quant import (
    Config,
    build_research_lock,
    deterministic_sample,
    estimate_order_cost,
    parameter_selection,
    scoring_universe,
    select_buffered_names,
    smooth_scores,
    true_weight_turnover,
    validate_no_future_features,
)


def selection_frame(scores: list[float], alphas: list[float] | None = None) -> pd.DataFrame:
    count = len(scores)
    if alphas is None:
        alphas = scores
    return pd.DataFrame({
        "raw_score": scores,
        "smoothed_score": scores,
        "predicted_alpha": alphas,
        "amount_clean": [100_000_000.0] * count,
        "volatility_clean": [0.20] * count,
        "forward_return": np.linspace(-0.02, 0.02, count),
    }, index=[f"S{i:02d}" for i in range(count)])


class TurnoverAwareTests(unittest.TestCase):
    def test_signal_smoothing_uses_current_and_previous_month_only(self):
        raw = pd.Series({"A": 1.0, "B": 2.0})
        result = smooth_scores(raw, {"A": 0.0, "C": 99.0}, 0.75)
        self.assertAlmostEqual(result["A"], 0.75)
        self.assertAlmostEqual(result["B"], 2.0)

    def test_buffered_holding_is_not_sold(self):
        frame = selection_frame(list(np.arange(10, dtype=float)))
        previous = {"S08": 1.0}
        names, _, attempts = select_buffered_names(
            frame, previous, "long", 1, 1.0, 10_000_000, Config(exit_fraction=0.20),
            "long_only", True,
        )
        self.assertEqual(names, {"S08"})
        self.assertEqual(attempts, [])

    def test_trade_is_rejected_when_alpha_gain_is_below_hurdle(self):
        scores = list(np.arange(10, dtype=float))
        alphas = [0.01] * 10
        frame = selection_frame(scores, alphas)
        previous = {"S00": 1.0}
        names, _, attempts = select_buffered_names(
            frame, previous, "long", 1, 1.0, 10_000_000,
            Config(exit_fraction=0.20, hurdle_multiple=2.0), "long_only", True,
        )
        self.assertEqual(names, {"S00"})
        self.assertTrue(any(row["rejection_reason"] == "alpha_gain_below_hurdle" for row in attempts))

    def test_cost_proxy_is_finite_and_non_negative(self):
        for adv in (0.0, np.nan, 100_000_000.0):
            cost = estimate_order_cost(1_000_000, adv, 0.30, Config())
            for key in ("estimated_base_cost", "estimated_impact_cost", "estimated_total_cost", "ADV_participation_5d"):
                self.assertTrue(np.isfinite(cost[key]))
                self.assertGreaterEqual(cost[key], 0.0)

    def test_adv_participation_is_order_over_five_day_adv(self):
        cost = estimate_order_cost(1_000_000, 10_000_000, 0.20, Config(execution_days=5))
        self.assertAlmostEqual(cost["ADV_participation_5d"], 0.02)

    def test_retrospective_rows_never_enter_parameter_selection(self):
        months = pd.date_range("2021-10-31", periods=9, freq="ME")
        records = []
        for month in months:
            period = "development" if month < pd.Timestamp("2022-01-01") else "selection"
            if month >= pd.Timestamp("2022-07-01"):
                period = "retrospective_test"
            for i in range(4):
                records.append({
                    "signal_month": month, "period": period, "raw_score": float(i),
                    "target": float(i),
                })
        scores = pd.DataFrame(records)
        bt_months = pd.date_range("2021-10-31", periods=6, freq="ME")
        fake_backtest = pd.DataFrame({
            "signal_month": bt_months,
            "period": ["development"] * 3 + ["selection"] * 3,
            "gross_return": [0.01, 0.02, 0.00, 0.02, 0.01, 0.03],
            "turnover": [0.2] * 6,
            "long_short_gross_return": [0.02, 0.01, 0.03, 0.03, 0.01, 0.02],
            "long_short_turnover": [0.5] * 6,
            "rank_ic": [0.10] * 6,
        })
        fake_deciles = pd.DataFrame([
            {"period": period, "decile": decile, "return": decile / 1000}
            for period in ("development", "selection") for decile in range(1, 11)
        ])
        with tempfile.TemporaryDirectory() as directory:
            advanced = Path(directory) / "advanced.csv"
            pd.DataFrame({
                "month": pd.date_range("2022-02-28", periods=12, freq="ME"),
                "long_short_turnover": [1.0] * 12,
            }).to_csv(advanced, index=False)

            def fake_simulation(panel, config, collect_trade_log=False):
                self.assertNotIn("retrospective_test", set(panel["period"]))
                return fake_backtest.copy(), fake_deciles.copy(), pd.DataFrame()

            with patch("turnover_aware_quant.simulate_portfolios", side_effect=fake_simulation):
                _, table = parameter_selection(scores, Config(), advanced)
        self.assertFalse(table["uses_retrospective_test"].any())

    def test_research_lock_matches_actual_config(self):
        config = replace(Config(), smoothing_weight=0.50, exit_fraction=0.20, hurdle_multiple=2.0)
        with tempfile.TemporaryDirectory() as directory:
            lock = build_research_lock(
                config, {"passes_constraints": True, "robust_neighbor_share": 0.75}, Path(directory),
            )
        self.assertEqual(lock["frozen_config"], asdict(config))
        self.assertEqual(lock["selected_parameters"]["hurdle_multiple"], 2.0)
        self.assertFalse(lock["retrospective_used_for_parameter_selection"])

    def test_generated_research_lock_is_internally_consistent(self):
        path = OUT / "research_lock.json"
        if not path.exists():
            self.skipTest("research lock is generated by the required full run")
        lock = json.loads(path.read_text(encoding="utf-8"))
        config = Config(**lock["frozen_config"])
        self.assertEqual(lock["selected_parameters"]["smoothing_weight"], config.smoothing_weight)
        self.assertEqual(lock["selected_parameters"]["exit_fraction"], config.exit_fraction)
        self.assertEqual(lock["selected_parameters"]["hurdle_multiple"], config.hurdle_multiple)

    def test_long_and_short_turnover_use_true_weights(self):
        previous_long = {"A": 0.5, "B": 0.5}
        current_long = {"A": 0.5, "C": 0.5}
        previous_short = {"X": 0.6, "Y": 0.4}
        current_short = {"X": 0.2, "Y": 0.4, "Z": 0.4}
        self.assertAlmostEqual(true_weight_turnover(current_long, previous_long), 0.5)
        self.assertAlmostEqual(true_weight_turnover(current_short, previous_short), 0.4)

    def test_future_month_features_are_rejected(self):
        valid = pd.DataFrame({
            "signal_month": ["2024-01-31"], "maximum_feature_month": ["2024-01-31"],
        })
        validate_no_future_features(valid)
        invalid = pd.DataFrame({
            "signal_month": ["2024-01-31"], "maximum_feature_month": ["2024-02-29"],
        })
        with self.assertRaises(ValueError):
            validate_no_future_features(invalid)

    def test_scoring_universe_does_not_filter_on_future_return(self):
        panel = pd.DataFrame({
            "month": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "eligible": [True, True], "forward_return": [0.01, np.nan],
        })
        scored = scoring_universe(panel, pd.Timestamp("2024-01-31"))
        self.assertEqual(len(scored), 2)

    def test_deterministic_sample_is_reproducible(self):
        frame = pd.DataFrame({"month": pd.date_range("2020-01-31", periods=100, freq="ME"), "x": range(100)})
        first = deterministic_sample(frame, 20, 7)
        second = deterministic_sample(frame, 20, 7)
        pd.testing.assert_frame_equal(first, second)

    def test_adv_limit_rejects_oversized_entry(self):
        frame = selection_frame(list(np.arange(10, dtype=float)))
        frame.loc["S09", "amount_clean"] = 1_000.0
        names, _, attempts = select_buffered_names(
            frame, {}, "long", 1, 1.0, 100_000_000,
            Config(max_adv_participation_5d=0.10), "long_only", True,
        )
        self.assertNotIn("S09", names)
        self.assertTrue(any(row["rejection_reason"] == "adv_limit" for row in attempts))


if __name__ == "__main__":
    unittest.main()
