from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from daily_risk_quant import (
    DailyRiskConfig,
    aggregate_daily_chunk,
    evaluate_candidates,
    build_risk_portfolio,
    finalize_daily_aggregates,
)
from optimized_quant import ResearchConfig


class DailyRiskQuantTest(unittest.TestCase):
    def _daily(self) -> pd.DataFrame:
        dates = pd.date_range("2022-01-03", periods=12, freq="B")
        return pd.DataFrame({
            "Stkcd": ["000001"] * len(dates), "Trddt": dates,
            "Hiprc": 10.2, "Loprc": 9.8, "Clsprc": 10.0,
            "Dnvaltrd": np.linspace(1e7, 2e7, len(dates)),
            "Dretwd": np.linspace(-0.02, 0.02, len(dates)),
            "Trdsta": 1, "LimitStatus": 0,
        })

    def test_daily_features_are_finite_and_causal(self):
        config = DailyRiskConfig(minimum_daily_observations=10)
        result = finalize_daily_aggregates(
            [aggregate_daily_chunk(self._daily(), config)], config
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result["source_max_date"].le(result["month"]).all())
        values = result[["realized_volatility", "downside_volatility", "illiquidity"]]
        self.assertTrue(np.isfinite(values.to_numpy()).all())
        self.assertTrue(values.ge(0).all().all())

    def test_chunk_combination_matches_full_aggregation(self):
        frame = self._daily()
        config = DailyRiskConfig(minimum_daily_observations=10)
        full = finalize_daily_aggregates([aggregate_daily_chunk(frame, config)], config)
        split = finalize_daily_aggregates([
            aggregate_daily_chunk(frame.iloc[:6], config),
            aggregate_daily_chunk(frame.iloc[6:], config),
        ], config)
        self.assertAlmostEqual(
            full.loc[0, "realized_volatility"], split.loc[0, "realized_volatility"], places=12
        )

    def test_buffered_holding_is_not_forced_out_by_risk(self):
        months = pd.to_datetime(["2022-01-31", "2022-02-28"])
        rows = []
        for month in months:
            for index in range(10):
                rows.append({
                    "month": month, "Stkcd": f"{index:06d}",
                    "forward_return": index / 100, "raw_score": index / 10,
                    "risk_composite": 1.0 if index == 9 else 0.0,
                })
        signals = pd.DataFrame(rows)
        # Force the low-risk second-best name into month one, then make the original
        # high-alpha name retainable in month two despite its high risk penalty.
        signals.loc[
            signals["month"].eq(months[0]) & signals["Stkcd"].eq("000009"),
            "risk_composite",
        ] = 0.0
        signals.loc[
            signals["month"].eq(months[0]) & signals["Stkcd"].eq("000008"),
            "raw_score",
        ] = 0.85
        result = build_risk_portfolio(
            signals, "composite", 0.1,
            DailyRiskConfig(top_fraction=0.10, exit_fraction=0.20,
                            smoothing_weight=1.0), 0,
        )
        # The selected holding remains inside the alpha exit buffer in month two;
        # its contemporaneous risk cannot force an otherwise-unnecessary sale.
        self.assertEqual(result.loc[1, "turnover"], 0.0)

    def test_turnover_uses_real_weight_changes(self):
        rows = []
        for month, high in [("2022-01-31", 9), ("2022-02-28", 8)]:
            for index in range(10):
                rows.append({
                    "month": pd.Timestamp(month), "Stkcd": f"{index:06d}",
                    "forward_return": 0.01, "raw_score": 1.0 if index == high else index / 100,
                    "risk_tail": 0.0,
                })
        result = build_risk_portfolio(
            pd.DataFrame(rows), "tail", 0,
            DailyRiskConfig(top_fraction=0.10, exit_fraction=0.10,
                            smoothing_weight=1.0), 0,
        )
        self.assertEqual(result.loc[0, "turnover"], 1.0)
        self.assertEqual(result.loc[1, "turnover"], 1.0)

    def test_deterministic(self):
        rows = []
        for month in pd.to_datetime(["2022-01-31", "2022-02-28"]):
            for index in range(20):
                rows.append({
                    "month": month, "Stkcd": f"{index:06d}",
                    "forward_return": index / 1000, "raw_score": index / 20,
                    "risk_liquidity": (20 - index) / 20,
                })
        frame = pd.DataFrame(rows)
        left = build_risk_portfolio(frame, "liquidity", 0.1, cost_bps=20)
        right = build_risk_portfolio(frame, "liquidity", 0.1, cost_bps=20)
        pd.testing.assert_frame_equal(left, right)

    def test_retrospective_is_absent_from_parameter_selection(self):
        months = pd.date_range("2018-01-31", "2023-12-31", freq="ME")
        rows = []
        risks = []
        for month_number, month in enumerate(months):
            for index in range(20):
                code = f"{index:06d}"
                rows.append({
                    "month": month, "realization_month": month + pd.offsets.MonthEnd(1),
                    "Stkcd": code, "industry": "A", "forward_return":
                    (index - 10) / 1000 + ((month_number % 3) - 1) / 100,
                    "ridge_score": index / 20, "mlp_score": index / 20,
                    "raw_score": index / 20 - 0.5,
                })
                risks.append({
                    "month": month, "Stkcd": code, "risk_tail": index / 20,
                    "risk_liquidity": (20 - index) / 20,
                    "risk_composite": abs(index - 10) / 10,
                })
        _, selection = evaluate_candidates(
            pd.DataFrame(rows), pd.DataFrame(risks),
            DailyRiskConfig(penalty_candidates=(0.05,), risk_model_candidates=("tail",)),
            ResearchConfig(),
        )
        self.assertFalse(any(
            column.startswith("retrospective_") for column in selection.columns
        ))
        self.assertTrue(selection["selection_used_retrospective"].eq(False).all())


if __name__ == "__main__":
    unittest.main()
