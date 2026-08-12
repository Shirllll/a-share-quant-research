from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from external_data_quant import ExternalConfig, _bounded_growth, point_in_time_features


class ExternalDataQuantTests(unittest.TestCase):
    def test_growth_is_finite_and_bounded(self) -> None:
        result = _bounded_growth(
            pd.Series([1.0, 0.0, -1.0]), pd.Series([2.0, 5.0, 10.0])
        )
        self.assertEqual(result.iloc[0], 1.0)
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertEqual(result.iloc[2], 3.0)

    def test_point_in_time_features_ignore_future_report(self) -> None:
        universe = pd.DataFrame({
            "month": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "Stkcd": ["000001", "000001"],
        })
        reports = pd.DataFrame({
            "Stkcd": ["000001", "000001"],
            "publish_date": pd.to_datetime(["2020-01-15", "2020-02-15"]),
            "org_code": ["a", "b"], "rating_value": [3.0, 4.0],
            "eps_current": [1.0, 1.0], "eps_next": [1.1, 1.2],
        })
        result = point_in_time_features(universe, reports, ExternalConfig())
        self.assertEqual(result.iloc[0]["report_count_90d"], 1)
        self.assertEqual(result.iloc[1]["report_count_90d"], 2)
        self.assertTrue(result["external_leakage_pass"].all())

    def test_revision_uses_only_prior_month_features(self) -> None:
        months = pd.date_range("2020-01-31", periods=4, freq="ME")
        universe = pd.DataFrame({"month": months, "Stkcd": "000001"})
        reports = pd.DataFrame({
            "Stkcd": ["000001", "000001"],
            "publish_date": pd.to_datetime(["2020-01-10", "2020-04-10"]),
            "org_code": ["a", "a"], "rating_value": [3.0, 3.0],
            "eps_current": [1.0, 1.0], "eps_next": [1.1, 1.4],
        })
        result = point_in_time_features(universe, reports, ExternalConfig())
        self.assertTrue(result["eps_growth_revision_3m"].iloc[:3].isna().all())
        self.assertAlmostEqual(result["eps_growth_revision_3m"].iloc[3], 0.15)


if __name__ == "__main__":
    unittest.main()
