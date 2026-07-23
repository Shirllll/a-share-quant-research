from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_cleaning import clean_monthly_panel


class DataCleaningTests(unittest.TestCase):
    def _panel(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Stkcd": ["1", "2"],
            "month": ["2024-01-31", "2024-01-31"],
            "Listdt": ["2020-01-01", "2020-01-01"],
            "Annodt": ["2023-12-01", "2022-01-01"],
            "Accper": ["2023-09-30", "2021-12-31"],
            "ret": [5.0, -2.0],
            "amount": [100.0, -1.0],
            "size": [1000.0, 0.0],
            "volatility": [0.20, 8.0],
            "illiq": [0.01, -0.10],
            "max_ret": [0.10, 0.90],
            "PE1TTM": [10.0, -5.0],
            "PBV1B": [1.0, 100.0],
            "PSTTM": [2.0, 0.0],
            "F050204C": [0.10, 0.20],
            "F050504C": [0.20, 0.30],
            "F053301C": [0.30, 0.40],
            "F052901C": [1.00, 1.20],
            "F011201A": [0.50, 0.60],
        })

    def test_cleaning_clips_returns_and_invalidates_bad_market_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.csv"
            self._panel().to_csv(path, index=False)
            cleaned = clean_monthly_panel(path)
        self.assertEqual(cleaned["Stkcd"].tolist(), ["000001", "000002"])
        self.assertEqual(cleaned["ret_clean"].tolist(), [3.0, -0.95])
        self.assertTrue(cleaned["flag_return_outlier"].all())
        self.assertTrue(cleaned.loc[1, "flag_market_data_invalid"])
        self.assertTrue(np.isnan(cleaned.loc[1, "PE1TTM_clean"]))

    def test_stale_financials_are_not_forward_filled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.csv"
            self._panel().to_csv(path, index=False)
            cleaned = clean_monthly_panel(path)
        self.assertFalse(cleaned.loc[0, "flag_financial_stale"])
        self.assertTrue(cleaned.loc[1, "flag_financial_stale"])
        self.assertTrue(np.isnan(cleaned.loc[1, "F050504C_clean"]))


if __name__ == "__main__":
    unittest.main()
