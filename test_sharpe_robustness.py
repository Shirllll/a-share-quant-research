from __future__ import annotations

import unittest

import pandas as pd

from sharpe_robustness import CandidateRule, accept_candidate


class SharpeRobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = pd.Series({
            "development_sharpe_50bp": 0.55,
            "selection_sharpe_50bp": 0.24,
            "average_pre_retro_sharpe_50bp": 0.395,
            "pre_retro_turnover": 0.24,
        })

    def test_high_turnover_candidate_is_rejected(self) -> None:
        candidate = pd.Series({
            "development_sharpe_50bp": 0.56,
            "selection_sharpe_50bp": 0.30,
            "average_pre_retro_sharpe_50bp": 0.43,
            "pre_retro_turnover": 0.40,
        })
        accepted, reason = accept_candidate(candidate, self.baseline)
        self.assertFalse(accepted)
        self.assertIn("turnover", reason)

    def test_balanced_improvement_is_accepted(self) -> None:
        candidate = pd.Series({
            "development_sharpe_50bp": 0.54,
            "selection_sharpe_50bp": 0.29,
            "average_pre_retro_sharpe_50bp": 0.415,
            "pre_retro_turnover": 0.245,
        })
        accepted, reason = accept_candidate(
            candidate, self.baseline, CandidateRule()
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")

    def test_gate_has_no_retrospective_dependency(self) -> None:
        candidate = pd.Series({
            "development_sharpe_50bp": 0.54,
            "selection_sharpe_50bp": 0.29,
            "average_pre_retro_sharpe_50bp": 0.415,
            "pre_retro_turnover": 0.245,
            "retrospective_sharpe_50bp": -10.0,
        })
        first = accept_candidate(candidate, self.baseline)
        candidate["retrospective_sharpe_50bp"] = 10.0
        second = accept_candidate(candidate, self.baseline)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
