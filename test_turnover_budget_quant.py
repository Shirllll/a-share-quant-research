from __future__ import annotations

import unittest

import pandas as pd

from turnover_budget_quant import apply_turnover_budget


class TurnoverBudgetTests(unittest.TestCase):
    def test_partial_rebalance_respects_budget(self):
        previous = {"A": 0.5, "B": 0.5}
        desired = {"C": 0.5, "D": 0.5}
        current, turnover, forced = apply_turnover_budget(
            previous, desired, {"A", "B", "C", "D"}, 0.20
        )
        self.assertAlmostEqual(turnover, 0.20)
        self.assertEqual(forced, 0.0)
        self.assertAlmostEqual(sum(current.values()), 1.0)

    def test_forced_exit_is_reported_without_future_data(self):
        previous = {"A": 0.5, "B": 0.5}
        desired = {"B": 0.5, "C": 0.5}
        current, turnover, forced = apply_turnover_budget(
            previous, desired, {"B", "C"}, 0.10
        )
        self.assertNotIn("A", current)
        self.assertAlmostEqual(forced, 0.25)
        self.assertAlmostEqual(turnover, 0.25)

    def test_initial_formation_is_not_artificially_capped(self):
        desired = {"A": 0.5, "B": 0.5}
        current, turnover, forced = apply_turnover_budget({}, desired, set(desired), 0.10)
        self.assertEqual(current, desired)
        self.assertEqual(turnover, 1.0)
        self.assertEqual(forced, 0.0)

    def test_same_target_has_zero_turnover(self):
        weights = {"A": 0.4, "B": 0.6}
        current, turnover, forced = apply_turnover_budget(weights, weights, set(weights), 0.10)
        self.assertEqual(current, weights)
        self.assertEqual(turnover, 0.0)
        self.assertEqual(forced, 0.0)

    def test_budget_function_is_deterministic(self):
        previous = {"A": 0.4, "B": 0.6}
        desired = {"A": 0.1, "C": 0.9}
        first = apply_turnover_budget(previous, desired, {"A", "B", "C"}, 0.15)
        second = apply_turnover_budget(previous, desired, {"A", "B", "C"}, 0.15)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
