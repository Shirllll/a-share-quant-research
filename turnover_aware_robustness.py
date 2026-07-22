from __future__ import annotations

"""Cost, subperiod, and same-convention comparisons for turnover-aware Ridge."""

import numpy as np
import pandas as pd

from advanced_quant import OUT
from ml_quant import metric
from turnover_aware_quant import FIXED_COST_GRID_BPS


def _strategy_frames(turnover_aware: pd.DataFrame, advanced: pd.DataFrame):
    advanced = advanced.copy()
    advanced["signal_month"] = advanced["month"] - pd.offsets.MonthEnd(1)
    advanced["period"] = np.select(
        [advanced["signal_month"] < "2022-01-01", advanced["signal_month"] < "2024-01-01"],
        ["development", "selection"], default="retrospective_test",
    )
    return {
        "turnover_aware_long_only": (
            turnover_aware, "gross_return", "turnover",
        ),
        "turnover_aware_industry_neutral_diagnostic": (
            turnover_aware, "long_short_gross_return", "long_short_turnover",
        ),
        "current_advanced_long_only": (
            advanced, "gross_return", "turnover",
        ),
        "current_advanced_industry_neutral_diagnostic": (
            advanced, "long_short_gross_return", "long_short_turnover",
        ),
    }


def cost_stress(turnover_aware: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (frame, return_column, turnover_column) in _strategy_frames(turnover_aware, advanced).items():
        for cost_bps in FIXED_COST_GRID_BPS:
            returns = frame[return_column] - frame[turnover_column] * cost_bps / 10_000
            rows.append({
                "series": name, "cost_bps": cost_bps, **metric(returns),
                "avg_turnover": frame[turnover_column].mean(),
            })
    return pd.DataFrame(rows)


def subperiod(turnover_aware: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (frame, return_column, turnover_column) in _strategy_frames(turnover_aware, advanced).items():
        for period in ("development", "selection", "retrospective_test"):
            sample = frame[frame["period"] == period]
            for cost_bps in (20, 50, 100):
                returns = sample[return_column] - sample[turnover_column] * cost_bps / 10_000
                rows.append({
                    "period": period, "series": name, "cost_bps": cost_bps,
                    **metric(returns), "avg_turnover": sample[turnover_column].mean(),
                })
    return pd.DataFrame(rows)


def _ic_statistics(backtest: pd.DataFrame, column: str) -> tuple[float, float, float]:
    ic = backtest[column].dropna()
    return float(ic.mean()), float(ic.mean() / ic.std() * np.sqrt(12)), float((ic > 0).mean())


def _monotonicity(deciles: pd.DataFrame) -> float:
    average = deciles.groupby("decile")["return"].mean()
    return float(average.index.to_series().corr(average, method="spearman"))


def _t_stat(values: pd.Series) -> float:
    values = values.dropna()
    return float(values.mean() / (values.std() / np.sqrt(len(values))))


def comparison_diagnostics(turnover_aware: pd.DataFrame, advanced: pd.DataFrame,
                           turnover_deciles: pd.DataFrame, advanced_deciles: pd.DataFrame) -> pd.DataFrame:
    current_ic = _ic_statistics(advanced, "cross_sectional_ic")
    aware_ic = _ic_statistics(turnover_aware, "rank_ic")
    rows = [
        {"metric": "mean_cross_sectional_ic", "current_advanced": current_ic[0], "turnover_aware": aware_ic[0]},
        {"metric": "cross_sectional_icir_annualized", "current_advanced": current_ic[1], "turnover_aware": aware_ic[1]},
        {"metric": "positive_ic_month_share", "current_advanced": current_ic[2], "turnover_aware": aware_ic[2]},
        {"metric": "decile_monotonicity_spearman", "current_advanced": _monotonicity(advanced_deciles),
         "turnover_aware": _monotonicity(turnover_deciles)},
        {"metric": "long_short_monthly_t_stat_20bps",
         "current_advanced": _t_stat(advanced["long_short_net_return"]),
         "turnover_aware": _t_stat(turnover_aware["long_short_net_return_20bps"])},
        {"metric": "long_only_avg_monthly_turnover", "current_advanced": advanced["turnover"].mean(),
         "turnover_aware": turnover_aware["turnover"].mean()},
        {"metric": "long_short_avg_monthly_turnover", "current_advanced": advanced["long_short_turnover"].mean(),
         "turnover_aware": turnover_aware["long_short_turnover"].mean()},
    ]
    for cost_bps in (20, 50, 100):
        for label, current_return, current_turnover, aware_return, aware_turnover in (
            ("long_only", "gross_return", "turnover", "gross_return", "turnover"),
            ("long_short", "long_short_gross_return", "long_short_turnover",
             "long_short_gross_return", "long_short_turnover"),
        ):
            current = metric(advanced[current_return] - advanced[current_turnover] * cost_bps / 10_000)
            aware = metric(turnover_aware[aware_return] - turnover_aware[aware_turnover] * cost_bps / 10_000)
            rows.extend([
                {"metric": f"{label}_sharpe_{cost_bps}bps", "current_advanced": current["sharpe_rf0"],
                 "turnover_aware": aware["sharpe_rf0"]},
                {"metric": f"{label}_annual_return_{cost_bps}bps", "current_advanced": current["annual_return"],
                 "turnover_aware": aware["annual_return"]},
                {"metric": f"{label}_max_drawdown_{cost_bps}bps", "current_advanced": current["max_drawdown"],
                 "turnover_aware": aware["max_drawdown"]},
            ])
    result = pd.DataFrame(rows)
    result["change"] = result["turnover_aware"] - result["current_advanced"]
    return result


def main() -> None:
    turnover_aware = pd.read_csv(OUT / "turnover_aware_backtest.csv", parse_dates=["signal_month", "month"])
    advanced = pd.read_csv(OUT / "advanced_backtest.csv", parse_dates=["month"])
    turnover_deciles = pd.read_csv(OUT / "turnover_aware_decile_returns.csv", parse_dates=["signal_month"])
    advanced_deciles = pd.read_csv(OUT / "decile_returns.csv", parse_dates=["month"])
    costs = cost_stress(turnover_aware, advanced)
    periods = subperiod(turnover_aware, advanced)
    comparison = comparison_diagnostics(turnover_aware, advanced, turnover_deciles, advanced_deciles)
    costs.to_csv(OUT / "turnover_aware_cost_stress.csv", index=False)
    periods.to_csv(OUT / "turnover_aware_subperiod.csv", index=False)
    comparison.to_csv(OUT / "turnover_aware_alpha_diagnostics.csv", index=False)
    print(costs.to_string(index=False))
    print(periods.to_string(index=False))


if __name__ == "__main__":
    main()
