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
    # The stable advanced diagnostic has two months with a missing constituent
    # return.  Use the intersection of non-missing months for every paired
    # return/turnover comparison instead of silently comparing 95 to 93 months.
    common = advanced[[
        "month", "gross_return", "turnover", "long_short_gross_return", "long_short_turnover",
    ]].merge(
        turnover_aware[[
            "month", "gross_return", "turnover", "long_short_gross_return", "long_short_turnover",
        ]],
        on="month", how="inner", suffixes=("_current", "_aware"),
    )
    common_long = common.dropna(subset=["gross_return_current", "gross_return_aware"])
    common_long_short = common.dropna(subset=[
        "long_short_gross_return_current", "long_short_gross_return_aware",
    ])
    current_ic = _ic_statistics(advanced, "cross_sectional_ic")
    aware_ic = _ic_statistics(turnover_aware, "rank_ic")
    rows = [
        {"metric": "mean_cross_sectional_ic", "current_advanced": current_ic[0], "turnover_aware": aware_ic[0]},
        {"metric": "cross_sectional_icir_annualized", "current_advanced": current_ic[1], "turnover_aware": aware_ic[1]},
        {"metric": "positive_ic_month_share", "current_advanced": current_ic[2], "turnover_aware": aware_ic[2]},
        {"metric": "decile_monotonicity_spearman", "current_advanced": _monotonicity(advanced_deciles),
         "turnover_aware": _monotonicity(turnover_deciles)},
        {"metric": "long_short_monthly_t_stat_20bps",
         "current_advanced": _t_stat(
             common_long_short["long_short_gross_return_current"]
             - common_long_short["long_short_turnover_current"] * 20 / 10_000
         ),
         "turnover_aware": _t_stat(
             common_long_short["long_short_gross_return_aware"]
             - common_long_short["long_short_turnover_aware"] * 20 / 10_000
         )},
        {"metric": "long_only_avg_monthly_turnover", "current_advanced": common_long["turnover_current"].mean(),
         "turnover_aware": common_long["turnover_aware"].mean()},
        {"metric": "long_short_avg_monthly_turnover",
         "current_advanced": common_long_short["long_short_turnover_current"].mean(),
         "turnover_aware": common_long_short["long_short_turnover_aware"].mean()},
        {"metric": "paired_long_only_months", "current_advanced": len(common_long),
         "turnover_aware": len(common_long)},
        {"metric": "paired_long_short_months", "current_advanced": len(common_long_short),
         "turnover_aware": len(common_long_short)},
    ]
    for cost_bps in (20, 50, 100):
        for label, current_return, current_turnover, aware_return, aware_turnover in (
            ("long_only", "gross_return", "turnover", "gross_return", "turnover"),
            ("long_short", "long_short_gross_return", "long_short_turnover",
             "long_short_gross_return", "long_short_turnover"),
        ):
            paired = common_long if label == "long_only" else common_long_short
            current = metric(
                paired[f"{current_return}_current"]
                - paired[f"{current_turnover}_current"] * cost_bps / 10_000
            )
            aware = metric(
                paired[f"{aware_return}_aware"]
                - paired[f"{aware_turnover}_aware"] * cost_bps / 10_000
            )
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
