from __future__ import annotations

"""Robustness tests for the frozen residual/cost-aware research candidate."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from advanced_quant import OUT
from cost_aware_portfolio import PortfolioConfig, simulate_portfolio
from residual_common import metric


def cost_and_subperiod(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in ("all", "development", "selection", "retrospective_test"):
        sample = backtest if period == "all" else backtest[backtest["period"] == period]
        for cost_bps in (0, 20, 50, 100, 150, 200):
            rows.append(
                {
                    "test": "cost_and_subperiod",
                    "period": period,
                    "cost_bps": cost_bps,
                    **metric(
                        sample["gross_return"]
                        - sample["turnover"] * cost_bps / 10_000
                    ),
                    "average_turnover": sample["turnover"].mean(),
                    "average_invested_weight": sample["invested_weight"].mean(),
                }
            )
    return pd.DataFrame(rows)


def signal_delay(
    holdings: pd.DataFrame, state: pd.DataFrame
) -> pd.DataFrame:
    state = state.copy()
    state["stock_code"] = state["stock_code"].astype(str).str.zfill(6)
    merged = holdings.merge(
        state[
            [
                "stock_code",
                "realization_month",
                "delayed_1d_return",
                "delayed_5d_return",
            ]
        ],
        on=["stock_code", "realization_month"],
        how="left",
    )
    rows = []
    for label, column, days in (
        ("delay_1_trading_day", "delayed_1d_return", 1),
        ("delay_5_trading_days", "delayed_5d_return", 5),
    ):
        monthly = []
        for realization_month, group in merged.groupby("realization_month", sort=True):
            unresolved = float(
                group.loc[group[column].isna(), "target_weight"].sum()
            )
            value = float((group["target_weight"] * group[column]).sum())
            monthly.append(
                {
                    "realization_month": realization_month,
                    "return": np.nan if unresolved > 1e-12 else value,
                    "unresolved_weight": unresolved,
                }
            )
        monthly_frame = pd.DataFrame(monthly)
        rows.append(
            {
                "test": label,
                "delay_trading_days": days,
                "calculation": "actual daily returns after the stated market trading-day delay",
                **metric(monthly_frame["return"]),
                "maximum_unresolved_weight": monthly_frame["unresolved_weight"].max(),
            }
        )
    return pd.DataFrame(rows)


def parameter_neighbors(
    predictions: pd.DataFrame, config: PortfolioConfig
) -> pd.DataFrame:
    candidates = []
    for lambda_risk in (0.5, 1.0, 2.0):
        if abs(lambda_risk - config.lambda_risk) <= 1.0:
            candidates.append(replace(config, lambda_risk=lambda_risk))
    for lambda_turnover in (0.5, 1.0, 2.0):
        if abs(lambda_turnover - config.lambda_turnover) <= 1.0:
            candidates.append(replace(config, lambda_turnover=lambda_turnover))
    candidates.extend(
        [
            replace(config, max_stock_weight=0.018),
            replace(config, max_stock_weight=0.022),
            replace(config, industry_absolute_cap=0.18),
            replace(config, industry_absolute_cap=0.22),
        ]
    )
    rows = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate.lambda_risk,
            candidate.lambda_turnover,
            candidate.max_stock_weight,
            candidate.industry_absolute_cap,
        )
        if key in seen:
            continue
        seen.add(key)
        backtest, _, _ = simulate_portfolio(predictions, candidate)
        for period in ("development", "selection", "retrospective_test"):
            sample = backtest[backtest["period"] == period]
            rows.append(
                {
                    "lambda_risk": candidate.lambda_risk,
                    "lambda_turnover": candidate.lambda_turnover,
                    "max_stock_weight": candidate.max_stock_weight,
                    "industry_absolute_cap": candidate.industry_absolute_cap,
                    "period": period,
                    **metric(
                        sample["gross_return"]
                        - sample["turnover"] * 50 / 10_000
                    ),
                    "average_turnover": sample["turnover"].mean(),
                    "uses_retrospective_test_for_selection": False,
                }
            )
    return pd.DataFrame(rows)


def extreme_month_tests(backtest: pd.DataFrame) -> pd.DataFrame:
    base = backtest["gross_return"] - backtest["turnover"] * 50 / 10_000
    rows = []
    scenarios = {
        "all_months": base.index,
        "drop_best_1": base.nsmallest(max(len(base) - 1, 0)).index,
        "drop_best_3": base.nsmallest(max(len(base) - 3, 0)).index,
        "drop_worst_1": base.nlargest(max(len(base) - 1, 0)).index,
        "drop_worst_3": base.nlargest(max(len(base) - 3, 0)).index,
    }
    for scenario, indices in scenarios.items():
        rows.append({"scenario": scenario, **metric(base.loc[indices])})
    return pd.DataFrame(rows)


def industry_exclusion(holdings: pd.DataFrame) -> pd.DataFrame:
    industries = sorted(holdings["industry"].dropna().astype(str).unique())
    rows = []
    for excluded in industries:
        monthly = []
        for _, group in holdings[holdings["industry"].astype(str) != excluded].groupby(
            "realization_month", sort=True
        ):
            weight = group["target_weight"]
            if weight.sum() <= 0 or group["forward_return"].isna().any():
                monthly.append(np.nan)
            else:
                monthly.append(float((weight / weight.sum() * group["forward_return"]).sum()))
        rows.append(
            {
                "excluded_industry": excluded,
                **metric(pd.Series(monthly, dtype=float)),
                "calculation": "remaining long holdings renormalized monthly; 50bp turnover not reoptimized",
            }
        )
    return pd.DataFrame(rows)


def block_bootstrap(
    values: pd.Series,
    simulations: int = 5_000,
    block_months: int = 6,
    seed: int = 20260723,
) -> pd.DataFrame:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    starts = np.arange(max(1, len(values) - block_months + 1))
    blocks_needed = int(np.ceil(len(values) / block_months))
    rng = np.random.default_rng(seed)
    annual = np.empty(simulations)
    for iteration in range(simulations):
        chosen = rng.choice(starts, blocks_needed, replace=True)
        sample = np.concatenate(
            [values[start : start + block_months] for start in chosen]
        )[: len(values)]
        annual[iteration] = sample.mean() * 12.0
    low, median, high = np.quantile(annual, [0.025, 0.50, 0.975])
    return pd.DataFrame(
        [
            {
                "block_months": block_months,
                "simulations": simulations,
                "seed": seed,
                "annual_excess_estimate": values.mean() * 12,
                "ci_2.5%": low,
                "median": median,
                "ci_97.5%": high,
                "probability_positive": float((annual > 0).mean()),
            }
        ]
    )


def concentration(holdings: pd.DataFrame) -> pd.DataFrame:
    frame = holdings.copy()
    frame["contribution"] = frame["target_weight"] * frame["forward_return"]
    frame["absolute_contribution"] = frame["contribution"].abs()
    total_absolute = float(frame["absolute_contribution"].sum())
    stock = frame.groupby("stock_code")["absolute_contribution"].sum().sort_values(
        ascending=False
    )
    industry = frame.groupby("industry")["absolute_contribution"].sum().sort_values(
        ascending=False
    )
    frame["year"] = pd.to_datetime(frame["realization_month"]).dt.year
    year = frame.groupby("year")["contribution"].sum().sort_values(ascending=False)
    positive_total = float(year.clip(lower=0).sum())
    return pd.DataFrame(
        [
            {
                "top_5_stocks_absolute_contribution_share": (
                    stock.head(5).sum() / total_absolute if total_absolute > 0 else np.nan
                ),
                "top_10_stocks_absolute_contribution_share": (
                    stock.head(10).sum() / total_absolute if total_absolute > 0 else np.nan
                ),
                "maximum_industry_absolute_contribution_share": (
                    industry.iloc[0] / total_absolute
                    if total_absolute > 0 and not industry.empty
                    else np.nan
                ),
                "best_year_positive_contribution_share": (
                    max(float(year.iloc[0]), 0.0) / positive_total
                    if positive_total > 0 and not year.empty
                    else np.nan
                ),
                "concentration_basis": "absolute stock/industry contribution; positive-return year contribution",
            }
        ]
    )


def main() -> None:
    lock = json.loads(
        (OUT / "residual_research_lock.json").read_text(encoding="utf-8")
    )
    config = PortfolioConfig(**lock["frozen_config"])
    predictions = pd.read_csv(
        OUT / "residual_predictions.csv",
        parse_dates=["signal_month", "realization_month"],
    )
    predictions["stock_code"] = predictions["stock_code"].astype(str).str.zfill(6)
    backtest = pd.read_csv(
        OUT / "cost_aware_backtest.csv",
        parse_dates=["signal_month", "realization_month"],
    )
    holdings = pd.read_csv(
        OUT / "cost_aware_holdings.csv",
        parse_dates=["signal_month", "realization_month"],
    )
    holdings["stock_code"] = holdings["stock_code"].astype(str).str.zfill(6)
    state = pd.read_csv(
        OUT / "execution_state_monthly.csv.gz",
        parse_dates=["realization_month"],
    )
    cost_and_subperiod(backtest).to_csv(
        OUT / "residual_robustness_summary.csv", index=False
    )
    signal_delay(holdings, state).to_csv(
        OUT / "residual_signal_delay.csv", index=False
    )
    parameter_neighbors(predictions, config).to_csv(
        OUT / "residual_parameter_neighbors.csv", index=False
    )
    extreme_month_tests(backtest).to_csv(
        OUT / "residual_extreme_months.csv", index=False
    )
    industry_exclusion(holdings).to_csv(
        OUT / "residual_industry_exclusion.csv", index=False
    )
    bootstrap = block_bootstrap(
        backtest["gross_return"] - backtest["turnover"] * 50 / 10_000
    )
    bootstrap.to_csv(OUT / "residual_bootstrap.csv", index=False)
    concentration(holdings).to_csv(
        OUT / "residual_concentration.csv", index=False
    )
    print(bootstrap.to_string(index=False))
    print(concentration(holdings).to_string(index=False))


if __name__ == "__main__":
    main()
