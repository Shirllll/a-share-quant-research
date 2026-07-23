from __future__ import annotations

"""Robustness summaries for the advanced walk-forward strategy."""

import numpy as np
import pandas as pd

from ml_quant import OUT, metric


def annual_returns(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["year"] = x["month"].dt.year
    rows = []
    for year, group in x.groupby("year"):
        rows.append({
            "year": year,
            "long_only": (1 + group["net_return"]).prod() - 1,
            "industry_neutral_long_short": (1 + group["long_short_net_return"]).prod() - 1,
            "benchmark": (1 + group["benchmark_return"]).prod() - 1,
        })
    return pd.DataFrame(rows)


def cost_stress(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bps in [0, 20, 50, 100, 150, 200]:
        returns = frame["gross_return"] - frame["turnover"] * bps / 10_000
        long_short = frame["long_short_gross_return"] - frame["long_short_turnover"] * bps / 10_000
        rows.append({"series": "long_only", "cost_bps": bps, **metric(returns),
                     "avg_turnover": frame["turnover"].mean()})
        rows.append({"series": "industry_neutral_long_short", "cost_bps": bps, **metric(long_short),
                     "avg_turnover": frame["long_short_turnover"].mean()})
    return pd.DataFrame(rows)


def block_bootstrap(frame: pd.DataFrame, simulations: int = 5_000, block: int = 6) -> pd.DataFrame:
    rows = []
    series_map = {
        "long_only_excess": frame["net_return"] - frame["benchmark_return"],
        "industry_neutral_long_short": frame["long_short_net_return"],
    }
    for series, values in series_map.items():
        excess = values.dropna().to_numpy()
        rng = np.random.default_rng(20260717)
        starts = np.arange(max(1, len(excess) - block + 1))
        blocks_needed = int(np.ceil(len(excess) / block))
        annual_excess = np.empty(simulations)
        for i in range(simulations):
            chosen = rng.choice(starts, blocks_needed, replace=True)
            sample = np.concatenate([excess[j:j + block] for j in chosen])[:len(excess)]
            annual_excess[i] = sample.mean() * 12
        low, median, high = np.quantile(annual_excess, [0.025, 0.5, 0.975])
        rows.append({
            "series": series, "block_months": block, "simulations": simulations,
            "annual_excess_estimate": excess.mean() * 12, "ci_2.5%": low,
            "median": median, "ci_97.5%": high,
            "probability_positive": (annual_excess > 0).mean(),
        })
    return pd.DataFrame(rows)


def subperiod_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = {
        "2018_2021": frame["month"] < "2022-01-01",
        "2022_2025": frame["month"] >= "2022-01-01",
    }
    for period, mask in periods.items():
        for series in ["net_return", "long_short_net_return"]:
            rows.append({"period": period, "series": series, **metric(frame.loc[mask, series])})
    return pd.DataFrame(rows)


def main() -> None:
    frame = pd.read_csv(OUT / "advanced_backtest.csv", parse_dates=["month"])
    annual_returns(frame).to_csv(OUT / "advanced_annual.csv", index=False)
    cost_stress(frame).to_csv(OUT / "advanced_cost_stress.csv", index=False)
    block_bootstrap(frame).to_csv(OUT / "advanced_bootstrap.csv", index=False)
    subperiod_stability(frame).to_csv(OUT / "advanced_subperiod.csv", index=False)
    print(cost_stress(frame).to_string(index=False))
    print(block_bootstrap(frame).to_string(index=False))


if __name__ == "__main__":
    main()
