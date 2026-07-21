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
            "advanced": (1 + group["net_return"]).prod() - 1,
            "advanced_regime_managed": (1 + group["risk_managed_return"]).prod() - 1,
            "previous_dl": (1 + group["previous_dl_return"]).prod() - 1,
            "benchmark": (1 + group["benchmark_return"]).prod() - 1,
        })
    return pd.DataFrame(rows)


def cost_stress(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bps in [0, 20, 50, 100]:
        returns = frame["gross_return"] - frame["turnover"] * bps / 10_000
        rows.append({"series": "raw", "cost_bps": bps, **metric(returns),
                     "avg_turnover": frame["turnover"].mean()})
        rows.append({"series": "regime_managed", "cost_bps": bps, **metric(frame["exposure"] * returns),
                     "avg_turnover": frame["turnover"].mean()})
    return pd.DataFrame(rows)


def block_bootstrap(frame: pd.DataFrame, simulations: int = 5_000, block: int = 6) -> pd.DataFrame:
    rows = []
    for series in ["net_return", "risk_managed_return"]:
        excess = (frame[series] - frame["benchmark_return"]).dropna().to_numpy()
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


def main() -> None:
    frame = pd.read_csv(OUT / "advanced_backtest.csv", parse_dates=["month"])
    annual_returns(frame).to_csv(OUT / "advanced_annual.csv", index=False)
    cost_stress(frame).to_csv(OUT / "advanced_cost_stress.csv", index=False)
    block_bootstrap(frame).to_csv(OUT / "advanced_bootstrap.csv", index=False)
    print(cost_stress(frame).to_string(index=False))
    print(block_bootstrap(frame).to_string(index=False))


if __name__ == "__main__":
    main()
