from __future__ import annotations

import numpy as np
import pandas as pd

from ml_quant import OUT, prepare, walk_forward, metric


def annual_table(r: pd.DataFrame) -> pd.DataFrame:
    x = r.copy()
    x["year"] = x["month"].dt.year
    rows = []
    for year, g in x.groupby("year"):
        rows.append({"year": year,
                     "ml": (1 + g["net_return"]).prod() - 1,
                     "linear": (1 + g["linear_return"]).prod() - 1,
                     "benchmark": (1 + g["benchmark_return"]).prod() - 1,
                     "ml_excess": (1 + g["net_return"]).prod() / (1 + g["benchmark_return"]).prod() - 1})
    return pd.DataFrame(rows)


def cost_stress(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bps in [0, 20, 50, 100]:
        ret = r["gross_return"] - r["turnover"] * bps / 10_000
        rows.append({"test": "cost_bps", "parameter": bps, **metric(ret)})
    return pd.DataFrame(rows)


def block_bootstrap(r: pd.DataFrame, n: int = 5000, block: int = 6) -> pd.DataFrame:
    excess = (r["net_return"] - r["benchmark_return"]).dropna().to_numpy()
    rng = np.random.default_rng(20260703)
    means = []
    starts = np.arange(max(1, len(excess) - block + 1))
    blocks_needed = int(np.ceil(len(excess) / block))
    for _ in range(n):
        idx = rng.choice(starts, blocks_needed, replace=True)
        sample = np.concatenate([excess[i:i+block] for i in idx])[:len(excess)]
        means.append(sample.mean() * 12)
    q = np.quantile(means, [.025, .5, .975])
    return pd.DataFrame([{"test": "block_bootstrap_annual_excess", "parameter": f"{block}m_blocks_{n}x",
                          "estimate": excess.mean()*12, "ci_2.5%": q[0], "median": q[1],
                          "ci_97.5%": q[2], "probability_positive": np.mean(np.array(means) > 0)}])


def main() -> None:
    base = pd.read_csv(OUT / "ml_backtest_monthly.csv", parse_dates=["month"])
    annual_table(base).to_csv(OUT / "robustness_annual.csv", index=False)
    cost_stress(base).to_csv(OUT / "robustness_cost.csv", index=False)
    block_bootstrap(base).to_csv(OUT / "robustness_bootstrap.csv", index=False)

    p = prepare(OUT / "monthly_panel.csv.gz")
    configs = [(36, .10), (60, .05), (60, .20), (84, .10)]
    rows = [{"train_months": 60, "top_frac": .10, **metric(base["net_return"]),
             "avg_turnover": base["turnover"].mean()}]
    for train_months, top_frac in configs:
        r, _ = walk_forward(p, "2018-01-01", train_months, top_frac, max(.15, top_frac+.05), 20)
        rows.append({"train_months": train_months, "top_frac": top_frac, **metric(r["net_return"]),
                     "avg_turnover": r["turnover"].mean()})
    sensitivity = pd.DataFrame(rows).sort_values(["train_months", "top_frac"])
    sensitivity.to_csv(OUT / "robustness_parameters.csv", index=False)
    print("PARAMETER SENSITIVITY")
    print(sensitivity.to_string(index=False))
    print("\nBOOTSTRAP")
    print(block_bootstrap(base).to_string(index=False))


if __name__ == "__main__":
    main()
