from __future__ import annotations

"""Small, pre-retrospective robustness grid for the frozen MLP portfolio.

This module is deliberately not another model zoo.  It reuses the frozen,
walk-forward signal panel and tests a handful of economically interpretable
portfolio variants.  A candidate can replace the production core only if it
improves the 2018-2023 risk-adjusted result without materially increasing
turnover.  The 2024-2025 columns are written to a separate file and are never
used by ``accept_candidate``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from optimized_quant import (
    OUT,
    ResearchConfig,
    build_portfolio,
    metric,
    period_metric,
    prepare_clean,
    signal_for_model,
    smooth_scores,
)


@dataclass(frozen=True)
class CandidateRule:
    maximum_development_sharpe_loss: float = 0.02
    minimum_selection_sharpe_gain: float = 0.0
    minimum_average_sharpe_gain: float = 0.01
    maximum_turnover_increase: float = 0.05


def rank_tilt_portfolio(signals: pd.DataFrame, tilt: float = 0.50,
                        cost_bps: float = 50.0) -> pd.DataFrame:
    """Use the core membership rule, then apply a bounded rank tilt to weights."""
    panel = smooth_scores(signals, 0.75)
    previous: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for signal_month, group in panel.groupby("month", sort=True):
        group = group.dropna(subset=["forward_return", "smoothed_score"]).copy()
        if group.empty:
            continue
        count = max(1, int(np.ceil(len(group) * 0.10)))
        exit_count = max(count, int(np.ceil(len(group) * 0.20)))
        permitted = set(group.nlargest(exit_count, "smoothed_score")["Stkcd"])
        retained = set(previous) & set(group["Stkcd"]) & permitted
        additions = group[~group["Stkcd"].isin(retained)].nlargest(
            max(0, count - len(retained)), "smoothed_score"
        )
        chosen = pd.concat([
            group[group["Stkcd"].isin(retained)], additions,
        ]).drop_duplicates("Stkcd").nlargest(count, "smoothed_score").copy()
        rank = chosen["smoothed_score"].rank(pct=True).to_numpy()
        raw_weight = np.maximum(1.0 + tilt * (rank - 0.5), 1e-8)
        raw_weight /= raw_weight.sum()
        weights = dict(zip(chosen["Stkcd"], raw_weight))
        union = set(previous) | set(weights)
        turnover = 0.5 * sum(
            abs(weights.get(code, 0.0) - previous.get(code, 0.0)) for code in union
        )
        realized = chosen.set_index("Stkcd")["forward_return"]
        gross_return = sum(
            weight * realized.loc[code] for code, weight in weights.items()
        )
        rows.append({
            "signal_month": signal_month,
            "month": pd.Timestamp(signal_month) + pd.offsets.MonthEnd(1),
            "gross_return": gross_return,
            "turnover": turnover,
            "holdings": len(weights),
        })
        previous = weights
    result = pd.DataFrame(rows)
    result["net_return"] = result["gross_return"] - result["turnover"] * cost_bps / 10_000
    return result


def candidate_backtests(signals: pd.DataFrame, clean_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construct the fixed candidate set without consulting retrospective returns."""
    mlp = signal_for_model(signals, "mlp")
    variants: dict[str, pd.DataFrame] = {
        "production_core": build_portfolio(mlp, 0.10, 0.20, 0.75, 50),
        "concentrated_top5": build_portfolio(mlp, 0.05, 0.10, 1.00, 50),
        "score_rank_tilt_050": rank_tilt_portfolio(mlp, 0.50, 50),
    }

    volatility = mlp.copy()
    volatility["volatility_percentile"] = volatility.groupby(
        "month", observed=True
    )["volatility_clean"].rank(pct=True)
    volatility = volatility[volatility["volatility_percentile"].le(0.90)]
    variants["volatility_cap90"] = build_portfolio(volatility, 0.10, 0.20, 0.75, 50)

    consensus = signals.copy()
    mlp_rank = consensus.groupby("month", observed=True)["mlp_score"].rank(pct=True)
    ridge_rank = consensus.groupby("month", observed=True)["ridge_score"].rank(pct=True)
    consensus["raw_score"] = 0.80 * mlp_rank + 0.20 * ridge_rank - 0.5
    variants["ridge_consensus20"] = build_portfolio(consensus, 0.10, 0.25, 1.00, 50)

    factor_columns = ["value_pe", "value_pb", "value_ps"]
    values = clean_panel[["month", "Stkcd", *factor_columns]].copy()
    values["Stkcd"] = values["Stkcd"].astype("string")
    value = mlp.copy()
    value["Stkcd"] = value["Stkcd"].astype("string")
    value = value.merge(values, on=["month", "Stkcd"], how="left", validate="one_to_one")
    value["value_score"] = value[factor_columns].mean(axis=1)
    value_rank = value.groupby("month", observed=True)["value_score"].rank(pct=True)
    mlp_rank = value.groupby("month", observed=True)["mlp_score"].rank(pct=True)
    value["raw_score"] = 0.90 * mlp_rank + 0.10 * value_rank - 0.5
    variants["value_tilt10"] = build_portfolio(value, 0.10, 0.20, 0.75, 50)
    return variants


def accept_candidate(row: pd.Series, baseline: pd.Series,
                     rule: CandidateRule = CandidateRule()) -> tuple[bool, str]:
    """Apply only development/selection metrics and turnover to the gate."""
    failures = []
    if row["development_sharpe_50bp"] < (
        baseline["development_sharpe_50bp"] - rule.maximum_development_sharpe_loss
    ):
        failures.append("development_sharpe")
    if row["selection_sharpe_50bp"] < (
        baseline["selection_sharpe_50bp"] + rule.minimum_selection_sharpe_gain
    ):
        failures.append("selection_sharpe")
    if row["average_pre_retro_sharpe_50bp"] < (
        baseline["average_pre_retro_sharpe_50bp"] + rule.minimum_average_sharpe_gain
    ):
        failures.append("average_sharpe_gain")
    if row["pre_retro_turnover"] > (
        baseline["pre_retro_turnover"] * (1 + rule.maximum_turnover_increase)
    ):
        failures.append("turnover")
    return not failures, "accepted" if not failures else ";".join(failures)


def evaluate(variants: dict[str, pd.DataFrame],
             config: ResearchConfig = ResearchConfig()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_rows = []
    retrospective_rows = []
    stress_rows = []
    for name, backtest in variants.items():
        development = period_metric(backtest, config.start, config.development_end, 50)
        selection = period_metric(
            backtest, config.selection_start, config.selection_end, 50
        )
        pre_retro = backtest[backtest["month"].between(config.start, config.selection_end)]
        selection_rows.append({
            "candidate": name,
            "development_sharpe_50bp": development["sharpe_rf0"],
            "selection_sharpe_50bp": selection["sharpe_rf0"],
            "average_pre_retro_sharpe_50bp": np.mean([
                development["sharpe_rf0"], selection["sharpe_rf0"]
            ]),
            "development_turnover": development["mean_turnover"],
            "selection_turnover": selection["mean_turnover"],
            "pre_retro_turnover": pre_retro["turnover"].mean(),
            "selection_used_retrospective": False,
        })
        retrospective = period_metric(
            backtest, config.retrospective_start, config.retrospective_end, 50
        )
        retrospective_rows.append({
            "candidate": name,
            "retrospective_annual_return_50bp": retrospective["annual_return"],
            "retrospective_sharpe_50bp": retrospective["sharpe_rf0"],
            "retrospective_max_drawdown_50bp": retrospective["max_drawdown"],
            "retrospective_turnover": retrospective["mean_turnover"],
        })
        for cost in (20, 50, 100):
            net = backtest["gross_return"] - backtest["turnover"] * cost / 10_000
            stress_rows.append({
                "candidate": name,
                "cost_bps": cost,
                "mean_turnover": backtest["turnover"].mean(),
                **metric(net),
            })

    selection_table = pd.DataFrame(selection_rows)
    baseline = selection_table.loc[
        selection_table["candidate"].eq("production_core")
    ].iloc[0]
    decisions = selection_table.apply(
        lambda row: (True, "frozen_baseline")
        if row["candidate"] == "production_core"
        else accept_candidate(row, baseline),
        axis=1,
    )
    selection_table[["accepted", "rejection_reason"]] = pd.DataFrame(
        decisions.tolist(), index=selection_table.index
    )
    return selection_table, pd.DataFrame(retrospective_rows), pd.DataFrame(stress_rows)


def main() -> None:
    signal_path = OUT / "optimized_signal_panel.csv.gz"
    panel_path = OUT / "monthly_panel.csv.gz"
    if not signal_path.exists() or not panel_path.exists():
        raise FileNotFoundError("Run quant_project.py and optimized_quant.py first")
    signals = pd.read_csv(
        signal_path, parse_dates=["month", "realization_month"], dtype={"Stkcd": "string"}
    )
    clean_panel = prepare_clean(panel_path)
    variants = candidate_backtests(signals, clean_panel)
    selection, retrospective, stress = evaluate(variants)
    selection.to_csv(OUT / "sharpe_candidate_selection.csv", index=False)
    retrospective.to_csv(OUT / "sharpe_candidate_retrospective.csv", index=False)
    stress.to_csv(OUT / "sharpe_candidate_cost_stress.csv", index=False)
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
