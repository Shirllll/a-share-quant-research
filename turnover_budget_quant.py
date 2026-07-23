from __future__ import annotations

"""Monthly partial rebalancing under a hard turnover budget."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_quant import OUT, prepare_advanced
from ml_quant import metric
from turnover_aware_quant import (
    Config,
    _portfolio_return,
    _proxy_drag,
    _weights_from_names,
    actual_trade_records,
    generate_ridge_scores,
    select_buffered_names,
    smooth_scores,
    true_weight_turnover,
    validate_no_future_features,
)


LONG_BUDGET_GRID = (0.10, 0.15, 0.20)
LEG_BUDGET_GRID = (0.15, 0.175, 0.20)
FIXED_COST_GRID_BPS = (0, 20, 50, 100, 150, 200)
SOURCE_FILES = (
    "data_cleaning.py",
    "advanced_quant.py",
    "turnover_aware_quant.py",
    "turnover_budget_quant.py",
    "test_turnover_budget_quant.py",
)


@dataclass(frozen=True)
class BudgetConfig:
    long_monthly_turnover_budget: float = 0.15
    diagnostic_leg_monthly_turnover_budget: float = 0.20
    seed: int = 20260723


def apply_turnover_budget(
    previous: dict[str, float],
    desired: dict[str, float],
    available_names: set[str],
    turnover_budget: float,
) -> tuple[dict[str, float], float, float]:
    """Move toward a fresh target without exceeding the budget except forced exits.

    Names that are no longer eligible are removed immediately.  If those forced
    exits consume the budget, the remaining capital stays as cash until a later
    month.  The function never uses returns from the realization month.
    """
    if not previous:
        return desired.copy(), 1.0 if desired else 0.0, 0.0
    unavailable = set(previous) - available_names
    forced_turnover = 0.5 * sum(abs(previous[name]) for name in unavailable)
    base = {name: weight for name, weight in previous.items() if name in available_names}
    names = set(base) | set(desired)
    transition_turnover = 0.5 * sum(
        abs(desired.get(name, 0.0) - base.get(name, 0.0)) for name in names
    )
    remaining_budget = max(float(turnover_budget) - forced_turnover, 0.0)
    scale = min(1.0, remaining_budget / transition_turnover) if transition_turnover else 1.0
    current = {
        name: base.get(name, 0.0) + scale * (desired.get(name, 0.0) - base.get(name, 0.0))
        for name in names
    }
    current = {name: weight for name, weight in current.items() if weight > 1e-10}
    actual = true_weight_turnover(current, previous)
    return current, actual, forced_turnover


def _last_info(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        name: {
            "amount_clean": float(row["amount_clean"]),
            "volatility_clean": float(row["volatility_clean"]),
            "raw_score": float(row["raw_score"]),
            "smoothed_score": float(row["smoothed_score"]),
            "predicted_alpha": float(row["predicted_alpha"]),
        }
        for name, row in frame.iterrows()
    }


def simulate_budgeted(
    score_panel: pd.DataFrame,
    base_config: Config,
    budget_config: BudgetConfig,
    collect_details: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    previous_smoothed: dict[str, float] = {}
    previous_long: dict[str, float] = {}
    previous_diag_long: dict[str, float] = {}
    previous_diag_short: dict[str, float] = {}
    last_info: dict[str, dict[str, float]] = {}
    monthly_config = replace(
        base_config,
        smoothing_weight=1.0,
        exit_fraction=0.25,
        hurdle_multiple=1.0,
        rebalance_frequency_months=1,
    )

    for signal_month, raw_frame in score_panel.groupby("signal_month", sort=True):
        frame = raw_frame.copy().set_index("stock_code", drop=True)
        frame["smoothed_score"] = smooth_scores(
            frame["raw_score"], previous_smoothed, monthly_config.smoothing_weight
        )
        previous_smoothed = frame["smoothed_score"].to_dict()
        available_names = set(frame.index)

        target_count = max(1, int(np.ceil(len(frame) * monthly_config.top_fraction)))
        desired_long_names, long_gains, attempts = select_buffered_names(
            frame,
            previous_long,
            "long",
            target_count,
            1.0 / target_count,
            monthly_config.portfolio_capital_rmb,
            monthly_config,
            "budget_long_only",
            True,
        )
        desired_long = _weights_from_names(desired_long_names)
        long_weights, long_turnover, long_forced = apply_turnover_budget(
            previous_long,
            desired_long,
            available_names,
            budget_config.long_monthly_turnover_budget,
        )

        industry_groups = [
            (industry, group)
            for industry, group in frame.groupby("industry", dropna=False)
            if len(group) >= 10
        ]
        desired_pairs: list[
            tuple[set[str], set[str], dict[str, float], dict[str, float], list[dict[str, object]]]
        ] = []
        industry_hint = 1.0 / max(len(industry_groups), 1)
        for _, group in industry_groups:
            count = max(1, int(np.floor(len(group) * monthly_config.top_fraction)))
            hint = industry_hint / count
            prev_l = {name: weight for name, weight in previous_diag_long.items() if name in group.index}
            prev_s = {name: weight for name, weight in previous_diag_short.items() if name in group.index}
            long_set, long_gain, long_attempts = select_buffered_names(
                group,
                prev_l,
                "long",
                count,
                hint,
                monthly_config.portfolio_capital_rmb,
                monthly_config,
                "budget_diagnostic_long",
                False,
            )
            short_set, short_gain, short_attempts = select_buffered_names(
                group,
                prev_s,
                "short",
                count,
                hint,
                monthly_config.portfolio_capital_rmb,
                monthly_config,
                "budget_diagnostic_short",
                False,
            )
            if long_set and short_set:
                desired_pairs.append(
                    (long_set, short_set, long_gain, short_gain, long_attempts + short_attempts)
                )

        desired_diag_long: dict[str, float] = {}
        desired_diag_short: dict[str, float] = {}
        diag_long_gains: dict[str, float] = {}
        diag_short_gains: dict[str, float] = {}
        if desired_pairs:
            industry_weight = 1.0 / len(desired_pairs)
            for long_set, short_set, long_gain, short_gain, pair_attempts in desired_pairs:
                desired_diag_long.update(
                    {name: industry_weight / len(long_set) for name in long_set}
                )
                desired_diag_short.update(
                    {name: industry_weight / len(short_set) for name in short_set}
                )
                diag_long_gains.update(long_gain)
                diag_short_gains.update(short_gain)
                attempts.extend(pair_attempts)

        diag_long_weights, diag_long_turnover, diag_long_forced = apply_turnover_budget(
            previous_diag_long,
            desired_diag_long,
            available_names,
            budget_config.diagnostic_leg_monthly_turnover_budget,
        )
        diag_short_weights, diag_short_turnover, diag_short_forced = apply_turnover_budget(
            previous_diag_short,
            desired_diag_short,
            available_names,
            budget_config.diagnostic_leg_monthly_turnover_budget,
        )

        orders = actual_trade_records(
            previous_long,
            long_weights,
            frame,
            last_info,
            "budget_long_only",
            long_gains,
            monthly_config.portfolio_capital_rmb,
            monthly_config,
        )
        orders += actual_trade_records(
            previous_diag_long,
            diag_long_weights,
            frame,
            last_info,
            "budget_diagnostic_long",
            diag_long_gains,
            monthly_config.portfolio_capital_rmb,
            monthly_config,
        )
        orders += actual_trade_records(
            previous_diag_short,
            diag_short_weights,
            frame,
            last_info,
            "budget_diagnostic_short",
            diag_short_gains,
            monthly_config.portfolio_capital_rmb,
            monthly_config,
        )
        if collect_details:
            for record in orders + attempts:
                record.update(
                    {
                        "signal_month": signal_month,
                        "realization_month": raw_frame["realization_month"].iloc[0],
                    }
                )
                trade_rows.append(record)

        long_gross, long_missing = _portfolio_return(long_weights, frame)
        diag_long_return, diag_long_missing = _portfolio_return(diag_long_weights, frame)
        diag_short_return, diag_short_missing = _portfolio_return(diag_short_weights, frame)
        long_short_gross = diag_long_return - diag_short_return
        long_short_turnover = diag_long_turnover + diag_short_turnover
        long_proxy_drag = _proxy_drag(orders, "budget_long_only")
        diagnostic_proxy_drag = _proxy_drag(orders, "budget_diagnostic_long") + _proxy_drag(
            orders, "budget_diagnostic_short"
        )
        rows.append(
            {
                "signal_month": signal_month,
                "month": raw_frame["realization_month"].iloc[0],
                "period": raw_frame["period"].iloc[0],
                "maximum_feature_month": raw_frame["maximum_feature_month"].max(),
                "gross_return": long_gross,
                "turnover": long_turnover,
                "turnover_budget": budget_config.long_monthly_turnover_budget,
                "forced_turnover": long_forced,
                "invested_weight": sum(long_weights.values()),
                "missing_forward_return_weight": long_missing,
                "net_return_20bps": long_gross - long_turnover * 20 / 10_000,
                "proxy_net_return": long_gross - long_proxy_drag,
                "holdings": len(long_weights),
                "rank_ic": frame["raw_score"].corr(frame["target"], method="spearman"),
                "long_short_gross_return": long_short_gross,
                "diagnostic_long_turnover": diag_long_turnover,
                "diagnostic_short_turnover": diag_short_turnover,
                "diagnostic_leg_turnover_budget": budget_config.diagnostic_leg_monthly_turnover_budget,
                "diagnostic_forced_turnover": diag_long_forced + diag_short_forced,
                "diagnostic_long_invested_weight": sum(diag_long_weights.values()),
                "diagnostic_short_invested_weight": sum(diag_short_weights.values()),
                "diagnostic_long_missing_forward_return_weight": diag_long_missing,
                "diagnostic_short_missing_forward_return_weight": diag_short_missing,
                "long_short_turnover": long_short_turnover,
                "long_short_net_return_20bps": long_short_gross - long_short_turnover * 20 / 10_000,
                "long_short_proxy_net_return": long_short_gross - diagnostic_proxy_drag,
            }
        )

        if collect_details:
            percentile = frame.groupby("industry", dropna=False)["raw_score"].rank(pct=True)
            percentile = percentile.fillna(frame["raw_score"].rank(pct=True))
            decile = np.ceil(percentile * 10).clip(1, 10).astype(int)
            for number, group in frame.assign(decile=decile).groupby("decile"):
                decile_rows.append(
                    {
                        "signal_month": signal_month,
                        "realization_month": raw_frame["realization_month"].iloc[0],
                        "period": raw_frame["period"].iloc[0],
                        "decile": number,
                        "return": group["forward_return"].mean(),
                        "stocks": len(group),
                    }
                )

        previous_long = long_weights
        previous_diag_long = diag_long_weights
        previous_diag_short = diag_short_weights
        last_info = _last_info(frame)

    return pd.DataFrame(rows), pd.DataFrame(decile_rows), pd.DataFrame(trade_rows)


def _sharpe(frame: pd.DataFrame, cost_bps: float) -> float:
    returns = frame["long_short_gross_return"] - frame["long_short_turnover"] * cost_bps / 10_000
    return float(metric(returns)["sharpe_rf0"])


def select_budgets(
    score_panel: pd.DataFrame,
    base_config: Config,
    quarterly_backtest: pd.DataFrame,
) -> tuple[BudgetConfig, pd.DataFrame]:
    selection_panel = score_panel[score_panel["period"].isin(["development", "selection"])].copy()
    quarterly_selection = quarterly_backtest[quarterly_backtest["period"] == "selection"]
    quarterly_development = quarterly_backtest[quarterly_backtest["period"] == "development"]
    baseline_selection_turnover = float(quarterly_selection["long_short_turnover"].mean())
    baseline_selection_sharpe_50 = _sharpe(quarterly_selection, 50)
    baseline_development_sharpe_50 = _sharpe(quarterly_development, 50)
    rows = []
    for long_budget in LONG_BUDGET_GRID:
        for leg_budget in LEG_BUDGET_GRID:
            budget = BudgetConfig(long_budget, leg_budget)
            backtest, _, _ = simulate_budgeted(selection_panel, base_config, budget)
            development = backtest[backtest["period"] == "development"]
            selection = backtest[backtest["period"] == "selection"]
            rows.append(
                {
                    "long_monthly_turnover_budget": long_budget,
                    "diagnostic_leg_monthly_turnover_budget": leg_budget,
                    "uses_retrospective_test": False,
                    "development_sharpe_50bps": _sharpe(development, 50),
                    "selection_sharpe_20bps": _sharpe(selection, 20),
                    "selection_sharpe_50bps": _sharpe(selection, 50),
                    "selection_sharpe_100bps": _sharpe(selection, 100),
                    "selection_long_turnover": selection["turnover"].mean(),
                    "selection_long_short_turnover": selection["long_short_turnover"].mean(),
                    "quarterly_selection_sharpe_50bps": baseline_selection_sharpe_50,
                    "quarterly_development_sharpe_50bps": baseline_development_sharpe_50,
                    "quarterly_selection_long_short_turnover": baseline_selection_turnover,
                }
            )
    table = pd.DataFrame(rows)
    table["turnover_change_vs_quarterly"] = (
        table["selection_long_short_turnover"] / table["quarterly_selection_long_short_turnover"] - 1
    )
    table["selection_sharpe_50_improvement"] = (
        table["selection_sharpe_50bps"] - table["quarterly_selection_sharpe_50bps"]
    )
    table["development_sharpe_50_improvement"] = (
        table["development_sharpe_50bps"] - table["quarterly_development_sharpe_50bps"]
    )
    table["passes_constraints"] = (
        (table["selection_sharpe_50_improvement"] > 0)
        & (table["development_sharpe_50_improvement"] >= -0.05)
        & (table["selection_long_short_turnover"] <= table["quarterly_selection_long_short_turnover"])
        & (table["selection_sharpe_20bps"] > 0)
    )
    table["selection_objective"] = (
        table["selection_sharpe_50bps"]
        + 0.25 * table["development_sharpe_50bps"]
        - 0.25 * table["selection_long_short_turnover"]
    )
    eligible = table[table["passes_constraints"]]
    pool = eligible if not eligible.empty else table
    selected_index = pool.sort_values(
        ["selection_objective", "selection_long_short_turnover"], ascending=[False, True]
    ).index[0]
    table["selected"] = table.index == selected_index
    chosen = table.loc[selected_index]
    return (
        BudgetConfig(
            float(chosen["long_monthly_turnover_budget"]),
            float(chosen["diagnostic_leg_monthly_turnover_budget"]),
        ),
        table.sort_values("selection_objective", ascending=False),
    )


def metrics_table(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, return_column, turnover_column in (
        ("turnover_budget_long_only", "gross_return", "turnover"),
        (
            "turnover_budget_industry_neutral_diagnostic",
            "long_short_gross_return",
            "long_short_turnover",
        ),
    ):
        for cost_bps in FIXED_COST_GRID_BPS:
            returns = backtest[return_column] - backtest[turnover_column] * cost_bps / 10_000
            rows.append(
                {
                    "series": name,
                    "cost_bps": cost_bps,
                    **metric(returns),
                    "avg_turnover": backtest[turnover_column].mean(),
                }
            )
    return pd.DataFrame(rows)


def subperiod_table(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in ("development", "selection", "retrospective_test"):
        sample = backtest[backtest["period"] == period]
        for cost_bps in (20, 50, 100):
            for name, return_column, turnover_column in (
                ("turnover_budget_long_only", "gross_return", "turnover"),
                (
                    "turnover_budget_industry_neutral_diagnostic",
                    "long_short_gross_return",
                    "long_short_turnover",
                ),
            ):
                returns = sample[return_column] - sample[turnover_column] * cost_bps / 10_000
                rows.append(
                    {
                        "period": period,
                        "series": name,
                        "cost_bps": cost_bps,
                        **metric(returns),
                        "avg_turnover": sample[turnover_column].mean(),
                    }
                )
    return pd.DataFrame(rows)


def comparison_table(budgeted: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    quarterly_columns = quarterly[
        ["month", "gross_return", "turnover", "long_short_gross_return", "long_short_turnover"]
    ].rename(
        columns={
            "gross_return": "gross_return_quarterly",
            "turnover": "turnover_quarterly",
            "long_short_gross_return": "long_short_gross_return_quarterly",
            "long_short_turnover": "long_short_turnover_quarterly",
        }
    )
    budget_columns = budgeted[
        ["month", "gross_return", "turnover", "long_short_gross_return", "long_short_turnover"]
    ].rename(
        columns={
            "gross_return": "gross_return_budget",
            "turnover": "turnover_budget",
            "long_short_gross_return": "long_short_gross_return_budget",
            "long_short_turnover": "long_short_turnover_budget",
        }
    )
    merged = quarterly_columns.merge(budget_columns, on="month")
    rows = []
    for cost_bps in (20, 50, 100):
        for label, return_column, turnover_column in (
            ("long_only", "gross_return", "turnover"),
            ("long_short", "long_short_gross_return", "long_short_turnover"),
        ):
            q = metric(
                merged[f"{return_column}_quarterly"]
                - merged[f"{turnover_column}_quarterly"] * cost_bps / 10_000
            )
            b = metric(
                merged[f"{return_column}_budget"]
                - merged[f"{turnover_column}_budget"] * cost_bps / 10_000
            )
            rows.append(
                {
                    "metric": f"{label}_sharpe_{cost_bps}bps",
                    "quarterly": q["sharpe_rf0"],
                    "turnover_budget": b["sharpe_rf0"],
                    "change": b["sharpe_rf0"] - q["sharpe_rf0"],
                    "common_months": len(merged),
                }
            )
    rows.extend(
        [
            {
                "metric": "long_only_avg_turnover",
                "quarterly": merged["turnover_quarterly"].mean(),
                "turnover_budget": merged["turnover_budget"].mean(),
                "change": merged["turnover_budget"].mean() - merged["turnover_quarterly"].mean(),
                "common_months": len(merged),
            },
            {
                "metric": "long_short_avg_turnover",
                "quarterly": merged["long_short_turnover_quarterly"].mean(),
                "turnover_budget": merged["long_short_turnover_budget"].mean(),
                "change": (
                    merged["long_short_turnover_budget"].mean()
                    - merged["long_short_turnover_quarterly"].mean()
                ),
                "common_months": len(merged),
            },
        ]
    )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_lock(
    selected: BudgetConfig,
    selected_row: dict[str, object],
    root: Path,
) -> dict[str, object]:
    return {
        "lock_version": 1,
        "status": "experimental_frozen_for_forward_observation",
        "method": "monthly_partial_rebalance_with_hard_turnover_budget",
        "budget_config": asdict(selected),
        "fixed_signal_config": {
            "smoothing_weight": 1.0,
            "exit_fraction": 0.25,
            "hurdle_multiple": 1.0,
            "ridge_only": True,
        },
        "selection_passed_constraints": bool(selected_row["passes_constraints"]),
        "retrospective_used_for_parameter_selection": False,
        "data_intervals": {
            "development": ["2018-01-01", "2021-12-31"],
            "selection": ["2022-01-01", "2023-12-31"],
            "retrospective_test": ["2024-01-01", "2025-12-31"],
            "true_forward_start": "2026-01-01",
        },
        "git_commit": _git_commit(root),
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": {
            name: _sha256(root / name) for name in SOURCE_FILES if (root / name).exists()
        },
    }


def refresh_lock(root: Path) -> None:
    path = OUT / "turnover_budget_lock.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    selected = BudgetConfig(**current["budget_config"])
    selected_row = {"passes_constraints": current["selection_passed_constraints"]}
    path.write_text(
        json.dumps(build_lock(selected, selected_row, root), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=OUT / "monthly_panel.csv.gz")
    parser.add_argument("--refresh-lock-only", action="store_true")
    parser.add_argument("--reuse-score-cache", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.refresh_lock_only:
        refresh_lock(root)
        return

    base_config = Config()
    score_cache = OUT / "turnover_budget_score_panel.pkl"
    if args.reuse_score_cache and score_cache.exists():
        score_panel = pd.read_pickle(score_cache)
    else:
        panel = prepare_advanced(args.panel)
        score_panel, _ = generate_ridge_scores(panel, base_config)
        score_panel.to_pickle(score_cache)
    quarterly = pd.read_csv(
        OUT / "turnover_aware_backtest.csv", parse_dates=["signal_month", "month"]
    )
    selected, selection = select_budgets(score_panel, base_config, quarterly)
    backtest, deciles, trades = simulate_budgeted(
        score_panel, base_config, selected, collect_details=True
    )
    validate_no_future_features(backtest)
    selected_row = selection[selection["selected"]].iloc[0].to_dict()

    backtest.to_csv(OUT / "turnover_budget_backtest.csv", index=False)
    metrics_table(backtest).to_csv(OUT / "turnover_budget_metrics.csv", index=False)
    subperiod_table(backtest).to_csv(OUT / "turnover_budget_subperiod.csv", index=False)
    selection.to_csv(OUT / "turnover_budget_parameter_selection.csv", index=False)
    comparison_table(backtest, quarterly).to_csv(
        OUT / "turnover_budget_comparison.csv", index=False
    )
    deciles.to_csv(OUT / "turnover_budget_decile_returns.csv", index=False)
    trades.to_csv(OUT / "turnover_budget_trade_log.csv", index=False)
    (OUT / "turnover_budget_lock.json").write_text(
        json.dumps(build_lock(selected, selected_row, root), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(selection[selection["selected"]].to_string(index=False))
    print(comparison_table(backtest, quarterly).to_string(index=False))


if __name__ == "__main__":
    main()
