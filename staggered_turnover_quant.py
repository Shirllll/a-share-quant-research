from __future__ import annotations

"""Staggered Ridge sleeves: refresh one independently held sleeve each month."""

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
from turnover_aware_quant import (
    Config,
    _portfolio_return,
    _weights_from_names,
    generate_ridge_scores,
    select_buffered_names,
    smooth_scores,
    true_weight_turnover,
    validate_no_future_features,
)
from turnover_budget_quant import (
    apply_turnover_budget,
    comparison_table,
    metrics_table,
    subperiod_table,
)


SLEEVE_CANDIDATES = (
    (2, None),
    (3, None),
    (4, None),
    (2, 0.15),
    (2, 0.175),
    (2, 0.20),
)
SOURCE_FILES = (
    "data_cleaning.py",
    "advanced_quant.py",
    "turnover_aware_quant.py",
    "staggered_turnover_quant.py",
    "test_staggered_turnover_quant.py",
)


@dataclass(frozen=True)
class StaggeredConfig:
    sleeves: int = 3
    long_monthly_turnover_cap: float | None = None
    diagnostic_leg_monthly_turnover_cap: float | None = None
    seed: int = 20260723


def combine_sleeves(sleeves: list[dict[str, float]]) -> dict[str, float]:
    if not sleeves:
        return {}
    scale = 1.0 / len(sleeves)
    combined: dict[str, float] = {}
    for sleeve in sleeves:
        for name, weight in sleeve.items():
            combined[name] = combined.get(name, 0.0) + scale * weight
    return {name: weight for name, weight in combined.items() if weight > 1e-12}


def carry_available(weights: dict[str, float], available: set[str]) -> dict[str, float]:
    """Inactive sleeves do not trade except for names that leave eligibility."""
    return {name: weight for name, weight in weights.items() if name in available}


def _desired_targets(
    frame: pd.DataFrame,
    previous_long: dict[str, float],
    previous_diag_long: dict[str, float],
    previous_diag_short: dict[str, float],
    config: Config,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    target_count = max(1, int(np.ceil(len(frame) * config.top_fraction)))
    long_names, _, _ = select_buffered_names(
        frame,
        previous_long,
        "long",
        target_count,
        1.0 / target_count,
        config.portfolio_capital_rmb,
        config,
        "staggered_long",
        True,
    )
    desired_long = _weights_from_names(long_names)

    industry_groups = [
        group for _, group in frame.groupby("industry", dropna=False) if len(group) >= 10
    ]
    pairs: list[tuple[set[str], set[str]]] = []
    industry_hint = 1.0 / max(len(industry_groups), 1)
    for group in industry_groups:
        count = max(1, int(np.floor(len(group) * config.top_fraction)))
        hint = industry_hint / count
        prev_l = {name: weight for name, weight in previous_diag_long.items() if name in group.index}
        prev_s = {name: weight for name, weight in previous_diag_short.items() if name in group.index}
        long_set, _, _ = select_buffered_names(
            group,
            prev_l,
            "long",
            count,
            hint,
            config.portfolio_capital_rmb,
            config,
            "staggered_diagnostic_long",
            False,
        )
        short_set, _, _ = select_buffered_names(
            group,
            prev_s,
            "short",
            count,
            hint,
            config.portfolio_capital_rmb,
            config,
            "staggered_diagnostic_short",
            False,
        )
        if long_set and short_set:
            pairs.append((long_set, short_set))
    desired_diag_long: dict[str, float] = {}
    desired_diag_short: dict[str, float] = {}
    if pairs:
        industry_weight = 1.0 / len(pairs)
        for long_set, short_set in pairs:
            desired_diag_long.update(
                {name: industry_weight / len(long_set) for name in long_set}
            )
            desired_diag_short.update(
                {name: industry_weight / len(short_set) for name in short_set}
            )
    return desired_long, desired_diag_long, desired_diag_short


def simulate_staggered(
    score_panel: pd.DataFrame,
    base_config: Config,
    staggered: StaggeredConfig,
) -> pd.DataFrame:
    config = replace(
        base_config,
        smoothing_weight=1.0,
        exit_fraction=0.25,
        hurdle_multiple=1.0,
        rebalance_frequency_months=1,
    )
    long_sleeves: list[dict[str, float]] = [{} for _ in range(staggered.sleeves)]
    diag_long_sleeves: list[dict[str, float]] = [{} for _ in range(staggered.sleeves)]
    diag_short_sleeves: list[dict[str, float]] = [{} for _ in range(staggered.sleeves)]
    previous_long: dict[str, float] = {}
    previous_diag_long: dict[str, float] = {}
    previous_diag_short: dict[str, float] = {}
    previous_smoothed: dict[str, float] = {}
    rows: list[dict[str, object]] = []

    for month_number, (signal_month, raw_frame) in enumerate(
        score_panel.groupby("signal_month", sort=True)
    ):
        frame = raw_frame.copy().set_index("stock_code", drop=True)
        frame["smoothed_score"] = smooth_scores(
            frame["raw_score"], previous_smoothed, config.smoothing_weight
        )
        previous_smoothed = frame["smoothed_score"].to_dict()
        available = set(frame.index)
        long_sleeves = [carry_available(sleeve, available) for sleeve in long_sleeves]
        diag_long_sleeves = [
            carry_available(sleeve, available) for sleeve in diag_long_sleeves
        ]
        diag_short_sleeves = [
            carry_available(sleeve, available) for sleeve in diag_short_sleeves
        ]

        active = month_number % staggered.sleeves
        desired = _desired_targets(
            frame,
            long_sleeves[active],
            diag_long_sleeves[active],
            diag_short_sleeves[active],
            config,
        )
        if month_number == 0:
            long_sleeves = [desired[0].copy() for _ in long_sleeves]
            diag_long_sleeves = [desired[1].copy() for _ in diag_long_sleeves]
            diag_short_sleeves = [desired[2].copy() for _ in diag_short_sleeves]
        else:
            long_sleeves[active] = desired[0]
            diag_long_sleeves[active] = desired[1]
            diag_short_sleeves[active] = desired[2]

        desired_long_weights = combine_sleeves(long_sleeves)
        desired_diag_long_weights = combine_sleeves(diag_long_sleeves)
        desired_diag_short_weights = combine_sleeves(diag_short_sleeves)
        if staggered.long_monthly_turnover_cap is None:
            long_weights = desired_long_weights
            long_turnover = true_weight_turnover(long_weights, previous_long)
        else:
            long_weights, long_turnover, _ = apply_turnover_budget(
                previous_long,
                desired_long_weights,
                available,
                staggered.long_monthly_turnover_cap,
            )
        if staggered.diagnostic_leg_monthly_turnover_cap is None:
            diag_long_weights = desired_diag_long_weights
            diag_short_weights = desired_diag_short_weights
            diag_long_turnover = true_weight_turnover(diag_long_weights, previous_diag_long)
            diag_short_turnover = true_weight_turnover(diag_short_weights, previous_diag_short)
        else:
            diag_long_weights, diag_long_turnover, _ = apply_turnover_budget(
                previous_diag_long,
                desired_diag_long_weights,
                available,
                staggered.diagnostic_leg_monthly_turnover_cap,
            )
            diag_short_weights, diag_short_turnover, _ = apply_turnover_budget(
                previous_diag_short,
                desired_diag_short_weights,
                available,
                staggered.diagnostic_leg_monthly_turnover_cap,
            )
        long_gross, long_missing = _portfolio_return(long_weights, frame)
        diag_long_return, diag_long_missing = _portfolio_return(diag_long_weights, frame)
        diag_short_return, diag_short_missing = _portfolio_return(diag_short_weights, frame)
        long_short_gross = diag_long_return - diag_short_return
        long_short_turnover = diag_long_turnover + diag_short_turnover
        rows.append(
            {
                "signal_month": signal_month,
                "month": raw_frame["realization_month"].iloc[0],
                "period": raw_frame["period"].iloc[0],
                "maximum_feature_month": raw_frame["maximum_feature_month"].max(),
                "active_sleeve": active,
                "sleeves": staggered.sleeves,
                "long_monthly_turnover_cap": staggered.long_monthly_turnover_cap,
                "diagnostic_leg_monthly_turnover_cap": staggered.diagnostic_leg_monthly_turnover_cap,
                "gross_return": long_gross,
                "turnover": long_turnover,
                "invested_weight": sum(long_weights.values()),
                "missing_forward_return_weight": long_missing,
                "net_return_20bps": long_gross - long_turnover * 20 / 10_000,
                "holdings": len(long_weights),
                "rank_ic": frame["raw_score"].corr(frame["target"], method="spearman"),
                "long_short_gross_return": long_short_gross,
                "diagnostic_long_turnover": diag_long_turnover,
                "diagnostic_short_turnover": diag_short_turnover,
                "long_short_turnover": long_short_turnover,
                "diagnostic_long_invested_weight": sum(diag_long_weights.values()),
                "diagnostic_short_invested_weight": sum(diag_short_weights.values()),
                "diagnostic_long_missing_forward_return_weight": diag_long_missing,
                "diagnostic_short_missing_forward_return_weight": diag_short_missing,
                "long_short_net_return_20bps": long_short_gross
                - long_short_turnover * 20 / 10_000,
            }
        )
        previous_long = long_weights
        previous_diag_long = diag_long_weights
        previous_diag_short = diag_short_weights
    return pd.DataFrame(rows)


def _sharpe(frame: pd.DataFrame, cost_bps: float) -> float:
    returns = frame["long_short_gross_return"] - frame["long_short_turnover"] * cost_bps / 10_000
    from ml_quant import metric

    return float(metric(returns)["sharpe_rf0"])


def select_sleeves(
    score_panel: pd.DataFrame,
    base_config: Config,
    quarterly: pd.DataFrame,
) -> tuple[StaggeredConfig, pd.DataFrame]:
    selection_panel = score_panel[score_panel["period"].isin(["development", "selection"])]
    quarterly_development = quarterly[quarterly["period"] == "development"]
    quarterly_selection = quarterly[quarterly["period"] == "selection"]
    baseline_development = _sharpe(quarterly_development, 50)
    baseline_selection = _sharpe(quarterly_selection, 50)
    baseline_turnover = float(quarterly_selection["long_short_turnover"].mean())
    rows = []
    for sleeves, leg_cap in SLEEVE_CANDIDATES:
        candidate = StaggeredConfig(
            sleeves=sleeves,
            long_monthly_turnover_cap=0.15 if leg_cap is not None else None,
            diagnostic_leg_monthly_turnover_cap=leg_cap,
        )
        backtest = simulate_staggered(
            selection_panel, base_config, candidate
        )
        development = backtest[backtest["period"] == "development"]
        selection = backtest[backtest["period"] == "selection"]
        rows.append(
            {
                "sleeves": sleeves,
                "long_monthly_turnover_cap": candidate.long_monthly_turnover_cap,
                "diagnostic_leg_monthly_turnover_cap": candidate.diagnostic_leg_monthly_turnover_cap,
                "uses_retrospective_test": False,
                "development_sharpe_50bps": _sharpe(development, 50),
                "selection_sharpe_20bps": _sharpe(selection, 20),
                "selection_sharpe_50bps": _sharpe(selection, 50),
                "selection_sharpe_100bps": _sharpe(selection, 100),
                "selection_long_turnover": selection["turnover"].mean(),
                "selection_long_short_turnover": selection["long_short_turnover"].mean(),
                "selection_long_invested_weight": selection["invested_weight"].mean(),
                "selection_diagnostic_min_invested_weight": min(
                    selection["diagnostic_long_invested_weight"].mean(),
                    selection["diagnostic_short_invested_weight"].mean(),
                ),
                "quarterly_development_sharpe_50bps": baseline_development,
                "quarterly_selection_sharpe_50bps": baseline_selection,
                "quarterly_selection_long_short_turnover": baseline_turnover,
            }
        )
    table = pd.DataFrame(rows)
    table["selection_sharpe_50_improvement"] = (
        table["selection_sharpe_50bps"] - table["quarterly_selection_sharpe_50bps"]
    )
    table["development_sharpe_50_improvement"] = (
        table["development_sharpe_50bps"] - table["quarterly_development_sharpe_50bps"]
    )
    table["turnover_change_vs_quarterly"] = (
        table["selection_long_short_turnover"]
        / table["quarterly_selection_long_short_turnover"]
        - 1
    )
    table["passes_constraints"] = (
        (table["selection_sharpe_50_improvement"] > 0)
        & (table["development_sharpe_50_improvement"] >= -0.05)
        & (
            table["selection_long_short_turnover"]
            <= table["quarterly_selection_long_short_turnover"]
        )
        & (table["selection_long_invested_weight"] >= 0.90)
        & (table["selection_diagnostic_min_invested_weight"] >= 0.90)
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
    long_cap = chosen["long_monthly_turnover_cap"]
    leg_cap = chosen["diagnostic_leg_monthly_turnover_cap"]
    return (
        StaggeredConfig(
            sleeves=int(chosen["sleeves"]),
            long_monthly_turnover_cap=None if pd.isna(long_cap) else float(long_cap),
            diagnostic_leg_monthly_turnover_cap=None if pd.isna(leg_cap) else float(leg_cap),
        ),
        table.sort_values("selection_objective", ascending=False),
    )


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
    config: StaggeredConfig, selected_row: dict[str, object], root: Path
) -> dict[str, object]:
    return {
        "lock_version": 1,
        "status": "experimental_frozen_for_forward_observation",
        "method": "staggered_ridge_sleeves",
        "config": asdict(config),
        "selection_passed_constraints": bool(selected_row["passes_constraints"]),
        "retrospective_used_for_parameter_selection": False,
        "git_commit": _git_commit(root),
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": {
            name: _sha256(root / name) for name in SOURCE_FILES if (root / name).exists()
        },
    }


def refresh_lock(root: Path) -> None:
    path = OUT / "staggered_turnover_lock.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    config = StaggeredConfig(**current["config"])
    path.write_text(
        json.dumps(
            build_lock(
                config,
                {"passes_constraints": current["selection_passed_constraints"]},
                root,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=OUT / "monthly_panel.csv.gz")
    parser.add_argument("--reuse-score-cache", action="store_true")
    parser.add_argument("--refresh-lock-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.refresh_lock_only:
        refresh_lock(root)
        return
    score_cache = OUT / "turnover_budget_score_panel.pkl"
    if args.reuse_score_cache and score_cache.exists():
        score_panel = pd.read_pickle(score_cache)
    else:
        panel = prepare_advanced(args.panel)
        score_panel, _ = generate_ridge_scores(panel, Config())
        score_panel.to_pickle(score_cache)
    quarterly = pd.read_csv(
        OUT / "turnover_aware_backtest.csv", parse_dates=["signal_month", "month"]
    )
    selected, selection = select_sleeves(score_panel, Config(), quarterly)
    backtest = simulate_staggered(score_panel, Config(), selected)
    validate_no_future_features(backtest)
    selected_row = selection[selection["selected"]].iloc[0].to_dict()
    metrics = metrics_table(backtest)
    metrics["series"] = metrics["series"].str.replace(
        "turnover_budget", "staggered_hybrid", regex=False
    )
    periods = subperiod_table(backtest)
    periods["series"] = periods["series"].str.replace(
        "turnover_budget", "staggered_hybrid", regex=False
    )
    comparison = comparison_table(backtest, quarterly).rename(
        columns={"turnover_budget": "staggered_hybrid"}
    )
    backtest.to_csv(OUT / "staggered_turnover_backtest.csv", index=False)
    metrics.to_csv(OUT / "staggered_turnover_metrics.csv", index=False)
    periods.to_csv(OUT / "staggered_turnover_subperiod.csv", index=False)
    comparison.to_csv(OUT / "staggered_turnover_comparison.csv", index=False)
    selection.to_csv(OUT / "staggered_turnover_parameter_selection.csv", index=False)
    (OUT / "staggered_turnover_lock.json").write_text(
        json.dumps(build_lock(selected, selected_row, root), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(selection.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
