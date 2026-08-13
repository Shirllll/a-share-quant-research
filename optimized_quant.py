from __future__ import annotations

"""Leakage-safe, turnover-aware optimization of the cleaned Ridge/MLP research model.

The existing first version is intentionally left untouched.  This module first applies
``data_cleaning.clean_monthly_panel``, generates walk-forward stock-level predictions,
selects portfolio controls using only 2018-2023, and reports 2024-2025 exactly once as
a retrospective (not pristine holdout) period.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_cleaning import clean_monthly_panel, cleaning_report
from deep_learning_quant import MLP, metric, ridge_fit, ridge_predict, sampled


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
FEATURES = [
    "momentum", "value_pe", "value_pb", "value_ps", "low_volatility",
    "small_size", "reversal", "liquidity", "roe", "roa", "gross_margin",
    "cash_quality", "low_leverage",
]


@dataclass(frozen=True)
class ResearchConfig:
    start: str = "2018-01-01"
    train_months: int = 60
    validation_months: int = 12
    top_fraction: float = 0.10
    smoothing_candidates: tuple[float, ...] = (1.0, 0.75, 0.50)
    exit_candidates: tuple[float, ...] = (0.15, 0.20, 0.25)
    mlp_weight_candidates: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)
    signal_model_candidates: tuple[str, ...] = ("ridge", "mlp", "dynamic_blend")
    selection_cost_bps: int = 50
    report_cost_bps: int = 20
    development_end: str = "2021-12-31"
    selection_start: str = "2022-01-01"
    selection_end: str = "2023-12-31"
    retrospective_start: str = "2024-01-01"
    retrospective_end: str = "2025-12-31"
    volatility_target: float = 0.15
    minimum_exposure: float = 0.50
    random_seed: int = 20260811


def _rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average") - 0.5


def prepare_clean(panel_path: Path) -> pd.DataFrame:
    """Apply the canonical cleaner before constructing any regression inputs."""
    panel = clean_monthly_panel(panel_path)
    panel = panel.sort_values(["Stkcd", "month"]).copy()
    panel["forward_return"] = panel.groupby("Stkcd", sort=False)["ret_clean"].shift(-1)
    panel["realization_month"] = panel["month"] + pd.offsets.MonthEnd(1)

    raw = {
        "momentum": panel["mom_12_1"],
        "value_pe": 1 / panel["PE1TTM_clean"],
        "value_pb": 1 / panel["PBV1B_clean"],
        "value_ps": 1 / panel["PSTTM_clean"],
        "low_volatility": -panel["volatility_clean"],
        "small_size": -np.log(panel["size_clean"]),
        "reversal": -panel["ret_clean"],
        "liquidity": panel["illiq_clean"],
        "roe": panel["F050504C_clean"],
        "roa": panel["F050204C_clean"],
        "gross_margin": panel["F053301C_clean"],
        "cash_quality": panel["F052901C_clean"],
        "low_leverage": -panel["F011201A_clean"],
    }
    for name, values in raw.items():
        within_industry = values.groupby(
            [panel["month"], panel["industry"]], dropna=False
        ).rank(pct=True)
        fallback = values.groupby(panel["month"]).rank(pct=True)
        ranked = within_industry.fillna(fallback) - 0.5
        panel[f"{name}_missing"] = ranked.isna().astype(np.float32)
        panel[name] = ranked.fillna(0).astype(np.float32)

    # The prediction target is explicitly an industry-relative next-month return rank.
    within_industry_target = panel["forward_return"].groupby(
        [panel["month"], panel["industry"]], dropna=False
    ).rank(pct=True)
    fallback_target = panel["forward_return"].groupby(panel["month"]).rank(pct=True)
    panel["target"] = within_industry_target.fillna(fallback_target) - 0.5

    liquid_cut = panel["amount_clean"].groupby(panel["month"]).transform(
        lambda values: values.quantile(0.20)
    )
    panel["eligible"] = (
        panel["trdsta"].eq(1)
        & panel["listed_days"].ge(180)
        & panel["amount_clean"].ge(liquid_cut)
        & panel["trading_days"].ge(15)
        & panel["special_state"].eq("A")
        & panel["listing_state"].eq("A")
        & ~panel["flag_return_outlier"]
    )
    return panel


def matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = FEATURES + [f"{name}_missing" for name in FEATURES]
    return frame[columns].to_numpy(np.float32)


def monthly_rank_ic(frame: pd.DataFrame, score: str) -> float:
    values = frame.groupby("month", observed=True).apply(
        lambda group: group[[score, "target"]].corr(method="spearman").iloc[0, 1]
        if group[[score, "target"]].dropna().shape[0] >= 30 else np.nan,
        include_groups=False,
    )
    return float(values.mean())


def choose_blend(validation: pd.DataFrame, ridge_score: np.ndarray,
                 mlp_score: np.ndarray, candidates: tuple[float, ...]) -> tuple[float, float]:
    """Choose the least-complex blend within 0.001 IC of the best monthly IC."""
    work = validation[["month", "target"]].copy()
    work["ridge_rank"] = pd.Series(ridge_score, index=work.index).groupby(work["month"]).rank(pct=True)
    work["mlp_rank"] = pd.Series(mlp_score, index=work.index).groupby(work["month"]).rank(pct=True)
    scored: list[tuple[float, float]] = []
    for weight in candidates:
        work["blend"] = (1 - weight) * work["ridge_rank"] + weight * work["mlp_rank"]
        scored.append((monthly_rank_ic(work, "blend"), weight))
    best_ic = max(value for value, _ in scored)
    eligible = [(value, weight) for value, weight in scored if value >= best_ic - 0.001]
    chosen_ic, chosen_weight = min(eligible, key=lambda item: item[1])
    return chosen_weight, chosen_ic


def generate_signals(panel: pd.DataFrame, config: ResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    logs: list[dict[str, object]] = []
    ridge = mlp = None
    mlp_weight = 0.0
    months = sorted(panel.loc[panel["month"].ge(config.start), "month"].unique())
    for month_value in months:
        month = pd.Timestamp(month_value)
        if ridge is None or month.month in (1, 4, 7, 10):
            lower = month - pd.DateOffset(months=config.train_months)
            history = panel[
                panel["month"].lt(month)
                & panel["month"].ge(lower)
                & panel["eligible"]
                & panel["target"].notna()
                & panel["realization_month"].le(month)
            ]
            if len(history) < 10_000:
                continue
            split = history["month"].max() - pd.DateOffset(months=config.validation_months)
            train = sampled(
                history[history["month"].le(split)], 120_000,
                config.random_seed + month.year * 100 + month.month,
            )
            validation = history[history["month"].gt(split)].copy()
            x, y = matrix(train), train["target"].to_numpy(np.float32)
            xv, yv = matrix(validation), validation["target"].to_numpy(np.float32)
            trial = MLP(x.shape[1], seed=config.random_seed + month.year * 100 + month.month)
            best_epoch, mlp_ic_raw = trial.fit(x, y, validation=(xv, yv))
            ridge_trial = ridge_fit(x, y)
            mlp_weight, validation_ic = choose_blend(
                validation, ridge_predict(xv, ridge_trial), trial.predict(xv),
                config.mlp_weight_candidates,
            )

            full = sampled(
                history, 150_000,
                config.random_seed + 1_000_000 + month.year * 100 + month.month,
            )
            xf, yf = matrix(full), full["target"].to_numpy(np.float32)
            ridge = ridge_fit(xf, yf)
            mlp = MLP(xf.shape[1], seed=config.random_seed + month.year * 100 + month.month)
            mlp.fit(xf, yf, epochs=best_epoch, validation=None)
            logs.append({
                "refit_month": month,
                "history_start": history["month"].min(),
                "history_end": history["month"].max(),
                "max_label_realization_month": history["realization_month"].max(),
                "future_leakage_pass": history["realization_month"].max() <= month,
                "samples": len(full),
                "epochs": best_epoch,
                "mlp_validation_ic_raw": mlp_ic_raw,
                "monthly_validation_ic": validation_ic,
                "mlp_weight": mlp_weight,
            })

        test = panel[panel["month"].eq(month) & panel["eligible"]].copy()
        if test.empty or ridge is None or mlp is None:
            continue
        x_test = matrix(test)
        test["ridge_score"] = ridge_predict(x_test, ridge)
        test["mlp_score"] = mlp.predict(x_test)
        test["raw_score"] = (1 - mlp_weight) * _rank(test["ridge_score"]) + mlp_weight * _rank(test["mlp_score"])
        test["mlp_weight"] = mlp_weight
        rows.append(test[[
            "month", "realization_month", "Stkcd", "industry", "forward_return",
            "volatility_clean", "amount_clean", "ridge_score", "mlp_score",
            "raw_score", "mlp_weight",
        ]])

    signals = pd.concat(rows, ignore_index=True)
    log = pd.DataFrame(logs)
    if log.empty or not log["future_leakage_pass"].all():
        raise RuntimeError("Optimized signal generation failed the future-leakage audit")
    return signals, log


def smooth_scores(signals: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Causal EWMA: each row uses only its current and earlier stock scores."""
    ordered = signals.sort_values(["Stkcd", "month"]).copy()
    ordered["smoothed_score"] = ordered.groupby("Stkcd", sort=False)["raw_score"].transform(
        lambda values: values.ewm(alpha=weight, adjust=False).mean()
    )
    return ordered.sort_values(["month", "Stkcd"])


def build_portfolio(signals: pd.DataFrame, top_fraction: float, exit_fraction: float,
                    smoothing_weight: float, cost_bps: float) -> pd.DataFrame:
    panel = smooth_scores(signals, smoothing_weight)
    previous: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for signal_month, group in panel.groupby("month", sort=True):
        group = group.dropna(subset=["forward_return", "smoothed_score"]).copy()
        if group.empty:
            continue
        count = max(1, int(np.ceil(len(group) * top_fraction)))
        exit_count = max(count, int(np.ceil(len(group) * exit_fraction)))
        current_names = set(group["Stkcd"])
        permitted = set(group.nlargest(exit_count, "smoothed_score")["Stkcd"])
        retained = set(previous) & current_names & permitted
        additions = group[~group["Stkcd"].isin(retained)].nlargest(
            max(0, count - len(retained)), "smoothed_score"
        )
        chosen = pd.concat([
            group[group["Stkcd"].isin(retained)], additions,
        ]).drop_duplicates("Stkcd").nlargest(count, "smoothed_score")
        weights = dict(zip(chosen["Stkcd"], np.repeat(1 / len(chosen), len(chosen))))
        union = set(previous) | set(weights)
        stock_l1 = sum(
            abs(weights.get(code, 0.0) - previous.get(code, 0.0)) for code in union
        )
        prior_cash = 1.0 - sum(previous.values())
        current_cash = 1.0 - sum(weights.values())
        turnover = 0.5 * (stock_l1 + abs(current_cash - prior_cash))
        realized = chosen.set_index("Stkcd")["forward_return"]
        gross_return = sum(weight_value * realized.loc[code] for code, weight_value in weights.items())
        rows.append({
            "signal_month": signal_month,
            "month": pd.Timestamp(signal_month) + pd.offsets.MonthEnd(1),
            "gross_return": gross_return,
            "turnover": turnover,
            "holdings": len(weights),
            "smoothing_weight": smoothing_weight,
            "exit_fraction": exit_fraction,
        })
        previous = weights
    result = pd.DataFrame(rows)
    result["net_return"] = result["gross_return"] - result["turnover"] * cost_bps / 10_000
    return result


def period_metric(result: pd.DataFrame, start: str, end: str, cost_bps: float) -> dict[str, float]:
    selected = result[result["month"].between(start, end)].copy()
    returns = selected["gross_return"] - selected["turnover"] * cost_bps / 10_000
    values = metric(returns)
    values["mean_turnover"] = float(selected["turnover"].mean())
    return values


def signal_for_model(signals: pd.DataFrame, model: str) -> pd.DataFrame:
    scored = signals.copy()
    if model == "ridge":
        source = "ridge_score"
    elif model == "mlp":
        source = "mlp_score"
    elif model == "dynamic_blend":
        return scored
    else:
        raise ValueError(f"Unknown signal model: {model}")
    scored["raw_score"] = scored.groupby("month", observed=True)[source].rank(pct=True) - 0.5
    return scored


def select_parameters(signals: pd.DataFrame, config: ResearchConfig) -> tuple[dict[str, object], pd.DataFrame]:
    records: list[dict[str, object]] = []
    for signal_model in config.signal_model_candidates:
        model_signals = signal_for_model(signals, signal_model)
        for smoothing in config.smoothing_candidates:
            for exit_fraction in config.exit_candidates:
                result = build_portfolio(
                    model_signals, config.top_fraction, exit_fraction, smoothing,
                    config.selection_cost_bps,
                )
                development = period_metric(
                    result, config.start, config.development_end, config.selection_cost_bps
                )
                selection = period_metric(
                    result, config.selection_start, config.selection_end,
                    config.selection_cost_bps,
                )
                records.append({
                    "signal_model": signal_model,
                    "smoothing_weight": smoothing,
                    "exit_fraction": exit_fraction,
                    "development_sharpe_50bp": development["sharpe_rf0"],
                    "selection_sharpe_50bp": selection["sharpe_rf0"],
                    "development_annual_return_50bp": development["annual_return"],
                    "selection_annual_return_50bp": selection["annual_return"],
                    "development_turnover": development["mean_turnover"],
                    "selection_turnover": selection["mean_turnover"],
                })
    table = pd.DataFrame(records)
    # Conservative objective: reward both periods and lower turnover.  No 2024-2025
    # field is available to this function, which is verified by a unit test.
    table["selection_objective"] = (
        table[["development_sharpe_50bp", "selection_sharpe_50bp"]].mean(axis=1)
        - 0.10 * table["selection_turnover"]
    )
    positive = table[
        table["development_annual_return_50bp"].gt(0)
        & table["selection_annual_return_50bp"].gt(0)
    ]
    candidates = positive if not positive.empty else table
    best = candidates.sort_values(
        ["selection_objective", "selection_turnover"], ascending=[False, True]
    ).iloc[0]
    chosen: dict[str, object] = {
        "signal_model": str(best["signal_model"]),
        "smoothing_weight": float(best["smoothing_weight"]),
        "exit_fraction": float(best["exit_fraction"]),
    }
    table["selected"] = (
        table["signal_model"].eq(chosen["signal_model"])
        & table["smoothing_weight"].eq(chosen["smoothing_weight"])
        & table["exit_fraction"].eq(chosen["exit_fraction"])
    )
    return chosen, table


def volatility_control(result: pd.DataFrame, target: float, minimum: float,
                       cost_bps: float) -> pd.DataFrame:
    """Scale exposure using only returns realized before the current month."""
    controlled = result.copy()
    trailing = controlled["gross_return"].rolling(12, min_periods=6).std().shift(1) * np.sqrt(12)
    controlled["exposure"] = (target / trailing).clip(minimum, 1.0).fillna(1.0)
    prior_exposure = controlled["exposure"].shift(1).fillna(0.0)
    membership_turnover = controlled["turnover"] * controlled[["exposure"]].join(
        prior_exposure.rename("prior")
    ).min(axis=1)
    exposure_turnover = (controlled["exposure"] - prior_exposure).abs()
    controlled["scaled_turnover"] = membership_turnover + exposure_turnover
    controlled["scaled_gross_return"] = controlled["exposure"] * controlled["gross_return"]
    controlled["scaled_net_return"] = (
        controlled["scaled_gross_return"] - controlled["scaled_turnover"] * cost_bps / 10_000
    )
    return controlled


def cost_stress(result: pd.DataFrame, scaled: bool = False) -> pd.DataFrame:
    gross = result["scaled_gross_return"] if scaled else result["gross_return"]
    turnover = result["scaled_turnover"] if scaled else result["turnover"]
    rows = []
    for cost in (0, 20, 50, 100, 150, 200):
        rows.append({"cost_bps": cost, **metric(gross - turnover * cost / 10_000)})
    return pd.DataFrame(rows)


def alpha_diagnostics(signals: pd.DataFrame, chosen: dict[str, object],
                      config: ResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report monthly industry-relative IC and selected-signal decile monotonicity."""
    work = signals.copy()
    work["target"] = work["forward_return"].groupby(
        [work["month"], work["industry"]], dropna=False
    ).rank(pct=True)
    work["target"] = work["target"].fillna(
        work["forward_return"].groupby(work["month"]).rank(pct=True)
    ) - 0.5
    work["ridge"] = work.groupby("month", observed=True)["ridge_score"].rank(pct=True)
    work["mlp"] = work.groupby("month", observed=True)["mlp_score"].rank(pct=True)
    work["dynamic_blend"] = work["raw_score"]
    selected = signal_for_model(work, str(chosen["signal_model"]))
    selected = smooth_scores(selected, float(chosen["smoothing_weight"]))
    work = work.merge(
        selected[["month", "Stkcd", "smoothed_score"]],
        on=["month", "Stkcd"], how="left",
    )
    periods = {
        "development": (config.start, config.development_end),
        "selection": (config.selection_start, config.selection_end),
        "retrospective_test": (config.retrospective_start, config.retrospective_end),
        "full": (config.start, config.retrospective_end),
    }
    rows: list[dict[str, object]] = []
    for period, (start, end) in periods.items():
        subset = work[work["month"].between(start, end)]
        for model, column in [
            ("ridge", "ridge"), ("mlp", "mlp"),
            ("dynamic_blend", "dynamic_blend"),
            ("selected_smoothed", "smoothed_score"),
        ]:
            monthly = subset.groupby("month", observed=True).apply(
                lambda group: group[[column, "target"]].corr(method="spearman").iloc[0, 1]
                if group[[column, "target"]].dropna().shape[0] >= 30 else np.nan,
                include_groups=False,
            ).dropna()
            rows.append({
                "period": period,
                "signal": model,
                "mean_monthly_rank_ic": monthly.mean(),
                "annualized_icir": monthly.mean() / monthly.std() * np.sqrt(12)
                if monthly.std() else np.nan,
                "positive_ic_month_share": monthly.gt(0).mean(),
                "months": len(monthly),
            })

    work["decile"] = np.ceil(
        work.groupby("month", observed=True)["smoothed_score"].rank(pct=True) * 10
    ).clip(1, 10).astype("Int64")
    decile = work.groupby(["month", "decile"], observed=True)["forward_return"].mean().reset_index()
    decile_rows: list[pd.DataFrame] = []
    for period, (start, end) in periods.items():
        averages = decile[decile["month"].between(start, end)].groupby(
            "decile", observed=True
        )["forward_return"].mean().reset_index()
        averages.insert(0, "period", period)
        monotonicity = averages[["decile", "forward_return"]].corr(
            method="spearman"
        ).iloc[0, 1]
        averages["decile_monotonicity"] = monotonicity
        decile_rows.append(averages)
    return pd.DataFrame(rows), pd.concat(decile_rows, ignore_index=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=OUT / "monthly_panel.csv.gz")
    parser.add_argument("--reuse-signals", action="store_true")
    args = parser.parse_args()
    config = ResearchConfig()
    OUT.mkdir(exist_ok=True)
    signal_cache = OUT / "optimized_signal_panel.csv.gz"

    panel = prepare_clean(args.panel)
    cleaning_report(panel).to_csv(OUT / "optimized_data_quality.csv", index=False)
    if args.reuse_signals and signal_cache.exists():
        signals = pd.read_csv(
            signal_cache, parse_dates=["month", "realization_month"],
            dtype={"Stkcd": "string"}, low_memory=False,
        )
        logs = pd.read_csv(OUT / "optimized_model_log.csv", parse_dates=[
            "refit_month", "history_start", "history_end", "max_label_realization_month"
        ])
    else:
        signals, logs = generate_signals(panel, config)
        signals.to_csv(signal_cache, index=False, compression="gzip")
    logs.to_csv(OUT / "optimized_model_log.csv", index=False)
    logs[[
        "refit_month", "history_start", "history_end",
        "max_label_realization_month", "future_leakage_pass",
    ]].to_csv(OUT / "optimized_leakage_audit.csv", index=False)

    chosen, selection = select_parameters(signals, config)
    selection.to_csv(OUT / "optimized_parameter_selection.csv", index=False)
    selected_signals = signal_for_model(signals, str(chosen["signal_model"]))
    result = build_portfolio(
        selected_signals, config.top_fraction, float(chosen["exit_fraction"]),
        float(chosen["smoothing_weight"]), config.report_cost_bps,
    )
    controlled = volatility_control(
        result, config.volatility_target, config.minimum_exposure,
        config.report_cost_bps,
    )
    controlled.to_csv(OUT / "optimized_backtest.csv", index=False)

    ridge = pd.read_csv(OUT / "deep_learning_backtest.csv", parse_dates=["month"])
    benchmark = ridge[["month", "net_return", "benchmark_return"]].rename(
        columns={"net_return": "first_version_return"}
    )
    controlled = controlled.merge(benchmark, on="month", how="left")
    metrics = pd.DataFrame([
        {"series": "optimized_fully_invested", **metric(controlled["net_return"])},
        {"series": "optimized_volatility_control", **metric(controlled["scaled_net_return"])},
        {"series": "first_version_mlp_ridge", **metric(controlled["first_version_return"])},
        {"series": "benchmark", **metric(controlled["benchmark_return"])},
    ])
    metrics.to_csv(OUT / "optimized_metrics.csv", index=False)

    stress = cost_stress(controlled, scaled=False).assign(series="optimized_fully_invested")
    stress = pd.concat([
        stress,
        cost_stress(controlled, scaled=True).assign(series="optimized_volatility_control"),
    ], ignore_index=True)
    stress.to_csv(OUT / "optimized_cost_stress.csv", index=False)

    first_stress = pd.read_csv(OUT / "deep_learning_cost_stress.csv")
    comparison = pd.DataFrame([
        {
            "series": "optimized_clean_mlp",
            "annual_return_20bp": metrics.loc[
                metrics["series"].eq("optimized_fully_invested"), "annual_return"
            ].iloc[0],
            "sharpe_20bp": metrics.loc[
                metrics["series"].eq("optimized_fully_invested"), "sharpe_rf0"
            ].iloc[0],
            "max_drawdown_20bp": metrics.loc[
                metrics["series"].eq("optimized_fully_invested"), "max_drawdown"
            ].iloc[0],
            "mean_monthly_turnover": controlled["turnover"].mean(),
            "sharpe_50bp": stress.loc[
                stress["series"].eq("optimized_fully_invested")
                & stress["cost_bps"].eq(50), "sharpe_rf0"
            ].iloc[0],
            "sharpe_100bp": stress.loc[
                stress["series"].eq("optimized_fully_invested")
                & stress["cost_bps"].eq(100), "sharpe_rf0"
            ].iloc[0],
        },
        {
            "series": "first_version_mlp_ridge",
            "annual_return_20bp": metrics.loc[
                metrics["series"].eq("first_version_mlp_ridge"), "annual_return"
            ].iloc[0],
            "sharpe_20bp": metrics.loc[
                metrics["series"].eq("first_version_mlp_ridge"), "sharpe_rf0"
            ].iloc[0],
            "max_drawdown_20bp": metrics.loc[
                metrics["series"].eq("first_version_mlp_ridge"), "max_drawdown"
            ].iloc[0],
            "mean_monthly_turnover": ridge["turnover"].mean(),
            "sharpe_50bp": first_stress.loc[
                first_stress["cost_bps"].eq(50), "sharpe_rf0"
            ].iloc[0],
            "sharpe_100bp": first_stress.loc[
                first_stress["cost_bps"].eq(100), "sharpe_rf0"
            ].iloc[0],
        },
    ])
    comparison.to_csv(OUT / "optimized_comparison.csv", index=False)

    diagnostics, deciles = alpha_diagnostics(signals, chosen, config)
    diagnostics.to_csv(OUT / "optimized_alpha_diagnostics.csv", index=False)
    deciles.to_csv(OUT / "optimized_decile_returns.csv", index=False)

    subperiod_rows = []
    periods = {
        "development": (config.start, config.development_end),
        "selection": (config.selection_start, config.selection_end),
        "retrospective_test": (config.retrospective_start, config.retrospective_end),
    }
    for name, (start, end) in periods.items():
        subset = controlled[controlled["month"].between(start, end)]
        for series, column in [
            ("optimized_fully_invested", "net_return"),
            ("optimized_volatility_control", "scaled_net_return"),
            ("first_version_mlp_ridge", "first_version_return"),
        ]:
            subperiod_rows.append({"period": name, "series": series, **metric(subset[column])})
    pd.DataFrame(subperiod_rows).to_csv(OUT / "optimized_subperiod.csv", index=False)

    lock = {
        "status": "frozen_after_development_and_selection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "true_forward_start": "2026-09-01",
        "retrospective_test_not_used_for_selection": True,
        "selected_parameters": chosen,
        "config": asdict(config),
        "code_sha256": {
            "optimized_quant.py": sha256(Path(__file__)),
            "data_cleaning.py": sha256(ROOT / "data_cleaning.py"),
            "deep_learning_quant.py": sha256(ROOT / "deep_learning_quant.py"),
        },
    }
    (OUT / "optimized_research_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Selected parameters:", chosen)
    print(metrics.to_string(index=False))
    print(stress.to_string(index=False))


if __name__ == "__main__":
    main()
