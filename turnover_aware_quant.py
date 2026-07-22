from __future__ import annotations

"""Turnover-aware Ridge research with a frozen 2026 forward configuration."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_quant import OUT, linear_design, monthly_rank_ic, prepare_advanced
from ml_quant import fit as ridge_fit
from ml_quant import metric
from ml_quant import predict as ridge_predict


SMOOTHING_GRID = (1.00, 0.75, 0.50)
EXIT_GRID = (0.20, 0.25)
HURDLE_GRID = (1.0, 1.5, 2.0)
FIXED_COST_GRID_BPS = (0, 20, 50, 100, 150, 200)
CAPITAL_GRID = (10_000_000, 50_000_000, 100_000_000)
SOURCE_FILES = (
    "data_cleaning.py", "quant_project.py", "advanced_quant.py",
    "turnover_aware_quant.py", "turnover_aware_robustness.py",
    "test_turnover_aware_quant.py",
)


@dataclass(frozen=True)
class Config:
    formation_start: str = "2014-01-01"
    development_start: str = "2018-01-01"
    selection_start: str = "2022-01-01"
    retrospective_start: str = "2024-01-01"
    retrospective_end: str = "2025-12-31"
    true_forward_start: str = "2026-01-01"
    train_months: int = 84
    validation_months: int = 12
    refit_months: int = 3
    max_ridge_samples: int = 150_000
    top_fraction: float = 0.10
    smoothing_weight: float = 0.75
    exit_fraction: float = 0.25
    hurdle_multiple: float = 1.5
    fixed_cost_bps: float = 20.0
    base_cost: float = 0.0020
    impact_coefficient: float = 0.10
    impact_ratio_floor: float = 0.0
    impact_ratio_cap: float = 0.25
    execution_days: int = 5
    portfolio_capital_rmb: float = 50_000_000
    max_adv_participation_5d: float = 0.10
    seed: int = 20260722


def period_label(month: pd.Timestamp, config: Config) -> str:
    month = pd.Timestamp(month)
    if month < pd.Timestamp(config.development_start):
        return "formation"
    if month < pd.Timestamp(config.selection_start):
        return "development"
    if month < pd.Timestamp(config.retrospective_start):
        return "selection"
    if month < pd.Timestamp(config.true_forward_start):
        return "retrospective_test"
    return "true_forward"


def deterministic_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.sample(maximum, random_state=seed).sort_values("month")


def smooth_scores(raw_scores: pd.Series, previous_month: dict[str, float], weight: float) -> pd.Series:
    """Use only the current raw score and the immediately preceding smoothed score."""
    previous = raw_scores.index.to_series().map(previous_month)
    smoothed = weight * raw_scores + (1.0 - weight) * previous
    return smoothed.where(previous.notna(), raw_scores)


def estimate_order_cost(order_value: float, adv: float, volatility: float, config: Config) -> dict[str, float]:
    """Transparent liquidity proxy; it is not a calibrated market-impact function."""
    order_value = max(float(np.nan_to_num(order_value, nan=0.0)), 0.0)
    adv = float(np.nan_to_num(adv, nan=0.0))
    volatility = max(float(np.nan_to_num(volatility, nan=0.0)), 0.0)
    if order_value == 0:
        participation = 0.0
    elif adv <= 0:
        participation = config.impact_ratio_cap
    else:
        participation = order_value / (adv * config.execution_days)
    clipped = float(np.clip(participation, config.impact_ratio_floor, config.impact_ratio_cap))
    base = config.base_cost if order_value > 0 else 0.0
    impact = config.impact_coefficient * volatility * np.sqrt(clipped)
    return {
        "estimated_base_cost": float(max(base, 0.0)),
        "estimated_impact_cost": float(max(impact, 0.0)),
        "estimated_total_cost": float(max(base + impact, 0.0)),
        "ADV_participation_5d": float(max(participation, 0.0)),
    }


def true_weight_turnover(current: dict[str, float], previous: dict[str, float]) -> float:
    if not previous:
        return 1.0 if current else 0.0
    names = set(current) | set(previous)
    return 0.5 * sum(abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in names)


def _industry_adjusted_return(frame: pd.DataFrame) -> pd.Series:
    industry_mean = frame.groupby(["month", "industry"], dropna=False)["forward_return"].transform("mean")
    market_mean = frame.groupby("month")["forward_return"].transform("mean")
    return frame["forward_return"] - industry_mean.fillna(market_mean)


def _alpha_calibration(prediction: np.ndarray, validation: pd.DataFrame) -> tuple[float, float]:
    y = _industry_adjusted_return(validation).to_numpy(float)
    x = np.asarray(prediction, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 100 or np.var(x[valid]) < 1e-12:
        return 0.0, 0.0
    beta = float(np.cov(x[valid], y[valid], ddof=0)[0, 1] / np.var(x[valid]))
    intercept = float(y[valid].mean() - beta * x[valid].mean())
    return intercept, beta


def generate_ridge_scores(panel: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one leakage-safe Ridge score panel used by every turnover rule."""
    rows: list[pd.DataFrame] = []
    logs: list[dict[str, object]] = []
    fitted = None
    refit_counter = 0
    start = pd.Timestamp(config.development_start)
    end = pd.Timestamp(config.retrospective_end)
    months = sorted(pd.Timestamp(m) for m in panel.loc[panel["month"].between(start, end), "month"].unique())
    for month in months:
        if fitted is None or refit_counter >= config.refit_months:
            lower = month - pd.DateOffset(months=config.train_months)
            history = panel[
                (panel["month"] < month) & (panel["month"] >= lower)
                & panel["eligible"] & panel["target"].notna()
            ].copy()
            if len(history) < 10_000:
                continue
            validation_start = history["month"].max() - pd.DateOffset(months=config.validation_months)
            train = history[history["month"] <= validation_start]
            validation = history[history["month"] > validation_start]
            train = deterministic_sample(train, config.max_ridge_samples, config.seed + month.year * 100 + month.month)
            x_train, medians = linear_design(train)
            y_train = train["target"].to_numpy(float)
            x_validation, _ = linear_design(validation, medians)
            candidates = []
            for alpha in (10.0, 100.0, 1000.0, 10000.0):
                model = ridge_fit(x_train, y_train, alpha)
                prediction = ridge_predict(x_validation, model)
                score = monthly_rank_ic(prediction, validation["target"].to_numpy(float), validation["month"].to_numpy())
                candidates.append((score, alpha, prediction))
            validation_ic, alpha, validation_prediction = max(candidates, key=lambda item: item[0])
            intercept, beta = _alpha_calibration(validation_prediction, validation)
            full = deterministic_sample(history, config.max_ridge_samples, config.seed + 17 + month.year * 100 + month.month)
            x_full, medians = linear_design(full)
            model = ridge_fit(x_full, full["target"].to_numpy(float), alpha)
            fitted = (model, medians, intercept, beta, alpha)
            logs.append({
                "refit_month": month, "history_start": history["month"].min(),
                "history_end": history["month"].max(), "history_samples": len(history),
                "ridge_alpha": alpha, "validation_ic": validation_ic,
                "alpha_intercept": intercept, "alpha_beta": beta,
                "maximum_feature_month": history["month"].max(),
            })
            refit_counter = 0

        test = panel[(panel["month"] == month) & panel["eligible"] & panel["forward_return"].notna()].copy()
        if fitted is None or test.empty:
            continue
        model, medians, intercept, beta, alpha = fitted
        x_test, _ = linear_design(test, medians)
        test["raw_score"] = ridge_predict(x_test, model)
        test["predicted_alpha"] = intercept + beta * test["raw_score"]
        test["signal_month"] = month
        test["realization_month"] = month + pd.offsets.MonthEnd(1)
        test["period"] = period_label(month, config)
        test["maximum_feature_month"] = month
        test["ridge_alpha"] = alpha
        rows.append(test[[
            "signal_month", "realization_month", "period", "maximum_feature_month",
            "Stkcd", "industry", "forward_return", "target", "amount_clean",
            "volatility_clean", "raw_score", "predicted_alpha", "ridge_alpha",
        ]].rename(columns={"Stkcd": "stock_code"}))
        refit_counter += 1
    return pd.concat(rows, ignore_index=True), pd.DataFrame(logs)


def _better_sort(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    return frame.sort_values("smoothed_score", ascending=(side == "short"))


def _replacement_gain(candidate: pd.Series, current: pd.Series, side: str) -> float:
    if side == "long":
        return float(candidate["predicted_alpha"] - current["predicted_alpha"])
    return float(current["predicted_alpha"] - candidate["predicted_alpha"])


def _attempt_record(candidate: pd.Series, label: str, weight_hint: float, cost: dict[str, float],
                    gain: float, allowed: bool, reason: str) -> dict[str, object]:
    return {
        "stock_code": candidate.name, "side": f"{label}_replacement_candidate",
        "old_weight": 0.0, "target_weight": weight_hint,
        "proposed_traded_weight": weight_hint, "traded_weight": 0.0,
        "raw_score": candidate["raw_score"], "smoothed_score": candidate["smoothed_score"],
        "predicted_alpha": candidate["predicted_alpha"], **cost,
        "expected_alpha_gain": gain, "trade_allowed": allowed,
        "rejection_reason": reason if not allowed else "accepted_replacement",
        "average_daily_amount": candidate["amount_clean"],
    }


def select_buffered_names(frame: pd.DataFrame, previous_weights: dict[str, float], side: str,
                          target_count: int, weight_hint: float, capital: float, config: Config,
                          label: str, enforce_adv_limit: bool) -> tuple[set[str], dict[str, float], list[dict[str, object]]]:
    """Apply entry tail, wider exit buffer, and a cost-aware replacement hurdle."""
    if frame.empty or target_count <= 0:
        return set(), {}, []
    frame = frame.copy()
    percentile = frame["smoothed_score"].rank(pct=True)
    if side == "long":
        entry = percentile >= 1.0 - config.top_fraction
        buffer = percentile >= 1.0 - config.exit_fraction
    else:
        entry = percentile <= config.top_fraction
        buffer = percentile <= config.exit_fraction
    previous = set(previous_weights)
    available_previous = previous & set(frame.index)
    retained = available_previous & set(frame.index[buffer])
    replaceable = available_previous - retained
    selected = set(available_previous)
    candidates = _better_sort(frame.loc[entry & ~frame.index.to_series().isin(selected)], side)
    candidate_names = list(candidates.index)
    attempts: list[dict[str, object]] = []
    gains: dict[str, float] = {}

    if not previous:
        selected.clear()
    else:
        worst = _better_sort(frame.loc[list(replaceable)], "short" if side == "long" else "long")
        for current_name in worst.index:
            while candidate_names:
                candidate_name = candidate_names.pop(0)
                if candidate_name not in selected:
                    break
            else:
                break
            candidate = frame.loc[candidate_name]
            current = frame.loc[current_name]
            gain = _replacement_gain(candidate, current, side)
            buy_cost = estimate_order_cost(capital * weight_hint, candidate["amount_clean"], candidate["volatility_clean"], config)
            old_weight = previous_weights.get(current_name, weight_hint)
            sell_cost = estimate_order_cost(capital * old_weight, current["amount_clean"], current["volatility_clean"], config)
            round_trip = buy_cost["estimated_total_cost"] + sell_cost["estimated_total_cost"]
            adv_ok = (not enforce_adv_limit) or buy_cost["ADV_participation_5d"] <= config.max_adv_participation_5d
            hurdle_ok = gain > config.hurdle_multiple * round_trip
            allowed = bool(adv_ok and hurdle_ok)
            reason = "" if allowed else ("adv_limit" if not adv_ok else "alpha_gain_below_hurdle")
            attempts.append(_attempt_record(candidate, label, weight_hint, buy_cost, gain, allowed, reason))
            if allowed:
                selected.remove(current_name)
                selected.add(candidate_name)
                gains[current_name] = gain
                gains[candidate_name] = gain

    remaining = [name for name in candidate_names if name not in selected]
    if not previous:
        remaining = list(_better_sort(frame.loc[entry], side).index)
    for candidate_name in remaining:
        if len(selected) >= target_count:
            break
        candidate = frame.loc[candidate_name]
        cost = estimate_order_cost(capital * weight_hint, candidate["amount_clean"], candidate["volatility_clean"], config)
        if enforce_adv_limit and cost["ADV_participation_5d"] > config.max_adv_participation_5d:
            attempts.append(_attempt_record(candidate, label, weight_hint, cost, np.nan, False, "adv_limit"))
            continue
        selected.add(candidate_name)
        gains[candidate_name] = np.nan

    if len(selected) > target_count:
        ordered = list(_better_sort(frame.loc[list(selected)], side).index)
        selected = set(ordered[:target_count])
    return selected, gains, attempts


def _row_info(name: str, frame: pd.DataFrame, last_info: dict[str, dict[str, float]]) -> dict[str, float]:
    if name in frame.index:
        row = frame.loc[name]
        return {
            "amount_clean": float(row["amount_clean"]), "volatility_clean": float(row["volatility_clean"]),
            "raw_score": float(row["raw_score"]), "smoothed_score": float(row["smoothed_score"]),
            "predicted_alpha": float(row["predicted_alpha"]),
        }
    return last_info.get(name, {
        "amount_clean": np.nan, "volatility_clean": np.nan, "raw_score": np.nan,
        "smoothed_score": np.nan, "predicted_alpha": np.nan,
    })


def actual_trade_records(previous: dict[str, float], current: dict[str, float], frame: pd.DataFrame,
                         last_info: dict[str, dict[str, float]], label: str, gains: dict[str, float],
                         capital: float, config: Config) -> list[dict[str, object]]:
    records = []
    for name in sorted(set(previous) | set(current)):
        old_weight = previous.get(name, 0.0)
        target_weight = current.get(name, 0.0)
        delta = target_weight - old_weight
        if abs(delta) < 1e-12:
            continue
        info = _row_info(name, frame, last_info)
        cost = estimate_order_cost(capital * abs(delta), info["amount_clean"], info["volatility_clean"], config)
        action = "buy" if delta > 0 else "sell"
        records.append({
            "stock_code": name, "side": f"{label}_{action}", "old_weight": old_weight,
            "target_weight": target_weight, "proposed_traded_weight": abs(delta),
            "traded_weight": abs(delta), "raw_score": info["raw_score"],
            "smoothed_score": info["smoothed_score"], "predicted_alpha": info["predicted_alpha"],
            **cost, "expected_alpha_gain": gains.get(name, np.nan), "trade_allowed": True,
            "rejection_reason": "", "average_daily_amount": info["amount_clean"],
        })
    return records


def _weights_from_names(names: set[str]) -> dict[str, float]:
    if not names:
        return {}
    weight = 1.0 / len(names)
    return {name: weight for name in names}


def _portfolio_return(weights: dict[str, float], frame: pd.DataFrame) -> float:
    if not weights:
        return np.nan
    return float(sum(weight * frame.loc[name, "forward_return"] for name, weight in weights.items()))


def _proxy_drag(records: list[dict[str, object]], prefix: str) -> float:
    return float(sum(
        float(row["traded_weight"]) * float(row["estimated_total_cost"])
        for row in records if bool(row["trade_allowed"]) and str(row["side"]).startswith(prefix)
    ))


def simulate_portfolios(score_panel: pd.DataFrame, config: Config,
                        collect_trade_log: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    previous_smoothed: dict[str, float] = {}
    previous_long: dict[str, float] = {}
    previous_diag_long: dict[str, float] = {}
    previous_diag_short: dict[str, float] = {}
    last_info: dict[str, dict[str, float]] = {}

    for signal_month, raw_frame in score_panel.groupby("signal_month", sort=True):
        frame = raw_frame.copy().set_index("stock_code", drop=True)
        frame["smoothed_score"] = smooth_scores(frame["raw_score"], previous_smoothed, config.smoothing_weight)
        previous_smoothed = frame["smoothed_score"].to_dict()
        target_count = max(1, int(np.ceil(len(frame) * config.top_fraction)))
        weight_hint = 1.0 / target_count
        long_names, long_gains, attempts = select_buffered_names(
            frame, previous_long, "long", target_count, weight_hint,
            config.portfolio_capital_rmb, config, "long_only", True,
        )
        long_weights = _weights_from_names(long_names)

        industry_groups = [(industry, group) for industry, group in frame.groupby("industry", dropna=False) if len(group) >= 10]
        diag_pairs: list[tuple[set[str], set[str], dict[str, float], dict[str, float], list[dict[str, object]]]] = []
        industry_weight_hint = 1.0 / max(len(industry_groups), 1)
        for _, group in industry_groups:
            count = max(1, int(np.floor(len(group) * config.top_fraction)))
            hint = industry_weight_hint / count
            prev_l = {name: weight for name, weight in previous_diag_long.items() if name in group.index}
            prev_s = {name: weight for name, weight in previous_diag_short.items() if name in group.index}
            long_set, long_gain, long_attempts = select_buffered_names(
                group, prev_l, "long", count, hint, config.portfolio_capital_rmb,
                config, "diagnostic_long", False,
            )
            short_set, short_gain, short_attempts = select_buffered_names(
                group, prev_s, "short", count, hint, config.portfolio_capital_rmb,
                config, "diagnostic_short", False,
            )
            if long_set and short_set:
                diag_pairs.append((long_set, short_set, long_gain, short_gain, long_attempts + short_attempts))

        diag_long_weights: dict[str, float] = {}
        diag_short_weights: dict[str, float] = {}
        diag_long_gains: dict[str, float] = {}
        diag_short_gains: dict[str, float] = {}
        if diag_pairs:
            industry_weight = 1.0 / len(diag_pairs)
            for long_set, short_set, long_gain, short_gain, pair_attempts in diag_pairs:
                diag_long_weights.update({name: industry_weight / len(long_set) for name in long_set})
                diag_short_weights.update({name: industry_weight / len(short_set) for name in short_set})
                diag_long_gains.update(long_gain)
                diag_short_gains.update(short_gain)
                attempts.extend(pair_attempts)

        orders = actual_trade_records(
            previous_long, long_weights, frame, last_info, "long_only", long_gains,
            config.portfolio_capital_rmb, config,
        )
        orders += actual_trade_records(
            previous_diag_long, diag_long_weights, frame, last_info, "diagnostic_long", diag_long_gains,
            config.portfolio_capital_rmb, config,
        )
        orders += actual_trade_records(
            previous_diag_short, diag_short_weights, frame, last_info, "diagnostic_short", diag_short_gains,
            config.portfolio_capital_rmb, config,
        )
        all_trade_records = orders + attempts
        if collect_trade_log:
            for record in all_trade_records:
                record.update({
                    "signal_month": signal_month,
                    "realization_month": raw_frame["realization_month"].iloc[0],
                })
                trade_rows.append(record)

        long_turnover = true_weight_turnover(long_weights, previous_long)
        diag_long_turnover = true_weight_turnover(diag_long_weights, previous_diag_long)
        diag_short_turnover = true_weight_turnover(diag_short_weights, previous_diag_short)
        long_short_turnover = diag_long_turnover + diag_short_turnover
        long_gross = _portfolio_return(long_weights, frame)
        long_short_gross = _portfolio_return(diag_long_weights, frame) - _portfolio_return(diag_short_weights, frame)
        raw_ic = frame["raw_score"].corr(frame["target"], method="spearman")
        smoothed_ic = frame["smoothed_score"].corr(frame["target"], method="spearman")

        percentile = frame.groupby("industry", dropna=False)["smoothed_score"].rank(pct=True)
        percentile = percentile.fillna(frame["smoothed_score"].rank(pct=True))
        decile = np.ceil(percentile * 10).clip(1, 10).astype(int)
        for number, group in frame.assign(decile=decile).groupby("decile"):
            decile_rows.append({
                "signal_month": signal_month, "realization_month": raw_frame["realization_month"].iloc[0],
                "period": raw_frame["period"].iloc[0], "decile": number,
                "return": group["forward_return"].mean(), "stocks": len(group),
            })

        long_proxy_drag = _proxy_drag(orders, "long_only")
        diagnostic_proxy_drag = _proxy_drag(orders, "diagnostic_long") + _proxy_drag(orders, "diagnostic_short")
        rows.append({
            "signal_month": signal_month, "month": raw_frame["realization_month"].iloc[0],
            "period": raw_frame["period"].iloc[0], "maximum_feature_month": raw_frame["maximum_feature_month"].max(),
            "gross_return": long_gross, "turnover": long_turnover,
            "net_return_20bps": long_gross - long_turnover * 20 / 10_000,
            "proxy_net_return": long_gross - long_proxy_drag,
            "holdings": len(long_weights), "raw_rank_ic": raw_ic, "rank_ic": smoothed_ic,
            "long_short_gross_return": long_short_gross,
            "diagnostic_long_turnover": diag_long_turnover,
            "diagnostic_short_turnover": diag_short_turnover,
            "long_short_turnover": long_short_turnover,
            "long_short_net_return_20bps": long_short_gross - long_short_turnover * 20 / 10_000,
            "long_short_proxy_net_return": long_short_gross - diagnostic_proxy_drag,
            "diagnostic_long_names": len(diag_long_weights),
            "diagnostic_short_names": len(diag_short_weights),
        })
        previous_long = long_weights
        previous_diag_long = diag_long_weights
        previous_diag_short = diag_short_weights
        last_info = {
            name: {
                "amount_clean": float(row["amount_clean"]), "volatility_clean": float(row["volatility_clean"]),
                "raw_score": float(row["raw_score"]), "smoothed_score": float(row["smoothed_score"]),
                "predicted_alpha": float(row["predicted_alpha"]),
            }
            for name, row in frame.iterrows()
        }
    return pd.DataFrame(rows), pd.DataFrame(decile_rows), pd.DataFrame(trade_rows)


def _sharpe(frame: pd.DataFrame, return_column: str, turnover_column: str, cost_bps: float) -> float:
    returns = frame[return_column] - frame[turnover_column] * cost_bps / 10_000
    return float(metric(returns)["sharpe_rf0"]) if returns.notna().sum() >= 3 else np.nan


def _decile_monotonicity(deciles: pd.DataFrame, period: str) -> float:
    average = deciles[deciles["period"] == period].groupby("decile")["return"].mean()
    return float(average.index.to_series().corr(average, method="spearman"))


def _mean_ic(frame: pd.DataFrame, period: str, column: str) -> float:
    return float(frame.loc[frame["period"] == period, column].mean())


def _baseline_selection_turnover(path: Path) -> float:
    baseline = pd.read_csv(path, parse_dates=["month"])
    signal_month = baseline["month"] - pd.offsets.MonthEnd(1)
    mask = signal_month.between("2022-01-01", "2023-12-31")
    return float(baseline.loc[mask, "long_short_turnover"].mean())


def parameter_selection(score_panel: pd.DataFrame, base_config: Config,
                        advanced_path: Path) -> tuple[Config, pd.DataFrame]:
    # The grid never receives retrospective rows.  Only the frozen winner is
    # subsequently run across 2024-2025 by main().
    selection_panel = score_panel[score_panel["period"].isin(["development", "selection"])].copy()
    baseline_turnover = _baseline_selection_turnover(advanced_path)
    raw_selection_ic = selection_panel[selection_panel["period"] == "selection"].groupby("signal_month").apply(
        lambda group: group["raw_score"].corr(group["target"], method="spearman"),
        include_groups=False,
    ).mean()
    rows = []
    for smoothing_weight in SMOOTHING_GRID:
        for exit_fraction in EXIT_GRID:
            for hurdle_multiple in HURDLE_GRID:
                config = replace(
                    base_config, smoothing_weight=smoothing_weight,
                    exit_fraction=exit_fraction, hurdle_multiple=hurdle_multiple,
                )
                backtest, deciles, _ = simulate_portfolios(selection_panel, config, collect_trade_log=False)
                development = backtest[backtest["period"] == "development"]
                selection = backtest[backtest["period"] == "selection"]
                row = {
                    "smoothing_weight": smoothing_weight, "exit_fraction": exit_fraction,
                    "hurdle_multiple": hurdle_multiple, "uses_retrospective_test": False,
                    "development_long_short_sharpe_50bps": _sharpe(
                        development, "long_short_gross_return", "long_short_turnover", 50,
                    ),
                    "selection_long_short_sharpe_20bps": _sharpe(
                        selection, "long_short_gross_return", "long_short_turnover", 20,
                    ),
                    "selection_long_short_sharpe_50bps": _sharpe(
                        selection, "long_short_gross_return", "long_short_turnover", 50,
                    ),
                    "selection_long_short_sharpe_100bps": _sharpe(
                        selection, "long_short_gross_return", "long_short_turnover", 100,
                    ),
                    "selection_long_turnover": selection["turnover"].mean(),
                    "selection_long_short_turnover": selection["long_short_turnover"].mean(),
                    "baseline_selection_long_short_turnover": baseline_turnover,
                    "turnover_reduction": 1.0 - selection["long_short_turnover"].mean() / baseline_turnover,
                    "selection_raw_ridge_ic": raw_selection_ic,
                    "selection_smoothed_ic": _mean_ic(backtest, "selection", "rank_ic"),
                    "selection_ic_drop": raw_selection_ic - _mean_ic(backtest, "selection", "rank_ic"),
                    "selection_decile_monotonicity": _decile_monotonicity(deciles, "selection"),
                }
                rows.append(row)
    table = pd.DataFrame(rows)
    index_map = {value: i for i, value in enumerate(SMOOTHING_GRID)}
    exit_map = {value: i for i, value in enumerate(EXIT_GRID)}
    hurdle_map = {value: i for i, value in enumerate(HURDLE_GRID)}
    neighbor_shares = []
    neighbor_counts = []
    for _, row in table.iterrows():
        distance = (
            (table["smoothing_weight"].map(index_map) - index_map[row["smoothing_weight"]]).abs()
            + (table["exit_fraction"].map(exit_map) - exit_map[row["exit_fraction"]]).abs()
            + (table["hurdle_multiple"].map(hurdle_map) - hurdle_map[row["hurdle_multiple"]]).abs()
        )
        neighbors = table[distance == 1]
        robust = (
            (neighbors["selection_long_short_sharpe_50bps"] > 0)
            & (neighbors["turnover_reduction"] >= 0.25)
            & (neighbors["selection_ic_drop"] <= 0.005)
        )
        neighbor_counts.append(len(neighbors))
        neighbor_shares.append(float(robust.mean()) if len(neighbors) else 0.0)
    table["adjacent_candidate_count"] = neighbor_counts
    table["robust_neighbor_share"] = neighbor_shares
    table["passes_constraints"] = (
        (table["selection_long_short_sharpe_20bps"] > 0)
        & (table["turnover_reduction"] >= 0.25)
        & (table["selection_ic_drop"] <= 0.005)
        & (table["selection_decile_monotonicity"] >= 0.70)
        & (table["robust_neighbor_share"] >= 0.50)
    )
    table["selection_objective"] = (
        table["selection_long_short_sharpe_50bps"]
        + 0.25 * table["development_long_short_sharpe_50bps"]
        - 0.10 * table["selection_long_short_turnover"]
    )
    eligible = table[table["passes_constraints"]]
    pool = eligible if not eligible.empty else table
    chosen_index = pool.sort_values(
        ["selection_objective", "robust_neighbor_share", "selection_long_short_turnover"],
        ascending=[False, False, True],
    ).index[0]
    table["selected"] = table.index == chosen_index
    chosen = table.loc[chosen_index]
    config = replace(
        base_config, smoothing_weight=float(chosen["smoothing_weight"]),
        exit_fraction=float(chosen["exit_fraction"]),
        hurdle_multiple=float(chosen["hurdle_multiple"]),
    )
    return config, table.sort_values("selection_objective", ascending=False)


def capacity_summary(trades: pd.DataFrame, config: Config) -> pd.DataFrame:
    allowed = trades[
        trades["trade_allowed"].astype(bool) & trades["side"].str.startswith("long_only")
        & trades["average_daily_amount"].notna() & trades["traded_weight"].gt(0)
    ].copy()
    rows = []
    for capital in CAPITAL_GRID:
        participation = capital * allowed["traded_weight"] / (
            allowed["average_daily_amount"] * config.execution_days
        )
        rows.append({
            "portfolio_capital_rmb": capital, "execution_days": config.execution_days,
            "median_participation": participation.median(), "p90_participation": participation.quantile(0.90),
            "p95_participation": participation.quantile(0.95), "p99_participation": participation.quantile(0.99),
            "trades_over_5pct_adv_share": (participation > 0.05).mean(),
            "trades_over_10pct_adv_share": (participation > 0.10).mean(),
        })
    return pd.DataFrame(rows)


def alpha_diagnostics(backtest: pd.DataFrame, deciles: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    ic = backtest["rank_ic"].dropna()
    long_short = backtest["long_short_net_return_20bps"].dropna()
    decile_average = deciles.groupby("decile")["return"].mean()
    selected = selection[selection["selected"]].iloc[0]
    rows = [
        {"metric": "mean_cross_sectional_ic", "value": ic.mean()},
        {"metric": "cross_sectional_icir_annualized", "value": ic.mean() / ic.std() * np.sqrt(12)},
        {"metric": "positive_ic_month_share", "value": (ic > 0).mean()},
        {"metric": "mean_raw_ridge_ic", "value": backtest["raw_rank_ic"].mean()},
        {"metric": "smoothed_minus_raw_ic", "value": (backtest["rank_ic"] - backtest["raw_rank_ic"]).mean()},
        {"metric": "decile_monotonicity_spearman", "value": decile_average.index.to_series().corr(decile_average, method="spearman")},
        {"metric": "long_short_monthly_t_stat", "value": long_short.mean() / (long_short.std() / np.sqrt(len(long_short)))},
        {"metric": "selected_neighbor_robust_share", "value": selected["robust_neighbor_share"]},
        {"metric": "selected_passes_constraints", "value": float(selected["passes_constraints"])},
    ]
    return pd.DataFrame(rows)


def metrics_table(backtest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"series": "turnover_aware_long_only_20bps", **metric(backtest["net_return_20bps"]),
         "avg_turnover": backtest["turnover"].mean()},
        {"series": "turnover_aware_long_only_proxy_cost", **metric(backtest["proxy_net_return"]),
         "avg_turnover": backtest["turnover"].mean()},
        {"series": "turnover_aware_industry_neutral_diagnostic_20bps", **metric(backtest["long_short_net_return_20bps"]),
         "avg_turnover": backtest["long_short_turnover"].mean()},
        {"series": "turnover_aware_industry_neutral_diagnostic_proxy_cost", **metric(backtest["long_short_proxy_net_return"]),
         "avg_turnover": backtest["long_short_turnover"].mean()},
    ])


def validate_no_future_features(backtest: pd.DataFrame) -> None:
    signal = pd.to_datetime(backtest["signal_month"])
    maximum_feature = pd.to_datetime(backtest["maximum_feature_month"])
    if maximum_feature.gt(signal).any():
        raise ValueError("Future-month feature detected in turnover-aware output")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_research_lock(config: Config, selected_row: dict[str, object], root: Path) -> dict[str, object]:
    hashes = {name: _sha256(root / name) for name in SOURCE_FILES if (root / name).exists()}
    return {
        "lock_version": 1,
        "status": "frozen_for_true_forward_from_2026",
        "frozen_config": asdict(config),
        "selected_parameters": {
            "smoothing_weight": config.smoothing_weight,
            "exit_fraction": config.exit_fraction,
            "hurdle_multiple": config.hurdle_multiple,
        },
        "data_intervals": {
            "formation": ["2014-01-01", "2017-12-31"],
            "development": ["2018-01-01", "2021-12-31"],
            "selection": ["2022-01-01", "2023-12-31"],
            "retrospective_test": ["2024-01-01", "2025-12-31"],
            "true_forward_start": "2026-01-01",
        },
        "retrospective_used_for_parameter_selection": False,
        "selection_passed_all_constraints": bool(selected_row.get("passes_constraints", False)),
        "selected_neighbor_robust_share": float(selected_row.get("robust_neighbor_share", np.nan)),
        "git_commit": git_commit(root),
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": hashes,
    }


def refresh_lock(root: Path) -> None:
    path = OUT / "research_lock.json"
    if not path.exists():
        raise FileNotFoundError("Run the full turnover-aware research before refreshing the lock")
    current = json.loads(path.read_text(encoding="utf-8"))
    config = Config(**current["frozen_config"])
    selected_row = {
        "passes_constraints": current["selection_passed_all_constraints"],
        "robust_neighbor_share": current["selected_neighbor_robust_share"],
    }
    path.write_text(json.dumps(build_research_lock(config, selected_row, root), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=OUT / "monthly_panel.csv.gz")
    parser.add_argument("--refresh-lock-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.refresh_lock_only:
        refresh_lock(root)
        return

    base_config = Config()
    panel = prepare_advanced(args.panel)
    score_panel, model_log = generate_ridge_scores(panel, base_config)
    selected_config, selection = parameter_selection(
        score_panel, base_config, OUT / "advanced_backtest.csv",
    )
    backtest, deciles, trades = simulate_portfolios(score_panel, selected_config, collect_trade_log=True)
    validate_no_future_features(backtest)
    selected_row = selection[selection["selected"]].iloc[0].to_dict()

    backtest.to_csv(OUT / "turnover_aware_backtest.csv", index=False)
    metrics_table(backtest).to_csv(OUT / "turnover_aware_metrics.csv", index=False)
    trades.to_csv(OUT / "turnover_aware_trade_log.csv", index=False)
    capacity_summary(trades, selected_config).to_csv(OUT / "turnover_aware_capacity.csv", index=False)
    selection.to_csv(OUT / "turnover_aware_parameter_selection.csv", index=False)
    deciles.to_csv(OUT / "turnover_aware_decile_returns.csv", index=False)
    alpha_diagnostics(backtest, deciles, selection).to_csv(
        OUT / "turnover_aware_alpha_diagnostics.csv", index=False,
    )
    model_log.to_csv(OUT / "turnover_aware_model_log.csv", index=False)
    (OUT / "research_lock.json").write_text(
        json.dumps(build_research_lock(selected_config, selected_row, root), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(selection[selection["selected"]].to_string(index=False))
    print(metrics_table(backtest).to_string(index=False))


if __name__ == "__main__":
    main()
