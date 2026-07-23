from __future__ import annotations

"""Compare equal, score-weighted and transparent cost-aware long portfolios."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_quant import OUT
from residual_common import (
    DataLimitations,
    RETROSPECTIVE_START,
    assert_selection_period_only,
    decile_labels,
    metric,
    true_weight_turnover,
    weighted_return,
)


TARGET_TYPES = ("original_1m", "residual_1m", "residual_3m", "fixed_combined")
PORTFOLIO_TYPES = ("equal_weight", "score_weight", "cost_aware")
LAMBDA_GRID = (0.5, 1.0, 2.0)
COST_GRID_BPS = (0, 20, 50, 100, 150, 200)
CAPITAL_GRID = (10_000_000.0, 50_000_000.0, 100_000_000.0)
SOURCE_FILES = (
    "data_cleaning.py",
    "advanced_quant.py",
    "residual_common.py",
    "residual_target_quant.py",
    "cost_aware_portfolio.py",
    "tradable_return_engine.py",
    "residual_robustness.py",
)


@dataclass(frozen=True)
class PortfolioConfig:
    target_type: str = "original_1m"
    portfolio_type: str = "cost_aware"
    lambda_risk: float = 1.0
    lambda_turnover: float = 1.0
    max_stock_weight: float = 0.02
    min_holdings: int = 80
    max_holdings: int = 200
    industry_absolute_cap: float = 0.20
    industry_deviation_cap: float = 0.05
    style_exposure_tolerance: float = 0.35
    max_cash_weight: float = 0.05
    portfolio_capital_rmb: float = 50_000_000.0
    max_adv_participation_5d: float = 0.10
    execution_days: int = 5
    base_cost: float = 0.0020
    impact_coefficient: float = 0.10
    impact_ratio_cap: float = 0.25
    seed: int = 20260723


def score_columns(target_type: str) -> tuple[str, str, str]:
    if target_type not in TARGET_TYPES:
        raise ValueError(f"unsupported target type: {target_type}")
    target_column = {
        "original_1m": "original_1m_target",
        "residual_1m": "residual_return_1m_target",
        "residual_3m": "residual_return_3m_target",
        "fixed_combined": "residual_return_1m_target",
    }[target_type]
    return f"{target_type}_score", f"{target_type}_alpha", target_column


def estimate_order_cost(
    traded_weight: float,
    amount: float,
    volatility: float,
    capital: float,
    config: PortfolioConfig,
) -> dict[str, float]:
    order_value = max(float(traded_weight), 0.0) * float(capital)
    adv = float(np.nan_to_num(amount, nan=0.0))
    volatility = max(float(np.nan_to_num(volatility, nan=0.0)), 0.0)
    if order_value == 0:
        participation = 0.0
    elif adv <= 0:
        participation = np.inf
    else:
        participation = order_value / (adv * config.execution_days)
    clipped = float(
        np.clip(
            np.nan_to_num(participation, nan=config.impact_ratio_cap, posinf=config.impact_ratio_cap),
            0.0,
            config.impact_ratio_cap,
        )
    )
    base = config.base_cost if order_value > 0 else 0.0
    impact = config.impact_coefficient * volatility * np.sqrt(clipped)
    total = max(base + impact, 0.0)
    return {
        "estimated_base_cost": float(max(base, 0.0)),
        "estimated_impact_cost": float(max(impact, 0.0)),
        "estimated_total_cost": float(total),
        "ADV_participation_5d": float(max(participation, 0.0)),
    }


def _capped_normalize(weights: pd.Series, cap: float) -> pd.Series:
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    if weights.sum() <= 0:
        return weights
    weights = weights / weights.sum()
    for _ in range(20):
        over = weights > cap + 1e-12
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights.loc[over] = cap
        under = ~over
        if not under.any() or weights.loc[under].sum() <= 0:
            break
        weights.loc[under] += excess * weights.loc[under] / weights.loc[under].sum()
    return weights / weights.sum()


def _industry_cap_weights(
    frame: pd.DataFrame,
    weights: pd.Series,
    config: PortfolioConfig,
) -> pd.Series:
    pool_share = frame["industry"].value_counts(normalize=True, dropna=False)
    industry = frame["industry"].fillna("UNKNOWN")
    caps = {
        name: min(
            config.industry_absolute_cap,
            float(pool_share.get(name if name != "UNKNOWN" else np.nan, 0.0))
            + config.industry_deviation_cap,
        )
        for name in industry.unique()
    }
    # If many tiny industries make the stated caps infeasible, allocate the
    # residual cap proportionally and record the pool-relative reference later.
    if sum(caps.values()) < 1.0:
        scale = 1.0 / max(sum(caps.values()), 1e-12)
        caps = {
            key: min(config.industry_absolute_cap, value * scale)
            for key, value in caps.items()
        }
    result = weights.copy()
    for _ in range(30):
        group_weight = result.groupby(industry).sum()
        over = [
            name for name, value in group_weight.items() if value > caps.get(name, 1.0) + 1e-10
        ]
        if not over:
            break
        freed = 0.0
        for name in over:
            indices = industry.index[industry == name]
            old = float(result.loc[indices].sum())
            target = caps[name]
            result.loc[indices] *= target / old
            freed += old - target
        eligible = industry.index[~industry.isin(over)]
        if len(eligible) == 0 or result.loc[eligible].sum() <= 0:
            break
        result.loc[eligible] += freed * result.loc[eligible] / result.loc[eligible].sum()
    return _capped_normalize(result, config.max_stock_weight)


def _style_tilt(
    frame: pd.DataFrame, weights: pd.Series, config: PortfolioConfig
) -> pd.Series:
    style_columns = [
        "market_beta",
        "log_size",
        "volatility_exposure",
        "liquidity_exposure",
    ]
    result = weights.copy()
    for column in style_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.fillna(values.median())
        scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale < 1e-12:
            continue
        z = (values - values.mean()) / scale
        exposure = float((result * z).sum())
        if abs(exposure) <= config.style_exposure_tolerance:
            continue
        direction = np.sign(exposure)
        result *= np.exp(-0.35 * direction * z)
        result = _capped_normalize(result, config.max_stock_weight)
    return result


def validate_weights(
    frame: pd.DataFrame, weights: dict[str, float], config: PortfolioConfig
) -> None:
    series = pd.Series(weights, dtype=float)
    if series.empty:
        raise ValueError("portfolio cannot be empty")
    if (series < -1e-12).any():
        raise ValueError("long-only portfolio contains a negative weight")
    if series.max() > config.max_stock_weight + 1e-8:
        raise ValueError("single-stock weight cap violated")
    if abs(series.sum() - 1.0) > 1e-8:
        raise ValueError("portfolio weights do not sum to one")
    local = frame.set_index("stock_code").loc[series.index]
    industry_weight = series.groupby(local["industry"].fillna("UNKNOWN")).sum()
    if industry_weight.max() > config.industry_absolute_cap + 1e-6:
        raise ValueError("absolute industry cap violated")


def _risk_standardized(frame: pd.DataFrame) -> pd.DataFrame:
    styles = frame[
        ["market_beta", "log_size", "volatility_exposure", "liquidity_exposure"]
    ].apply(pd.to_numeric, errors="coerce")
    styles = styles.fillna(styles.median()).fillna(0.0)
    scale = styles.std(ddof=0).replace(0.0, 1.0)
    return (styles - styles.mean()) / scale


def build_target_weights(
    frame: pd.DataFrame,
    previous: dict[str, float],
    config: PortfolioConfig,
) -> tuple[dict[str, float], pd.Series]:
    """Transparent deterministic approximation to the stated convex objective."""
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    frame = frame.drop_duplicates("stock_code").set_index("stock_code", drop=False)
    score_column, alpha_column, _ = score_columns(config.target_type)
    score = pd.to_numeric(frame[score_column], errors="coerce")
    alpha = pd.to_numeric(frame[alpha_column], errors="coerce")
    if config.portfolio_type == "equal_weight":
        count = max(1, int(np.ceil(len(frame) * 0.10)))
        selected = frame.loc[score.nlargest(count).index]
        weights = pd.Series(1.0 / len(selected), index=selected.index)
        return weights.to_dict(), score

    clipped = alpha.clip(alpha.quantile(0.01), alpha.quantile(0.99))
    if config.portfolio_type == "score_weight":
        positive = clipped[clipped > 0].nlargest(config.max_holdings)
        if len(positive) < config.min_holdings:
            positive = (score.rank(pct=True).nlargest(config.min_holdings) + 1e-6)
        weights = _capped_normalize(positive - positive.min() + 1e-8, config.max_stock_weight)
        local = frame.loc[weights.index]
        weights = _industry_cap_weights(local, weights, config)
        weights = _style_tilt(local, weights, config)
        return weights.to_dict(), score

    styles = _risk_standardized(frame)
    volatility_risk = styles["volatility_exposure"].abs()
    beta_risk = styles["market_beta"].abs()
    size_risk = styles["log_size"].abs()
    liquidity_risk = styles["liquidity_exposure"].abs()
    alpha_scale = float(clipped.std(ddof=0))
    standardized_alpha = (
        (clipped - clipped.median()) / alpha_scale if alpha_scale > 1e-12 else score.rank(pct=True) - 0.5
    )
    held_bonus = pd.Series(
        [1.0 if name in previous else 0.0 for name in frame.index], index=frame.index
    )
    reference_weight = 1.0 / min(max(config.min_holdings, 150), config.max_holdings)
    costs = pd.Series(
        {
            name: estimate_order_cost(
                reference_weight,
                row["amount_clean"],
                row["volatility_exposure"],
                config.portfolio_capital_rmb,
                config,
            )["estimated_total_cost"]
            for name, row in frame.iterrows()
        }
    )
    cost_scale = float(costs.std(ddof=0))
    standardized_cost = (
        (costs - costs.median()) / cost_scale if cost_scale > 1e-12 else costs * 0.0
    )
    # Diagonal idiosyncratic risk plus beta/style exposure proxy.  This is not
    # presented as a commercial covariance model.
    risk_proxy = (
        0.45 * volatility_risk
        + 0.30 * beta_risk
        + 0.15 * size_risk
        + 0.10 * liquidity_risk
    )
    utility = (
        standardized_alpha
        - 0.12 * config.lambda_risk * risk_proxy
        + 0.25 * config.lambda_turnover * held_bonus
        - 0.08 * standardized_cost
    )
    candidate_count = min(config.max_holdings, max(config.min_holdings, 150))
    selected_index = utility.nlargest(candidate_count).index
    desired_raw = (utility.loc[selected_index] - utility.loc[selected_index].min() + 0.05)
    desired = _capped_normalize(desired_raw, config.max_stock_weight)
    desired = _industry_cap_weights(frame.loc[selected_index], desired, config)
    desired = _style_tilt(frame.loc[selected_index], desired, config)

    previous_series = pd.Series(previous, dtype=float)
    union = desired.index.union(previous_series.index.intersection(frame.index))
    desired = desired.reindex(union, fill_value=0.0)
    previous_series = previous_series.reindex(union, fill_value=0.0)
    step = 1.0 / (1.0 + config.lambda_turnover)
    blended = (1.0 - step) * previous_series + step * desired
    if len(blended[blended > 1e-10]) > config.max_holdings:
        blended = blended.nlargest(config.max_holdings)
    if len(blended[blended > 1e-10]) < config.min_holdings:
        additions = utility.drop(blended.index).nlargest(
            config.min_holdings - len(blended[blended > 1e-10])
        )
        blended = pd.concat([blended, pd.Series(1e-4, index=additions.index)])
    local = frame.loc[blended.index]
    blended = _capped_normalize(blended, config.max_stock_weight)
    blended = _industry_cap_weights(local, blended, config)
    blended = _style_tilt(local, blended, config)
    weights = blended[blended > 1e-10].to_dict()
    validate_weights(frame.reset_index(drop=True), weights, config)
    return weights, utility


def _portfolio_cost(
    previous: dict[str, float],
    current: dict[str, float],
    frame: pd.DataFrame,
    config: PortfolioConfig,
) -> tuple[float, list[dict[str, object]]]:
    indexed = frame.set_index("stock_code")
    total = 0.0
    rows = []
    for name in sorted(set(previous) | set(current)):
        delta = float(current.get(name, 0.0) - previous.get(name, 0.0))
        if abs(delta) < 1e-12:
            continue
        row = indexed.loc[name] if name in indexed.index else pd.Series(dtype=float)
        cost = estimate_order_cost(
            abs(delta),
            row.get("amount_clean", np.nan),
            row.get("volatility_exposure", np.nan),
            config.portfolio_capital_rmb,
            config,
        )
        total += abs(delta) * cost["estimated_total_cost"]
        rows.append({"stock_code": name, "delta": delta, **cost})
    return float(total), rows


def simulate_portfolio(
    predictions: pd.DataFrame,
    config: PortfolioConfig,
    collect_details: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    previous: dict[str, float] = {}
    score_column, alpha_column, target_column = score_columns(config.target_type)
    for signal_month, month_frame in predictions.groupby("signal_month", sort=True):
        frame = month_frame.copy()
        frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
        weights, utility = build_target_weights(frame, previous, config)
        turnover = true_weight_turnover(weights, previous)
        known_return, unresolved_weight = weighted_return(
            weights, frame.set_index("stock_code")["forward_return_1m"]
        )
        gross_return = np.nan if unresolved_weight > 1e-12 else known_return
        proxy_cost, trades = _portfolio_cost(previous, weights, frame, config)
        invested_weight = float(sum(weights.values()))
        monthly_rows.append(
            {
                "signal_month": signal_month,
                "realization_month": frame["realization_month"].iloc[0],
                "period": frame["period"].iloc[0],
                "target_type": config.target_type,
                "portfolio_type": config.portfolio_type,
                "lambda_risk": config.lambda_risk,
                "lambda_turnover": config.lambda_turnover,
                "gross_return": gross_return,
                "turnover": turnover,
                "explicit_cost_20bps": turnover * 20 / 10_000,
                "impact_proxy_cost": proxy_cost,
                "net_return_20bps": gross_return - turnover * 20 / 10_000,
                "net_return_50bps": gross_return - turnover * 50 / 10_000,
                "net_return_100bps": gross_return - turnover * 100 / 10_000,
                "proxy_net_return": gross_return - proxy_cost,
                "invested_weight": invested_weight,
                "cash_weight": max(0.0, 1.0 - invested_weight),
                "holdings": len(weights),
                "unresolved_return_weight": unresolved_weight,
                "rank_ic": frame[score_column].corr(
                    frame["original_1m_target"], method="spearman"
                ),
                "target_rank_ic": frame[score_column].corr(
                    frame[target_column], method="spearman"
                ),
                "maximum_feature_month": signal_month,
            }
        )
        if collect_details:
            indexed = frame.set_index("stock_code")
            utility = utility.reindex(indexed.index)
            for name, weight in weights.items():
                row = indexed.loc[name]
                holding_rows.append(
                    {
                        "signal_month": signal_month,
                        "realization_month": frame["realization_month"].iloc[0],
                        "stock_code": name,
                        "industry": row["industry"],
                        "target_weight": weight,
                        "raw_ridge_score": row[score_column],
                        "calibrated_alpha": row[alpha_column],
                        "utility": utility.get(name, np.nan),
                        "target_type": config.target_type,
                        "portfolio_type": config.portfolio_type,
                        "forward_return": row["forward_return_1m"],
                        "volatility": row["volatility_exposure"],
                        "beta": row["market_beta"],
                        "log_size": row["log_size"],
                        "liquidity": row["liquidity_exposure"],
                        "amount_clean": row["amount_clean"],
                        "value_exposure": row["value_composite"],
                        "reversal_exposure": row["short_reversal"],
                    }
                )
            trade_lookup = {row["stock_code"]: row for row in trades}
            for name in sorted(set(previous) | set(weights)):
                old_weight = previous.get(name, 0.0)
                final_weight = weights.get(name, 0.0)
                delta = final_weight - old_weight
                if abs(delta) < 1e-12:
                    continue
                row = indexed.loc[name] if name in indexed.index else pd.Series(dtype=float)
                cost = trade_lookup[name]
                trade_rows.append(
                    {
                        "signal_month": signal_month,
                        "realization_month": frame["realization_month"].iloc[0],
                        "stock_code": name,
                        "industry": row.get("industry", np.nan),
                        "old_weight": old_weight,
                        "unconstrained_target_weight": final_weight,
                        "final_target_weight": final_weight,
                        "traded_weight": abs(delta),
                        "raw_ridge_score": row.get(score_column, np.nan),
                        "calibrated_alpha": row.get(alpha_column, np.nan),
                        "target_type": config.target_type,
                        "portfolio_type": config.portfolio_type,
                        "volatility": row.get("volatility_exposure", np.nan),
                        "beta": row.get("market_beta", np.nan),
                        "log_size": row.get("log_size", np.nan),
                        "liquidity": row.get("liquidity_exposure", np.nan),
                        "estimated_base_cost": cost["estimated_base_cost"],
                        "estimated_impact_cost": cost["estimated_impact_cost"],
                        "estimated_total_cost": cost["estimated_total_cost"],
                        "ADV_participation_5d": cost["ADV_participation_5d"],
                        "trade_allowed": bool(
                            delta < 0
                            or cost["ADV_participation_5d"]
                            <= config.max_adv_participation_5d
                        ),
                        "rejection_reason": (
                            ""
                            if delta < 0
                            or cost["ADV_participation_5d"]
                            <= config.max_adv_participation_5d
                            else "ADV_participation_limit"
                        ),
                        "execution_status": "pending_tradable_return_engine",
                        "unresolved_return_flag": bool(
                            pd.isna(row.get("forward_return_1m", np.nan))
                        ),
                    }
                )
        previous = weights
    return (
        pd.DataFrame(monthly_rows),
        pd.DataFrame(holding_rows),
        pd.DataFrame(trade_rows),
    )


def _decile_monotonicity(predictions: pd.DataFrame, score_column: str) -> float:
    frame = predictions.copy()
    frame["decile"] = frame.groupby("signal_month", observed=True).apply(
        lambda group: decile_labels(group, score_column), include_groups=False
    ).reset_index(level=0, drop=True)
    means = frame.groupby("decile", observed=True)["forward_return_1m"].mean()
    return float(means.index.to_series().corr(means, method="spearman"))


def select_configuration(
    predictions: pd.DataFrame,
    quarterly_path: Path = OUT / "turnover_aware_backtest.csv",
) -> tuple[PortfolioConfig, pd.DataFrame]:
    selection_input = predictions[
        predictions["period"].isin(["development", "selection"])
    ].copy()
    assert_selection_period_only(selection_input)
    quarterly = pd.read_csv(quarterly_path)
    quarterly_selection_turnover = float(
        quarterly.loc[quarterly["period"] == "selection", "turnover"].mean()
    )
    rows = []
    for target_type in TARGET_TYPES:
        candidates = [
            PortfolioConfig(target_type=target_type, portfolio_type="equal_weight"),
            PortfolioConfig(target_type=target_type, portfolio_type="score_weight"),
        ]
        candidates.extend(
            PortfolioConfig(
                target_type=target_type,
                portfolio_type="cost_aware",
                lambda_risk=lambda_risk,
                lambda_turnover=lambda_turnover,
            )
            for lambda_risk in LAMBDA_GRID
            for lambda_turnover in LAMBDA_GRID
        )
        score_column, _, _ = score_columns(target_type)
        monotonicity = _decile_monotonicity(selection_input, score_column)
        for config in candidates:
            backtest, _, _ = simulate_portfolio(selection_input, config)
            development = backtest[backtest["period"] == "development"]
            selection = backtest[backtest["period"] == "selection"]
            development_metric = metric(development["net_return_50bps"])
            selection_metric = metric(selection["net_return_50bps"])
            rows.append(
                {
                    "target_type": target_type,
                    "portfolio_type": config.portfolio_type,
                    "lambda_risk": config.lambda_risk,
                    "lambda_turnover": config.lambda_turnover,
                    "uses_retrospective_test": False,
                    "development_50bps_annual_return": development_metric["annual_return"],
                    "development_50bps_sharpe": development_metric["sharpe_rf0"],
                    "selection_50bps_annual_return": selection_metric["annual_return"],
                    "selection_50bps_sharpe": selection_metric["sharpe_rf0"],
                    "selection_20bps_sharpe": metric(selection["net_return_20bps"])[
                        "sharpe_rf0"
                    ],
                    "selection_100bps_sharpe": metric(selection["net_return_100bps"])[
                        "sharpe_rf0"
                    ],
                    "selection_average_turnover": selection["turnover"].mean(),
                    "quarterly_selection_turnover": quarterly_selection_turnover,
                    "selection_mean_rank_ic": selection["rank_ic"].mean(),
                    "selection_mean_target_rank_ic": selection["target_rank_ic"].mean(),
                    "selection_decile_monotonicity": monotonicity,
                    "selection_average_invested_weight": selection["invested_weight"].mean(),
                    "selection_max_unresolved_weight": selection[
                        "unresolved_return_weight"
                    ].max(),
                }
            )
    table = pd.DataFrame(rows)
    original_ic = float(
        table[
            (table["target_type"] == "original_1m")
            & (table["portfolio_type"] == "equal_weight")
        ]["selection_mean_rank_ic"].iloc[0]
    )
    table["rank_ic_drop_vs_original_ridge"] = original_ic - table["selection_mean_rank_ic"]
    table["passes_constraints"] = (
        (table["selection_50bps_annual_return"] > 0)
        & (table["development_50bps_annual_return"] > 0)
        & (
            table["selection_average_turnover"]
            <= table["quarterly_selection_turnover"] * 1.05
        )
        & (table["rank_ic_drop_vs_original_ridge"] <= 0.005)
        & (table["selection_decile_monotonicity"] >= 0.80)
        & (table["selection_average_invested_weight"] >= 0.95)
    )
    table["selection_objective"] = (
        table["selection_50bps_sharpe"]
        + 0.20 * table["development_50bps_sharpe"]
        - 0.25 * table["selection_average_turnover"]
    )
    pool = table[table["passes_constraints"]]
    if pool.empty:
        pool = table
    selected_index = pool.sort_values(
        ["selection_objective", "selection_average_turnover"], ascending=[False, True]
    ).index[0]
    table["selected"] = table.index == selected_index
    table["selection_status"] = np.where(
        table["selected"] & table["passes_constraints"],
        "frozen_candidate",
        np.where(
            table["selected"],
            "failed_fallback_for_reporting_only",
            "not_selected",
        ),
    )
    chosen = table.loc[selected_index]
    config = PortfolioConfig(
        target_type=str(chosen["target_type"]),
        portfolio_type=str(chosen["portfolio_type"]),
        lambda_risk=float(chosen["lambda_risk"]),
        lambda_turnover=float(chosen["lambda_turnover"]),
    )
    return config, table.sort_values("selection_objective", ascending=False)


def _metrics_and_stress(backtest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stress_rows = []
    for bps in COST_GRID_BPS:
        returns = backtest["gross_return"] - backtest["turnover"] * bps / 10_000
        stress_rows.append(
            {
                "cost_bps": bps,
                **metric(returns),
                "average_turnover": backtest["turnover"].mean(),
                "average_invested_weight": backtest["invested_weight"].mean(),
            }
        )
    stress = pd.DataFrame(stress_rows)
    metrics_table = stress[stress["cost_bps"].isin([20, 50, 100])].copy()
    subperiod_rows = []
    for period, group in backtest.groupby("period", observed=True):
        for bps in (20, 50, 100):
            subperiod_rows.append(
                {
                    "period": period,
                    "cost_bps": bps,
                    **metric(group["gross_return"] - group["turnover"] * bps / 10_000),
                    "average_turnover": group["turnover"].mean(),
                    "average_invested_weight": group["invested_weight"].mean(),
                    "mean_rank_ic": group["rank_ic"].mean(),
                }
            )
    return metrics_table, stress, pd.DataFrame(subperiod_rows)


def _capacity(trades: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    rows = []
    buys = trades[trades["final_target_weight"] > trades["old_weight"]]
    for capital in CAPITAL_GRID:
        participation = (
            buys["traded_weight"]
            * capital
            / (
                buys["ADV_participation_5d"]
                .where(buys["ADV_participation_5d"] > 0)
                .rdiv(buys["traded_weight"] * config.portfolio_capital_rmb)
            )
        )
        # Recompute robustly from the stored default-capital participation.
        participation = (
            buys["ADV_participation_5d"] * capital / config.portfolio_capital_rmb
        ).replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "portfolio_capital_rmb": capital,
                "execution_days": config.execution_days,
                "p50_ADV_participation_5d": participation.quantile(0.50),
                "p90_ADV_participation_5d": participation.quantile(0.90),
                "p95_ADV_participation_5d": participation.quantile(0.95),
                "p99_ADV_participation_5d": participation.quantile(0.99),
                "max_ADV_participation_5d": participation.max(),
                "share_over_limit": (
                    participation > config.max_adv_participation_5d
                ).mean(),
                "scope": "long_only_liquidity_proxy_not_short_capacity",
            }
        )
    return pd.DataFrame(rows)


def exposure_and_attribution(
    predictions: pd.DataFrame, holdings: pd.DataFrame, backtest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposure_rows = []
    attribution_rows = []
    for signal_month, group in holdings.groupby("signal_month", sort=True):
        pool = predictions[predictions["signal_month"] == signal_month].copy()
        weights = group.set_index("stock_code")["target_weight"]
        local = group.set_index("stock_code")
        style_map = {
            "market_beta_exposure": "beta",
            "size_exposure": "log_size",
            "volatility_exposure": "volatility",
            "liquidity_exposure": "liquidity",
            "value_exposure": "value_exposure",
            "reversal_exposure": "reversal_exposure",
        }
        record: dict[str, object] = {"signal_month": signal_month}
        for output_column, input_column in style_map.items():
            values = pd.to_numeric(local[input_column], errors="coerce").fillna(
                pd.to_numeric(local[input_column], errors="coerce").median()
            )
            record[output_column] = float((weights * values).sum())
        portfolio_industry = weights.groupby(local["industry"].fillna("UNKNOWN")).sum()
        pool_industry = pool["industry"].fillna("UNKNOWN").value_counts(normalize=True)
        record["industry_exposure"] = float(
            max(
                abs(portfolio_industry.get(name, 0.0) - pool_industry.get(name, 0.0))
                for name in set(portfolio_industry.index) | set(pool_industry.index)
            )
        )
        exposure_rows.append(record)

        realized = pd.to_numeric(pool["forward_return_1m"], errors="coerce")
        market_return = float(realized.mean())
        industry_return = pool.assign(_return=realized).groupby(
            pool["industry"].fillna("UNKNOWN")
        )["_return"].mean()
        industry_component = float(
            sum(
                (portfolio_industry.get(name, 0.0) - pool_industry.get(name, 0.0))
                * industry_return.get(name, 0.0)
                for name in portfolio_industry.index
            )
        )
        regression_columns = [
            "market_beta",
            "log_size",
            "volatility_exposure",
            "liquidity_exposure",
            "value_composite",
            "short_reversal",
        ]
        design_frame = pool[regression_columns].apply(pd.to_numeric, errors="coerce")
        design_frame = design_frame.fillna(design_frame.median()).fillna(0.0)
        design_frame = (
            design_frame - design_frame.mean()
        ) / design_frame.std(ddof=0).replace(0.0, 1.0)
        valid = realized.notna()
        beta = np.linalg.lstsq(
            np.column_stack([np.ones(valid.sum()), design_frame.loc[valid].to_numpy()]),
            realized.loc[valid].to_numpy(),
            rcond=1e-10,
        )[0]
        held_pool = pool.set_index("stock_code").loc[weights.index]
        held_styles = held_pool[regression_columns].apply(pd.to_numeric, errors="coerce")
        held_styles = held_styles.fillna(pool[regression_columns].median()).fillna(0.0)
        held_styles = (
            held_styles - pool[regression_columns].mean()
        ) / pool[regression_columns].std(ddof=0).replace(0.0, 1.0)
        style_component = float(
            (weights.to_numpy()[:, None] * held_styles.to_numpy()).sum(axis=0)
            @ beta[1:]
        )
        monthly = backtest[backtest["signal_month"] == signal_month].iloc[0]
        market_component = float(record["market_beta_exposure"] * market_return)
        specific = float(
            monthly["gross_return"]
            - market_component
            - industry_component
            - style_component
        )
        attribution_rows.append(
            {
                "signal_month": signal_month,
                "realization_month": monthly["realization_month"],
                "market_component": market_component,
                "industry_component": industry_component,
                "style_component": style_component,
                "specific_return_proxy": specific,
                "transaction_cost": monthly["explicit_cost_20bps"],
                "portfolio_gross_return": monthly["gross_return"],
                "portfolio_net_return_20bps": monthly["net_return_20bps"],
                "attribution_type": "proxy_attribution_not_commercial_risk_model",
            }
        )
    return pd.DataFrame(exposure_rows), pd.DataFrame(attribution_rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_research_lock(
    root: Path,
    config: PortfolioConfig,
    selected_row: dict[str, object],
) -> dict[str, object]:
    return {
        "lock_version": 1,
        "status": (
            "frozen_for_2026_forward_shadow_observation"
            if bool(selected_row["passes_constraints"])
            else "research_failed_not_a_forward_candidate"
        ),
        "research_name": "residual_multi_horizon_ridge_cost_aware_portfolio",
        "frozen_config": asdict(config),
        "selected_parameters": {
            "target_type": config.target_type,
            "portfolio_type": config.portfolio_type,
            "lambda_risk": config.lambda_risk,
            "lambda_turnover": config.lambda_turnover,
        },
        "data_intervals": {
            "formation": ["2014-01-01", "2017-12-31"],
            "development": ["2018-01-01", "2021-12-31"],
            "selection": ["2022-01-01", "2023-12-31"],
            "retrospective_test": ["2024-01-01", "2025-12-31"],
            "true_forward_start": "2026-01-01",
        },
        "retrospective_used_for_parameter_selection": False,
        "selection_passed_constraints": bool(selected_row["passes_constraints"]),
        "limitations": DataLimitations().__dict__,
        "git_commit": _git_commit(root),
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": {
            name: _sha256(root / name) for name in SOURCE_FILES if (root / name).exists()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=OUT / "residual_predictions.csv")
    args = parser.parse_args()
    predictions = pd.read_csv(
        args.predictions, parse_dates=["signal_month", "realization_month"]
    )
    predictions["stock_code"] = predictions["stock_code"].astype(str).str.zfill(6)
    selected, selection = select_configuration(predictions)
    backtest, holdings, trades = simulate_portfolio(
        predictions, selected, collect_details=True
    )
    metrics_table, stress, subperiod = _metrics_and_stress(backtest)
    capacity = _capacity(trades, selected)
    exposure, attribution = exposure_and_attribution(
        predictions, holdings, backtest
    )
    selection.to_csv(OUT / "cost_aware_parameter_selection.csv", index=False)
    backtest.to_csv(OUT / "cost_aware_backtest.csv", index=False)
    holdings.to_csv(OUT / "cost_aware_holdings.csv", index=False)
    trades.to_csv(OUT / "cost_aware_trade_log.csv", index=False)
    exposure.to_csv(OUT / "cost_aware_exposure.csv", index=False)
    metrics_table.to_csv(OUT / "cost_aware_metrics.csv", index=False)
    subperiod.to_csv(OUT / "cost_aware_subperiod.csv", index=False)
    stress.to_csv(OUT / "cost_aware_cost_stress.csv", index=False)
    capacity.to_csv(OUT / "cost_aware_capacity.csv", index=False)
    attribution.to_csv(OUT / "cost_aware_attribution.csv", index=False)
    selected_row = selection[selection["selected"]].iloc[0].to_dict()
    root = Path(__file__).resolve().parent
    (OUT / "residual_research_lock.json").write_text(
        json.dumps(
            build_research_lock(root, selected, selected_row),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(selection[selection["selected"]].to_string(index=False))
    print(subperiod.to_string(index=False))


if __name__ == "__main__":
    main()
