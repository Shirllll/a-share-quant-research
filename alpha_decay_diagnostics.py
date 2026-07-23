from __future__ import annotations

"""Parameter-free diagnosis of where the 2024-2025 alpha decay occurred."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_quant import FEATURES, OUT
from residual_common import (
    DataLimitations,
    build_residual_panel,
    decile_labels,
    metric,
    period_label,
    quantile_bucket,
    true_weight_turnover,
    weighted_return,
)


PERIODS = ("development", "selection", "retrospective_test")


def _safe_ic(frame: pd.DataFrame, x: str, y: str) -> float:
    usable = frame[[x, y]].dropna()
    if len(usable) < 10:
        return np.nan
    return float(usable[x].corr(usable[y], method="spearman"))


def factor_ic_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor in FEATURES:
        values = panel[factor].mask(panel[f"{factor}_missing"].eq(1))
        local = panel.loc[panel["eligible"] & panel["target"].notna(), ["month", "period", "target"]].copy()
        local["factor_value"] = values.loc[local.index]
        monthly = (
            local.groupby("month", observed=True)
            .apply(lambda group: _safe_ic(group, "factor_value", "target"), include_groups=False)
            .rename("ic")
            .reset_index()
        )
        monthly["period"] = monthly["month"].map(period_label)
        monthly["rolling_12m_ic"] = monthly["ic"].rolling(12, min_periods=6).mean()
        period_means: dict[str, float] = {}
        for period in PERIODS:
            sample = monthly[monthly["period"] == period].dropna(subset=["ic"])
            mean_ic = float(sample["ic"].mean())
            standard_deviation = float(sample["ic"].std(ddof=1))
            period_means[period] = mean_ic
            rows.append(
                {
                    "factor": factor,
                    "period": period,
                    "mean_monthly_rank_ic": mean_ic,
                    "ic_standard_deviation": standard_deviation,
                    "annualized_icir": (
                        mean_ic / standard_deviation * np.sqrt(12)
                        if standard_deviation > 0
                        else np.nan
                    ),
                    "positive_ic_month_share": float((sample["ic"] > 0).mean()),
                    "months": int(len(sample)),
                    "rolling_12m_ic_at_period_end": (
                        float(sample["rolling_12m_ic"].iloc[-1]) if not sample.empty else np.nan
                    ),
                }
            )
        for row in rows[-len(PERIODS) :]:
            row["retrospective_change_vs_development"] = (
                period_means["retrospective_test"] - period_means["development"]
            )
            row["retrospective_change_vs_selection"] = (
                period_means["retrospective_test"] - period_means["selection"]
            )
    return pd.DataFrame(rows)


def coefficient_diagnostics(path: Path = OUT / "linear_factor_coefficients.csv") -> pd.DataFrame:
    coefficients = pd.read_csv(path, parse_dates=["refit_month"])
    coefficients = coefficients[coefficients["feature"].isin(FEATURES)].sort_values(
        ["feature", "refit_month"]
    )
    coefficients["period"] = coefficients["refit_month"].map(period_label)
    rows = []
    for (feature, period), group in coefficients.groupby(["feature", "period"], observed=True):
        values = pd.to_numeric(group["standardized_coefficient"], errors="coerce").dropna()
        signs = np.sign(values).replace(0, np.nan).dropna()
        rows.append(
            {
                "factor": feature,
                "period": period,
                "mean_standardized_coefficient": float(values.mean()),
                "mean_absolute_coefficient": float(values.abs().mean()),
                "absolute_coefficient_stability": (
                    float(1.0 - values.abs().std(ddof=1) / values.abs().mean())
                    if values.abs().mean() > 0
                    else np.nan
                ),
                "coefficient_direction_changes": int((signs.diff().dropna() != 0).sum()),
                "refits": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def _industry_leg_weights(frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    pairs = []
    for _, group in frame.groupby("industry", dropna=False):
        if len(group) < 10:
            continue
        count = max(1, int(np.floor(len(group) * 0.10)))
        ordered = group.sort_values("raw_score")
        pairs.append((ordered.tail(count), ordered.head(count)))
    if not pairs:
        return {}, {}
    industry_weight = 1.0 / len(pairs)
    long_weights: dict[str, float] = {}
    short_weights: dict[str, float] = {}
    for long_group, short_group in pairs:
        long_weights.update(
            {
                str(code): industry_weight / len(long_group)
                for code in long_group["stock_code"]
            }
        )
        short_weights.update(
            {
                str(code): industry_weight / len(short_group)
                for code in short_group["stock_code"]
            }
        )
    return long_weights, short_weights


def long_short_leg_diagnostics(score_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    previous_long: dict[str, float] = {}
    previous_short: dict[str, float] = {}
    for signal_month, group in score_panel.groupby("signal_month", sort=True):
        group = group.copy()
        group["stock_code"] = group["stock_code"].astype(str).str.zfill(6)
        indexed_return = group.set_index("stock_code")["forward_return"]
        long_weights, short_weights = _industry_leg_weights(group)
        long_return, long_unresolved = weighted_return(long_weights, indexed_return)
        bottom_return, short_unresolved = weighted_return(short_weights, indexed_return)
        long_turnover = true_weight_turnover(long_weights, previous_long)
        short_turnover = true_weight_turnover(short_weights, previous_short)
        # Short-leg return is consistently signed as the P&L from selling short.
        short_sale_return = -bottom_return
        rows.append(
            {
                "signal_month": signal_month,
                "realization_month": group["realization_month"].iloc[0],
                "period": group["period"].iloc[0],
                "long_gross_return": long_return,
                "short_sale_gross_return": short_sale_return,
                "long_turnover": long_turnover,
                "short_turnover": short_turnover,
                "long_net_return_20bps": long_return - long_turnover * 20 / 10_000,
                "short_sale_net_return_20bps": short_sale_return - short_turnover * 20 / 10_000,
                "long_contribution": long_return,
                "short_contribution": short_sale_return,
                "long_unresolved_weight": long_unresolved,
                "short_unresolved_weight": short_unresolved,
                "long_names": len(long_weights),
                "short_names": len(short_weights),
            }
        )
        previous_long, previous_short = long_weights, short_weights
    result = pd.DataFrame(rows)
    for period in PERIODS:
        mask = result["period"] == period
        result.loc[mask, "period_long_sharpe_20bps"] = metric(
            result.loc[mask, "long_net_return_20bps"]
        )["sharpe_rf0"]
        result.loc[mask, "period_short_sharpe_20bps"] = metric(
            result.loc[mask, "short_sale_net_return_20bps"]
        )["sharpe_rf0"]
    return result


def decile_diagnostics(score_panel: pd.DataFrame) -> pd.DataFrame:
    monthly_rows: list[dict[str, object]] = []
    previous: dict[int, dict[str, float]] = {decile: {} for decile in range(1, 11)}
    for signal_month, group in score_panel.groupby("signal_month", sort=True):
        group = group.copy()
        group["stock_code"] = group["stock_code"].astype(str).str.zfill(6)
        group["decile"] = decile_labels(group, "raw_score")
        for decile, local in group.groupby("decile", observed=True):
            weights = {str(code): 1.0 / len(local) for code in local["stock_code"]}
            gross, unresolved = weighted_return(
                weights, local.set_index("stock_code")["forward_return"]
            )
            turnover = true_weight_turnover(weights, previous[int(decile)])
            monthly_rows.append(
                {
                    "signal_month": signal_month,
                    "period": group["period"].iloc[0],
                    "decile": int(decile),
                    "gross_return": gross,
                    "turnover": turnover,
                    "net_return_20bps": gross - turnover * 20 / 10_000,
                    "net_return_50bps": gross - turnover * 50 / 10_000,
                    "stocks": len(local),
                    "unresolved_weight": unresolved,
                }
            )
            previous[int(decile)] = weights
    monthly = pd.DataFrame(monthly_rows)
    rows = []
    for period, sample in monthly.groupby("period", observed=True):
        means = sample.groupby("decile")["gross_return"].mean()
        spreads = {
            "g10_minus_g1": means.get(10, np.nan) - means.get(1, np.nan),
            "g10_minus_g5": means.get(10, np.nan) - means.get(5, np.nan),
            "g6_minus_g1": means.get(6, np.nan) - means.get(1, np.nan),
            "decile_spearman": means.index.to_series().corr(means, method="spearman"),
        }
        for decile, group in sample.groupby("decile", observed=True):
            rows.append(
                {
                    "period": period,
                    "decile": int(decile),
                    "average_gross_return": float(group["gross_return"].mean()),
                    "average_net_return_20bps": float(group["net_return_20bps"].mean()),
                    "average_net_return_50bps": float(group["net_return_50bps"].mean()),
                    "return_standard_error": float(
                        group["gross_return"].std(ddof=1) / np.sqrt(group["gross_return"].count())
                    ),
                    "average_turnover": float(group["turnover"].mean()),
                    "average_stocks": float(group["stocks"].mean()),
                    **spreads,
                }
            )
    result = pd.DataFrame(rows)
    spread = result.drop_duplicates("period").set_index("period")["g10_minus_g1"]
    narrowing = spread.get("retrospective_test", np.nan) - np.nanmean(
        [spread.get("development", np.nan), spread.get("selection", np.nan)]
    )
    result["retrospective_g10_g1_change_vs_prior"] = narrowing
    result["top_bottom_spread_narrowed"] = bool(narrowing < 0)
    return result


def _stratified_table(
    score_panel: pd.DataFrame, dimension: str, output_name: str
) -> pd.DataFrame:
    monthly_rows = []
    previous: dict[str, dict[str, float]] = {}
    for signal_month, month_group in score_panel.groupby("signal_month", sort=True):
        for group_name, group in month_group.dropna(subset=[dimension]).groupby(
            dimension, observed=True
        ):
            if len(group) < 20:
                continue
            group = group.copy()
            group["stock_code"] = group["stock_code"].astype(str).str.zfill(6)
            count = max(1, int(np.floor(len(group) * 0.10)))
            ordered = group.sort_values("raw_score")
            top = ordered.tail(count)
            bottom = ordered.head(count)
            weights = {str(code): 1.0 / len(top) for code in top["stock_code"]}
            key = str(group_name)
            turnover = true_weight_turnover(weights, previous.get(key, {}))
            previous[key] = weights
            spread = top["forward_return"].mean() - bottom["forward_return"].mean()
            monthly_rows.append(
                {
                    output_name: group_name,
                    "signal_month": signal_month,
                    "period": month_group["period"].iloc[0],
                    "sample_count": len(group),
                    "rank_ic": _safe_ic(group, "raw_score", "target"),
                    "g10_minus_g1": spread,
                    "turnover": turnover,
                    "net_return_20bps": spread - 2.0 * turnover * 20 / 10_000,
                    "net_return_50bps": spread - 2.0 * turnover * 50 / 10_000,
                }
            )
    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        return monthly
    aggregate = (
        monthly.groupby([output_name, "period"], observed=True)
        .agg(
            sample_count=("sample_count", "sum"),
            mean_rank_ic=("rank_ic", "mean"),
            g10_minus_g1_return=("g10_minus_g1", "mean"),
            average_turnover=("turnover", "mean"),
            average_net_return_20bps=("net_return_20bps", "mean"),
            average_net_return_50bps=("net_return_50bps", "mean"),
            months=("signal_month", "nunique"),
        )
        .reset_index()
    )
    prior = (
        aggregate[aggregate["period"].isin(["development", "selection"])]
        .groupby(output_name)["g10_minus_g1_return"]
        .mean()
    )
    retrospective = (
        aggregate[aggregate["period"] == "retrospective_test"]
        .set_index(output_name)["g10_minus_g1_return"]
    )
    aggregate["retrospective_change_vs_prior"] = aggregate[output_name].map(
        retrospective - prior
    )
    return aggregate


def stratified_diagnostics(score_panel: pd.DataFrame, panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    exposure = panel[
        [
            "month",
            "Stkcd",
            "market_beta",
            "log_size",
            "liquidity_exposure",
            "volatility_exposure",
            "listing_years",
        ]
    ].rename(columns={"month": "signal_month", "Stkcd": "stock_code"})
    exposure["stock_code"] = exposure["stock_code"].astype(str).str.zfill(6)
    score_panel = score_panel.copy()
    score_panel["stock_code"] = score_panel["stock_code"].astype(str).str.zfill(6)
    merged = score_panel.merge(exposure, on=["signal_month", "stock_code"], how="left")
    merged["size_group"] = merged.groupby("signal_month")["log_size"].transform(quantile_bucket)
    merged["liquidity_group"] = merged.groupby("signal_month")[
        "liquidity_exposure"
    ].transform(quantile_bucket)
    merged["volatility_group"] = merged.groupby("signal_month")[
        "volatility_exposure"
    ].transform(quantile_bucket)
    merged["beta_group"] = merged.groupby("signal_month")["market_beta"].transform(
        quantile_bucket
    )
    merged["listing_age_group"] = pd.cut(
        merged["listing_years"],
        [-np.inf, 1, 3, 5, 10, np.inf],
        labels=["<1y", "1-3y", "3-5y", "5-10y", "10y+"],
    )
    return {
        "industry": _stratified_table(merged, "industry", "industry"),
        "size": _stratified_table(merged, "size_group", "size_group"),
        "liquidity": _stratified_table(
            merged, "liquidity_group", "liquidity_group"
        ),
        "volatility": _stratified_table(
            merged, "volatility_group", "volatility_group"
        ),
        "beta": _stratified_table(merged, "beta_group", "beta_group"),
        "listing_age": _stratified_table(
            merged, "listing_age_group", "listing_age_group"
        ),
    }


def holding_status_attribution(score_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    previous: dict[str, float] = {}
    holding_age: dict[str, int] = {}
    for signal_month, group in score_panel.groupby("signal_month", sort=True):
        group = group.copy()
        group["stock_code"] = group["stock_code"].astype(str).str.zfill(6)
        count = max(1, int(np.ceil(len(group) * 0.10)))
        chosen = group.nlargest(count, "raw_score")
        current = {code: 1.0 / len(chosen) for code in chosen["stock_code"]}
        available = set(group["stock_code"])
        returns = group.set_index("stock_code")["forward_return"]
        categories: dict[str, list[str]] = {
            "retained": [],
            "new_entry": [],
            "planned_exit": [],
            "forced_exit": [],
            "unfilled_or_carried": [],
        }
        for name in current:
            if pd.isna(returns.get(name, np.nan)):
                categories["unfilled_or_carried"].append(name)
            elif name in previous:
                categories["retained"].append(name)
            else:
                categories["new_entry"].append(name)
            holding_age[name] = holding_age.get(name, 0) + 1 if name in previous else 1
        for name in set(previous) - set(current):
            categories["planned_exit" if name in available else "forced_exit"].append(name)
        for category, names in categories.items():
            weight = sum(current.get(name, 0.0) for name in names)
            gross = sum(
                current.get(name, 0.0) * float(returns.get(name, np.nan))
                for name in names
                if pd.notna(returns.get(name, np.nan))
            )
            traded_weight = sum(
                abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in names
            )
            # 0.5 * sum(abs(delta weight)) is the standard one-way turnover.
            cost = 0.5 * traded_weight * 20 / 10_000
            rows.append(
                {
                    "signal_month": signal_month,
                    "realization_month": group["realization_month"].iloc[0],
                    "period": group["period"].iloc[0],
                    "holding_status": category,
                    "weight": weight,
                    "gross_return_contribution": gross,
                    "explicit_cost": cost,
                    "net_return_contribution": gross - cost,
                    "average_holding_period_months": (
                        float(np.mean([holding_age.get(name, 0) for name in names]))
                        if names
                        else np.nan
                    ),
                    "stocks": len(names),
                }
            )
        previous = current
        holding_age = {name: holding_age[name] for name in current}
    return pd.DataFrame(rows)


def gross_cost_attribution() -> pd.DataFrame:
    backtest = pd.read_csv(
        OUT / "turnover_aware_backtest.csv", parse_dates=["signal_month", "month"]
    )
    result = backtest[
        [
            "signal_month",
            "month",
            "period",
            "long_short_gross_return",
            "long_short_turnover",
            "long_short_net_return_20bps",
            "long_short_proxy_net_return",
        ]
    ].copy()
    result["explicit_cost"] = result["long_short_turnover"] * 20 / 10_000
    result["impact_proxy_cost"] = (
        result["long_short_net_return_20bps"] - result["long_short_proxy_net_return"]
    ).clip(lower=0)
    result["net_return"] = (
        result["long_short_gross_return"]
        - result["explicit_cost"]
        - result["impact_proxy_cost"]
    )
    for bps in (0, 20, 50, 100):
        result[f"net_return_{bps}bps"] = (
            result["long_short_gross_return"]
            - result["long_short_turnover"] * bps / 10_000
        )
    return result.rename(columns={"month": "realization_month"})


def main() -> None:
    score_path = OUT / "turnover_budget_score_panel.pkl"
    if not score_path.exists():
        raise FileNotFoundError(
            "Run turnover_aware_quant.py first or generate output/turnover_budget_score_panel.pkl"
        )
    score_panel = pd.read_pickle(score_path)
    score_panel["stock_code"] = score_panel["stock_code"].astype(str).str.zfill(6)
    cache = OUT / "residual_panel_cache.pkl"
    if cache.exists():
        panel, _ = pd.read_pickle(cache)
    else:
        panel, regressions = build_residual_panel()
        pd.to_pickle((panel, regressions), cache)

    factor_table = factor_ic_diagnostics(panel)
    coefficient_table = coefficient_diagnostics()
    deciles = decile_diagnostics(score_panel)
    legs = long_short_leg_diagnostics(score_panel)
    strata = stratified_diagnostics(score_panel, panel)
    holding = holding_status_attribution(score_panel)
    cost = gross_cost_attribution()

    factor_table.to_csv(OUT / "alpha_decay_factor_ic.csv", index=False)
    coefficient_table.to_csv(OUT / "alpha_decay_coefficients.csv", index=False)
    deciles.to_csv(OUT / "alpha_decay_deciles.csv", index=False)
    legs.to_csv(OUT / "alpha_decay_long_short_legs.csv", index=False)
    strata["industry"].to_csv(OUT / "alpha_decay_by_industry.csv", index=False)
    strata["size"].to_csv(OUT / "alpha_decay_by_size.csv", index=False)
    strata["liquidity"].to_csv(OUT / "alpha_decay_by_liquidity.csv", index=False)
    strata["volatility"].to_csv(OUT / "alpha_decay_by_volatility.csv", index=False)
    strata["beta"].to_csv(OUT / "alpha_decay_by_beta.csv", index=False)
    strata["listing_age"].to_csv(OUT / "alpha_decay_by_listing_age.csv", index=False)
    holding.to_csv(OUT / "alpha_decay_holding_status.csv", index=False)
    cost.to_csv(OUT / "alpha_decay_gross_cost_attribution.csv", index=False)

    leg_period = (
        legs.groupby("period", observed=True)[
            ["long_net_return_20bps", "short_sale_net_return_20bps"]
        ]
        .mean()
        .to_dict("index")
    )
    retrospective_cost = cost[cost["period"] == "retrospective_test"]
    decile_period = deciles.drop_duplicates("period").set_index("period")
    summary = {
        "method": "parameter_free_alpha_decay_diagnostics",
        "short_leg_sign_convention": (
            "short_sale_gross_return is positive when the sold-short bottom leg falls"
        ),
        "top_bottom_spread": {
            period: float(decile_period.loc[period, "g10_minus_g1"])
            for period in decile_period.index
        },
        "retrospective_leg_mean_returns_20bps": leg_period.get(
            "retrospective_test", {}
        ),
        "retrospective_zero_cost_annualized_mean": float(
            retrospective_cost["net_return_0bps"].mean() * 12
        ),
        "retrospective_20bps_annualized_mean": float(
            retrospective_cost["net_return_20bps"].mean() * 12
        ),
        "retrospective_50bps_annualized_mean": float(
            retrospective_cost["net_return_50bps"].mean() * 12
        ),
        "retrospective_100bps_annualized_mean": float(
            retrospective_cost["net_return_100bps"].mean() * 12
        ),
        "limitations": DataLimitations().__dict__,
    }
    (OUT / "alpha_decay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
