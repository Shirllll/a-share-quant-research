from __future__ import annotations

"""Daily-information risk overlay for the frozen monthly MLP strategy.

The daily inputs are used only through the signal month-end.  They do not enter
the return-prediction model: a fixed risk penalty only changes which *new*
positions are admitted, while the frozen alpha exit buffer protects existing
holdings from unnecessary trading.
"""

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_cleaning import resolve_data_dir
from optimized_quant import (
    OUT,
    ResearchConfig,
    build_portfolio,
    metric,
    period_metric,
    signal_for_model,
    smooth_scores,
)


ROOT = Path(__file__).resolve().parent
DAILY_COLUMNS = [
    "Stkcd", "Trddt", "Hiprc", "Loprc", "Clsprc", "Dnvaltrd",
    "Dretwd", "Trdsta", "LimitStatus",
]
SUM_COLUMNS = [
    "trading_days", "sum_return", "sum_return_sq", "sum_downside_sq",
    "sum_intraday_range", "sum_log_amount", "sum_log_amount_sq",
    "sum_illiquidity", "sum_limit",
]
RISK_COMPONENTS = {
    "tail": ["realized_volatility", "downside_volatility", "max_abs_return", "intraday_range"],
    "liquidity": ["illiquidity", "amount_instability", "limit_share", "intraday_range"],
    "composite": [
        "realized_volatility", "downside_volatility", "max_abs_return",
        "intraday_range", "illiquidity", "amount_instability", "limit_share",
    ],
}


@dataclass(frozen=True)
class DailyRiskConfig:
    start: str = "2017-12-01"
    end: str = "2025-12-31"
    minimum_daily_observations: int = 10
    penalty_candidates: tuple[float, ...] = (0.05, 0.10, 0.15)
    risk_model_candidates: tuple[str, ...] = ("tail", "liquidity", "composite")
    top_fraction: float = 0.10
    exit_fraction: float = 0.20
    smoothing_weight: float = 0.75
    selection_cost_bps: int = 50
    report_cost_bps: int = 20
    maximum_turnover_increase: float = 0.05
    minimum_average_sharpe_gain: float = 0.01
    maximum_development_sharpe_loss: float = 0.02
    minimum_selection_sharpe_gain: float = 0.0


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def aggregate_daily_chunk(frame: pd.DataFrame, config: DailyRiskConfig) -> pd.DataFrame:
    """Create additive monthly sufficient statistics from one daily-data chunk."""
    work = frame.copy()
    work["Stkcd"] = work["Stkcd"].astype("string").str.replace(
        r"\.0$", "", regex=True
    ).str.zfill(6)
    work["Trddt"] = pd.to_datetime(work["Trddt"], errors="coerce")
    _numeric(work, [
        "Hiprc", "Loprc", "Clsprc", "Dnvaltrd", "Dretwd", "Trdsta", "LimitStatus"
    ])
    work = work[
        work["Trddt"].between(config.start, config.end)
        & work["Trdsta"].eq(1)
        & work["Dretwd"].between(-0.50, 0.50)
        & work["Clsprc"].gt(0)
    ].copy()
    if work.empty:
        return pd.DataFrame()
    work["month"] = work["Trddt"] + pd.offsets.MonthEnd(0)
    amount = work["Dnvaltrd"].where(work["Dnvaltrd"].gt(0))
    log_amount = np.log(amount)
    intraday_range = ((work["Hiprc"] - work["Loprc"]) / work["Clsprc"]).clip(0, 1)
    returns = work["Dretwd"]
    work = work.assign(
        return_sq=returns.pow(2),
        downside_sq=returns.clip(upper=0).pow(2),
        abs_return=returns.abs(),
        intraday_range=intraday_range,
        log_amount=log_amount,
        log_amount_sq=log_amount.pow(2),
        illiquidity=(returns.abs() / (amount / 1e8)).clip(0, 100),
        limit_hit=work["LimitStatus"].fillna(0).ne(0).astype(float),
    )
    grouped = work.groupby(["Stkcd", "month"], observed=True, sort=False)
    return grouped.agg(
        trading_days=("Dretwd", "count"),
        sum_return=("Dretwd", "sum"),
        sum_return_sq=("return_sq", "sum"),
        sum_downside_sq=("downside_sq", "sum"),
        max_abs_return=("abs_return", "max"),
        min_daily_return=("Dretwd", "min"),
        sum_intraday_range=("intraday_range", "sum"),
        sum_log_amount=("log_amount", "sum"),
        sum_log_amount_sq=("log_amount_sq", "sum"),
        sum_illiquidity=("illiquidity", "sum"),
        sum_limit=("limit_hit", "sum"),
        source_max_date=("Trddt", "max"),
    ).reset_index()


def finalize_daily_aggregates(partials: list[pd.DataFrame],
                              config: DailyRiskConfig) -> pd.DataFrame:
    """Combine chunk summaries exactly and construct cross-sectional risk ranks."""
    valid = [frame for frame in partials if not frame.empty]
    if not valid:
        raise ValueError("No valid daily observations were found")
    combined = pd.concat(valid, ignore_index=True)
    aggregations: dict[str, str] = {column: "sum" for column in SUM_COLUMNS}
    aggregations.update({
        "max_abs_return": "max", "min_daily_return": "min", "source_max_date": "max"
    })
    monthly = combined.groupby(
        ["Stkcd", "month"], observed=True, as_index=False
    ).agg(aggregations)
    n = monthly["trading_days"].astype(float)
    variance = (
        monthly["sum_return_sq"] - monthly["sum_return"].pow(2) / n
    ) / (n - 1).clip(lower=1)
    log_amount_variance = (
        monthly["sum_log_amount_sq"] - monthly["sum_log_amount"].pow(2) / n
    ) / (n - 1).clip(lower=1)
    monthly["realized_volatility"] = np.sqrt(variance.clip(lower=0) * 252)
    monthly["downside_volatility"] = np.sqrt(
        monthly["sum_downside_sq"].div(n).clip(lower=0) * 252
    )
    monthly["intraday_range"] = monthly["sum_intraday_range"] / n
    monthly["illiquidity"] = monthly["sum_illiquidity"] / n
    monthly["amount_instability"] = np.sqrt(log_amount_variance.clip(lower=0))
    monthly["limit_share"] = monthly["sum_limit"] / n
    monthly = monthly[monthly["trading_days"].ge(config.minimum_daily_observations)].copy()

    all_components = sorted({item for values in RISK_COMPONENTS.values() for item in values})
    for component in all_components:
        clipped = monthly.groupby("month", observed=True)[component].transform(
            lambda values: values.clip(values.quantile(0.01), values.quantile(0.99))
        )
        monthly[f"{component}_rank"] = clipped.groupby(monthly["month"]).rank(pct=True)
    for model, components in RISK_COMPONENTS.items():
        monthly[f"risk_{model}"] = monthly[
            [f"{component}_rank" for component in components]
        ].mean(axis=1)
    if (monthly["source_max_date"] > monthly["month"]).any():
        raise RuntimeError("Daily risk features contain observations after signal month-end")
    keep = [
        "month", "Stkcd", "trading_days", "source_max_date",
        "realized_volatility", "downside_volatility", "max_abs_return",
        "min_daily_return", "intraday_range", "illiquidity",
        "amount_instability", "limit_share", "risk_tail", "risk_liquidity",
        "risk_composite",
    ]
    return monthly[keep].sort_values(["month", "Stkcd"]).reset_index(drop=True)


def build_daily_risk_features(data_dir: Path,
                              config: DailyRiskConfig = DailyRiskConfig()) -> pd.DataFrame:
    partials: list[pd.DataFrame] = []
    paths = sorted(data_dir.glob("02_daily_return_*.csv.zip"))
    if not paths:
        raise FileNotFoundError(f"Daily return archives not found under {data_dir}")
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            for member in members:
                with archive.open(member) as stream:
                    chunks = pd.read_csv(
                        stream, usecols=DAILY_COLUMNS, dtype={"Stkcd": "string"},
                        chunksize=250_000, low_memory=False,
                    )
                    partials.extend(aggregate_daily_chunk(chunk, config) for chunk in chunks)
    return finalize_daily_aggregates(partials, config)


def attach_risk(signals: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    left = signals.copy()
    right = risk.copy()
    left["Stkcd"] = left["Stkcd"].astype("string").str.zfill(6)
    right["Stkcd"] = right["Stkcd"].astype("string").str.zfill(6)
    merged = left.merge(right, on=["month", "Stkcd"], how="left", validate="one_to_one")
    for column in ["risk_tail", "risk_liquidity", "risk_composite"]:
        merged[column] = merged[column].fillna(
            merged.groupby("month", observed=True)[column].transform("median")
        ).fillna(0.5)
    return merged


def build_risk_portfolio(signals: pd.DataFrame, risk_model: str, penalty: float,
                         config: DailyRiskConfig = DailyRiskConfig(),
                         cost_bps: float | None = None) -> pd.DataFrame:
    """Penalize risk for new admissions, without forcing buffered holdings out."""
    risk_column = f"risk_{risk_model}"
    if risk_column not in signals:
        raise KeyError(risk_column)
    panel = smooth_scores(signals, config.smoothing_weight)
    previous: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for signal_month, group in panel.groupby("month", sort=True):
        group = group.dropna(subset=["forward_return", "smoothed_score"]).copy()
        if group.empty:
            continue
        count = max(1, int(np.ceil(len(group) * config.top_fraction)))
        exit_count = max(count, int(np.ceil(len(group) * config.exit_fraction)))
        permitted = set(group.nlargest(exit_count, "smoothed_score")["Stkcd"])
        retained = set(previous) & set(group["Stkcd"]) & permitted
        candidates = group[~group["Stkcd"].isin(retained)].copy()
        candidates["admission_score"] = (
            candidates["smoothed_score"] - penalty * candidates[risk_column]
        )
        additions = candidates.nlargest(max(0, count - len(retained)), "admission_score")
        chosen = pd.concat([
            group[group["Stkcd"].isin(retained)], additions,
        ]).drop_duplicates("Stkcd")
        weights = dict(zip(chosen["Stkcd"], np.repeat(1 / len(chosen), len(chosen))))
        union = set(previous) | set(weights)
        stock_l1 = sum(
            abs(weights.get(code, 0.0) - previous.get(code, 0.0)) for code in union
        )
        prior_cash = 1.0 - sum(previous.values())
        current_cash = 1.0 - sum(weights.values())
        turnover = 0.5 * (stock_l1 + abs(current_cash - prior_cash))
        realized = chosen.set_index("Stkcd")["forward_return"]
        rows.append({
            "signal_month": signal_month,
            "month": pd.Timestamp(signal_month) + pd.offsets.MonthEnd(1),
            "gross_return": sum(weight * realized.loc[code] for code, weight in weights.items()),
            "turnover": turnover,
            "holdings": len(weights),
            "risk_model": risk_model,
            "risk_penalty": penalty,
            "mean_selected_risk": chosen[risk_column].mean(),
        })
        previous = weights
    result = pd.DataFrame(rows)
    applied_cost = config.report_cost_bps if cost_bps is None else cost_bps
    result["net_return"] = result["gross_return"] - result["turnover"] * applied_cost / 10_000
    return result


def _accept(row: pd.Series, baseline: pd.Series,
            config: DailyRiskConfig) -> tuple[bool, str]:
    failures: list[str] = []
    if row["development_sharpe_50bp"] < baseline["development_sharpe_50bp"] - config.maximum_development_sharpe_loss:
        failures.append("development_sharpe")
    if row["selection_sharpe_50bp"] < baseline["selection_sharpe_50bp"] + config.minimum_selection_sharpe_gain:
        failures.append("selection_sharpe")
    if row["average_pre_retro_sharpe_50bp"] < baseline["average_pre_retro_sharpe_50bp"] + config.minimum_average_sharpe_gain:
        failures.append("average_sharpe_gain")
    if row["pre_retro_turnover"] > baseline["pre_retro_turnover"] * (1 + config.maximum_turnover_increase):
        failures.append("turnover")
    return not failures, "accepted" if not failures else ";".join(failures)


def evaluate_candidates(signals: pd.DataFrame,
                        risk: pd.DataFrame,
                        config: DailyRiskConfig = DailyRiskConfig(),
                        research: ResearchConfig = ResearchConfig(),
                        ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Select only with development/selection columns; retrospective is absent."""
    mlp = attach_risk(signal_for_model(signals, "mlp"), risk)
    variants = {
        "production_core": build_portfolio(
            mlp, config.top_fraction, config.exit_fraction,
            config.smoothing_weight, config.selection_cost_bps,
        )
    }
    for model in config.risk_model_candidates:
        for penalty in config.penalty_candidates:
            name = f"{model}_penalty_{penalty:.2f}"
            variants[name] = build_risk_portfolio(
                mlp, model, penalty, config, config.selection_cost_bps
            )
    rows = []
    for name, result in variants.items():
        development = period_metric(result, research.start, research.development_end, 50)
        selection = period_metric(result, research.selection_start, research.selection_end, 50)
        pre_retro = result[result["month"].between(research.start, research.selection_end)]
        rows.append({
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
    table = pd.DataFrame(rows)
    baseline = table[table["candidate"].eq("production_core")].iloc[0]
    decisions = table.apply(
        lambda row: (True, "frozen_baseline") if row["candidate"] == "production_core"
        else _accept(row, baseline, config), axis=1,
    )
    table[["accepted", "rejection_reason"]] = pd.DataFrame(
        decisions.tolist(), index=table.index
    )
    table["selected"] = False
    accepted = table[table["accepted"] & ~table["candidate"].eq("production_core")]
    chosen_name = "production_core" if accepted.empty else accepted.sort_values(
        ["average_pre_retro_sharpe_50bp", "pre_retro_turnover"], ascending=[False, True]
    ).iloc[0]["candidate"]
    table.loc[table["candidate"].eq(chosen_name), "selected"] = True
    return variants, table


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = DailyRiskConfig()
    research = ResearchConfig()
    signal_path = OUT / "optimized_signal_panel.csv.gz"
    risk_path = OUT / "daily_risk_features.csv.gz"
    if not signal_path.exists():
        raise FileNotFoundError("Run optimized_quant.py before this experiment")
    signals = pd.read_csv(
        signal_path, parse_dates=["month", "realization_month"], dtype={"Stkcd": "string"}
    )
    if risk_path.exists():
        risk = pd.read_csv(
            risk_path, parse_dates=["month", "source_max_date"], dtype={"Stkcd": "string"}
        )
    else:
        data_dir = resolve_data_dir()
        if data_dir is None:
            raise FileNotFoundError("Local daily data directory was not found")
        risk = build_daily_risk_features(data_dir, config)
        risk.to_csv(risk_path, index=False, compression="gzip")

    variants, selection = evaluate_candidates(signals, risk, config, research)
    selection.to_csv(OUT / "daily_risk_parameter_selection.csv", index=False)
    selected_name = selection.loc[selection["selected"], "candidate"].iloc[0]
    selected = variants[selected_name].copy()
    selected["net_return"] = selected["gross_return"] - selected["turnover"] * config.report_cost_bps / 10_000
    selected.to_csv(OUT / "daily_risk_backtest.csv", index=False)

    retrospective_rows = []
    subperiod_rows = []
    stress_rows = []
    periods = {
        "development": (research.start, research.development_end),
        "selection": (research.selection_start, research.selection_end),
        "retrospective_test": (research.retrospective_start, research.retrospective_end),
    }
    for name, result in variants.items():
        retrospective = period_metric(
            result, research.retrospective_start, research.retrospective_end, 50
        )
        retrospective_rows.append({"candidate": name, **retrospective})
        for period, (start, end) in periods.items():
            subperiod_rows.append({
                "candidate": name, "period": period,
                **period_metric(result, start, end, config.report_cost_bps),
            })
        for cost in (0, 20, 50, 100, 150, 200):
            net = result["gross_return"] - result["turnover"] * cost / 10_000
            stress_rows.append({
                "candidate": name, "cost_bps": cost,
                "mean_turnover": result["turnover"].mean(), **metric(net),
            })
    pd.DataFrame(retrospective_rows).to_csv(
        OUT / "daily_risk_retrospective.csv", index=False
    )
    pd.DataFrame(subperiod_rows).to_csv(OUT / "daily_risk_subperiod.csv", index=False)
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(OUT / "daily_risk_cost_stress.csv", index=False)

    quality = pd.DataFrame([{
        "rows": len(risk), "stocks": risk["Stkcd"].nunique(),
        "months": risk["month"].nunique(), "first_month": risk["month"].min(),
        "last_month": risk["month"].max(),
        "minimum_source_date": risk["source_max_date"].min(),
        "maximum_source_date": risk["source_max_date"].max(),
        "future_leakage_rows": int((risk["source_max_date"] > risk["month"]).sum()),
        "input_frequency": "daily", "not_order_book_or_intraday": True,
    }])
    quality.to_csv(OUT / "daily_risk_data_quality.csv", index=False)
    risk[["month", "source_max_date"]].drop_duplicates().assign(
        leakage_pass=lambda frame: frame["source_max_date"].le(frame["month"])
    ).to_csv(OUT / "daily_risk_leakage_audit.csv", index=False)

    lock = {
        "status": "experiment_does_not_replace_frozen_core" if selected_name == "production_core"
        else "accepted_using_development_and_selection_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrospective_test_not_used_for_selection": True,
        "true_forward_start": "2026-09-01",
        "selected_candidate": selected_name,
        "config": asdict(config),
        "code_sha256": {"daily_risk_quant.py": sha256(Path(__file__))},
    }
    (OUT / "daily_risk_research_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(selection.to_string(index=False))
    print("Selected:", selected_name)
    print(stress[stress["candidate"].isin(["production_core", selected_name])].to_string(index=False))


if __name__ == "__main__":
    main()
