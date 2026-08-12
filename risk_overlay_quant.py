from __future__ import annotations

"""Causal CSI 500 futures-overlay research for the frozen long-only stock signal.

This is reported separately from the cash-equity strategy.  It is a research proxy:
the hedge return uses the CSI 500 cash index while turnover and roll costs approximate
an index-futures implementation.  It does not model contract basis or margin calls.
"""

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from optimized_quant import OUT, ROOT, ResearchConfig, metric


DATA = ROOT / "data"


@dataclass(frozen=True)
class OverlayConfig:
    index_code: str = "000905"
    beta_window_months: int = 12
    minimum_beta_months: int = 12
    hedge_fraction: float = 0.50
    beta_lower: float = 0.0
    beta_upper: float = 1.5
    futures_trade_cost_bps: float = 5.0
    annual_roll_cost_bps: float = 100.0


def _members(archive: zipfile.ZipFile, prefix: str) -> list[str]:
    return [
        name for name in archive.namelist()
        if Path(name).name.startswith(prefix) and name.lower().endswith(".csv")
    ]


def index_monthly(index_code: str, start: str = "2018-01-01",
                  end: str = "2025-12-31") -> pd.DataFrame:
    pieces = []
    for path in sorted(DATA.glob("09_index_daily_*.zip")):
        with zipfile.ZipFile(path) as archive:
            for member in _members(archive, "IDX_Idxtrd"):
                with archive.open(member) as stream:
                    frame = pd.read_csv(
                        stream, usecols=["Indexcd", "Idxtrd01", "Idxtrd08"],
                        dtype={"Indexcd": "string"}, low_memory=False,
                    )
                frame["Indexcd"] = frame["Indexcd"].str.zfill(6)
                frame["date"] = pd.to_datetime(frame["Idxtrd01"], errors="coerce")
                frame = frame[
                    frame["Indexcd"].eq(index_code)
                    & frame["date"].between(start, end)
                ].copy()
                if frame.empty:
                    continue
                frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp("M")
                frame["daily_return"] = pd.to_numeric(frame["Idxtrd08"], errors="coerce") / 100
                pieces.append(frame[["date", "month", "daily_return"]])
    if not pieces:
        raise RuntimeError(f"No index observations for {index_code}")
    daily = pd.concat(pieces, ignore_index=True).drop_duplicates("date", keep="last")
    return daily.groupby("month", observed=True)["daily_return"].apply(
        lambda values: (1 + values.dropna()).prod() - 1
    ).rename("hedge_index_return").reset_index()


def causal_beta(portfolio_return: pd.Series, index_return: pd.Series,
                window: int, minimum: int, lower: float, upper: float) -> pd.Series:
    """Rolling beta shifted once so the current month's return is never used."""
    covariance = portfolio_return.rolling(window, min_periods=minimum).cov(index_return).shift(1)
    variance = index_return.rolling(window, min_periods=minimum).var().shift(1)
    return (covariance / variance).clip(lower, upper).fillna(0.0)


def apply_overlay(result: pd.DataFrame, config: OverlayConfig) -> pd.DataFrame:
    overlay = result.copy().sort_values("month")
    overlay["estimated_beta"] = causal_beta(
        overlay["gross_return"], overlay["hedge_index_return"],
        config.beta_window_months, config.minimum_beta_months,
        config.beta_lower, config.beta_upper,
    )
    overlay["hedge_notional"] = config.hedge_fraction * overlay["estimated_beta"]
    overlay["hedge_turnover"] = overlay["hedge_notional"].diff().abs()
    overlay.loc[overlay.index[0], "hedge_turnover"] = abs(overlay.iloc[0]["hedge_notional"])
    overlay["hedge_trade_cost"] = (
        overlay["hedge_turnover"] * config.futures_trade_cost_bps / 10_000
    )
    overlay["hedge_roll_cost"] = (
        overlay["hedge_notional"].abs() * config.annual_roll_cost_bps / 10_000 / 12
    )
    overlay["overlay_gross_return"] = (
        overlay["gross_return"]
        - overlay["hedge_notional"] * overlay["hedge_index_return"]
    )
    overlay["overlay_net_return"] = (
        overlay["net_return"]
        - overlay["hedge_notional"] * overlay["hedge_index_return"]
        - overlay["hedge_trade_cost"]
        - overlay["hedge_roll_cost"]
    )
    return overlay


def subperiod_metrics(result: pd.DataFrame, research: ResearchConfig) -> pd.DataFrame:
    periods = {
        "development": (research.start, research.development_end),
        "selection": (research.selection_start, research.selection_end),
        "retrospective_test": (research.retrospective_start, research.retrospective_end),
        "full": (research.start, research.retrospective_end),
    }
    rows = []
    for period, (start, end) in periods.items():
        subset = result[result["month"].between(start, end)]
        rows.append({
            "period": period,
            "series": "csi500_futures_overlay_proxy",
            **metric(subset["overlay_net_return"]),
            "mean_hedge_notional": subset["hedge_notional"].mean(),
            "mean_hedge_turnover": subset["hedge_turnover"].mean(),
        })
        rows.append({
            "period": period,
            "series": "unhedged_clean_mlp",
            **metric(subset["net_return"]),
            "mean_hedge_notional": 0.0,
            "mean_hedge_turnover": 0.0,
        })
    return pd.DataFrame(rows)


def cost_stress(result: pd.DataFrame, config: OverlayConfig) -> pd.DataFrame:
    rows = []
    for stock_cost in (20, 50, 100):
        for futures_cost in (5, 10, 20):
            returns = (
                result["gross_return"]
                - result["turnover"] * stock_cost / 10_000
                - result["hedge_notional"] * result["hedge_index_return"]
                - result["hedge_turnover"] * futures_cost / 10_000
                - result["hedge_notional"].abs() * config.annual_roll_cost_bps / 10_000 / 12
            )
            rows.append({
                "stock_cost_bps": stock_cost,
                "futures_trade_cost_bps": futures_cost,
                "annual_roll_cost_bps": config.annual_roll_cost_bps,
                **metric(returns),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", type=Path, default=OUT / "optimized_backtest.csv")
    args = parser.parse_args()
    config = OverlayConfig()
    research = ResearchConfig()
    portfolio = pd.read_csv(args.backtest, parse_dates=["month"])
    index = index_monthly(config.index_code)
    result = apply_overlay(portfolio.merge(index, on="month", how="left"), config)
    if result["hedge_index_return"].isna().any():
        raise RuntimeError("Missing CSI 500 return in overlay period")
    result.to_csv(OUT / "risk_overlay_backtest.csv", index=False)
    subperiod_metrics(result, research).to_csv(
        OUT / "risk_overlay_metrics.csv", index=False
    )
    cost_stress(result, config).to_csv(OUT / "risk_overlay_cost_stress.csv", index=False)
    print(subperiod_metrics(result, research).to_string(index=False))
    print(cost_stress(result, config).to_string(index=False))


if __name__ == "__main__":
    main()
