from __future__ import annotations

"""Shared, leakage-safe utilities for the residual-target research branch.

The module deliberately reuses ``prepare_advanced``.  It does not create a
second cleaning pipeline and it does not add predictors to the Ridge model.
The additional columns below are labels, risk diagnostics, or portfolio
constraints only.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_quant import FEATURES, OUT, prepare_advanced


FORMATION_START = pd.Timestamp("2014-01-01")
DEVELOPMENT_START = pd.Timestamp("2018-01-01")
SELECTION_START = pd.Timestamp("2022-01-01")
RETROSPECTIVE_START = pd.Timestamp("2024-01-01")
TRUE_FORWARD_START = pd.Timestamp("2026-01-01")
RANDOM_SEED = 20260723


def period_label(month: pd.Timestamp) -> str:
    month = pd.Timestamp(month)
    if month < DEVELOPMENT_START:
        return "formation"
    if month < SELECTION_START:
        return "development"
    if month < RETROSPECTIVE_START:
        return "selection"
    if month < TRUE_FORWARD_START:
        return "retrospective_test"
    return "true_forward"


def assert_selection_period_only(frame: pd.DataFrame, date_column: str = "signal_month") -> None:
    """Fail closed if a selector is ever handed 2024-or-later observations."""
    if frame.empty:
        return
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.ge(RETROSPECTIVE_START).any():
        raise ValueError("retrospective_test data must not enter parameter selection")


def metric(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe_rf0": np.nan,
            "max_drawdown": np.nan,
            "total_return": np.nan,
            "months": 0,
        }
    nav = (1.0 + values).cumprod()
    annual_return = float(nav.iloc[-1] ** (12.0 / len(values)) - 1.0)
    annual_volatility = float(values.std(ddof=1) * np.sqrt(12.0))
    drawdown = nav / nav.cummax() - 1.0
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_rf0": annual_return / annual_volatility if annual_volatility > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "total_return": float(nav.iloc[-1] - 1.0),
        "months": int(len(values)),
    }


def true_weight_turnover(current: dict[str, float], previous: dict[str, float]) -> float:
    if not previous:
        return 1.0 if current else 0.0
    names = set(current) | set(previous)
    return float(
        0.5
        * sum(abs(float(current.get(name, 0.0)) - float(previous.get(name, 0.0))) for name in names)
    )


def weighted_return(
    weights: dict[str, float], returns: pd.Series
) -> tuple[float, float]:
    """Return known contribution and unresolved weight without zero imputation."""
    if not weights:
        return np.nan, 0.0
    known = 0.0
    unresolved = 0.0
    for name, weight in weights.items():
        value = returns.get(name, np.nan)
        if pd.isna(value):
            unresolved += float(weight)
        else:
            known += float(weight) * float(value)
    return float(known), float(unresolved)


def _rolling_market_beta(panel: pd.DataFrame, window: int = 12, minimum: int = 8) -> pd.Series:
    """Monthly beta using only returns available at the signal month or earlier."""
    market = panel.groupby("month", observed=True)["ret_clean"].transform("mean")
    result = pd.Series(np.nan, index=panel.index, dtype=float)
    work = panel.assign(_market_return=market)
    for _, group in work.groupby("Stkcd", sort=False):
        stock = group["ret_clean"]
        market_return = group["_market_return"]
        covariance = stock.rolling(window, min_periods=minimum).cov(market_return)
        variance = market_return.rolling(window, min_periods=minimum).var()
        result.loc[group.index] = (covariance / variance.where(variance > 1e-12)).to_numpy()
    return result


def _compound_forward(group: pd.Series, horizon: int) -> pd.Series:
    shifted = [group.shift(-step) for step in range(1, horizon + 1)]
    values = pd.concat(shifted, axis=1)
    complete = values.notna().all(axis=1)
    compounded = (1.0 + values).prod(axis=1) - 1.0
    return compounded.where(complete)


def _stable_cross_sectional_residual(
    group: pd.DataFrame, return_column: str
) -> tuple[pd.Series, dict[str, object]]:
    exposure_columns = ["log_size", "market_beta", "volatility_exposure", "liquidity_exposure"]
    usable = group[return_column].notna()
    local = group.loc[usable, ["industry", return_column, *exposure_columns]].copy()
    result = pd.Series(np.nan, index=group.index, dtype=float)
    record: dict[str, object] = {
        "signal_month": group["month"].iloc[0],
        "return_column": return_column,
        "observations": int(len(local)),
        "design_columns": 0,
        "condition_number": np.nan,
        "intercept": np.nan,
        "beta_log_size": np.nan,
        "beta_market": np.nan,
        "beta_volatility": np.nan,
        "beta_liquidity": np.nan,
        "residual_mean": np.nan,
        "residual_std": np.nan,
        "regression_valid": False,
    }
    if local.empty:
        return result, record
    # Missing beta remains explicitly flagged in the exported panel.  For the
    # cross-section only, a median plus missing indicator prevents unnecessary
    # sample deletion while preserving the fact that beta was unavailable.
    style = local[exposure_columns].apply(pd.to_numeric, errors="coerce")
    missing = style.isna().astype(float)
    medians = style.median().fillna(0.0)
    style = style.fillna(medians)
    style_mean = style.mean()
    style_scale = style.std(ddof=0).replace(0.0, 1.0)
    style = (style - style_mean) / style_scale
    dummies = pd.get_dummies(local["industry"].fillna("UNKNOWN"), drop_first=True, dtype=float)
    design = np.column_stack(
        [
            np.ones(len(local)),
            style.to_numpy(float),
            missing.to_numpy(float),
            dummies.to_numpy(float),
        ]
    )
    y = pd.to_numeric(local[return_column], errors="coerce").to_numpy(float)
    record["design_columns"] = int(design.shape[1])
    if len(local) < max(30, design.shape[1] + 5):
        return result, record
    ridge = np.eye(design.shape[1]) * 1e-8
    ridge[0, 0] = 0.0
    gram = design.T @ design + ridge
    try:
        coefficients = np.linalg.solve(gram, design.T @ y)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(design, y, rcond=1e-10)[0]
    fitted = design @ coefficients
    residual = y - fitted
    result.loc[local.index] = residual
    record.update(
        {
            "condition_number": float(np.linalg.cond(gram)),
            "intercept": float(coefficients[0]),
            "beta_log_size": float(coefficients[1]),
            "beta_market": float(coefficients[2]),
            "beta_volatility": float(coefficients[3]),
            "beta_liquidity": float(coefficients[4]),
            "residual_mean": float(np.mean(residual)),
            "residual_std": float(np.std(residual, ddof=1)),
            "regression_valid": bool(np.isfinite(residual).all()),
        }
    )
    return result, record


def build_residual_panel(
    panel_path: Path = OUT / "monthly_panel.csv.gz",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the cleaned advanced panel and add only labels/risk diagnostics."""
    panel = prepare_advanced(panel_path).sort_values(["Stkcd", "month"]).copy()
    panel["period"] = panel["month"].map(period_label)
    panel["market_return"] = panel.groupby("month", observed=True)["ret_clean"].transform("mean")
    panel["market_beta"] = _rolling_market_beta(panel)
    panel["market_beta_missing"] = panel["market_beta"].isna()
    panel["log_size"] = np.log(panel["size_clean"].where(panel["size_clean"] > 0))
    panel["volatility_exposure"] = panel["volatility_clean"]
    panel["liquidity_exposure"] = np.log(panel["amount_clean"].where(panel["amount_clean"] > 0))
    panel["listing_years"] = panel["listed_days"] / 365.25
    panel["forward_return_1m"] = panel["forward_return"]
    panel["forward_return_3m"] = panel.groupby("Stkcd", sort=False)["ret_clean"].transform(
        lambda values: _compound_forward(values, 3)
    )
    panel["target_realization_end_1m"] = panel["month"] + pd.offsets.MonthEnd(1)
    panel["target_realization_end_3m"] = panel["month"] + pd.offsets.MonthEnd(3)

    regression_rows: list[dict[str, object]] = []
    for return_column, residual_column in (
        ("forward_return_1m", "residual_return_1m"),
        ("forward_return_3m", "residual_return_3m"),
    ):
        residual = pd.Series(np.nan, index=panel.index, dtype=float)
        for _, group in panel.groupby("month", sort=True):
            values, record = _stable_cross_sectional_residual(group, return_column)
            residual.loc[group.index] = values
            regression_rows.append(record)
        panel[residual_column] = residual
        panel[f"{residual_column}_target"] = (
            panel.groupby("month", observed=True)[residual_column].rank(pct=True) - 0.5
        )

    panel["original_1m_target"] = panel["target"]
    return panel, pd.DataFrame(regression_rows)


def purged_validation_split(
    history: pd.DataFrame,
    prediction_month: pd.Timestamp,
    horizon_months: int,
    validation_months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create an auditable validation split with a three-month gap for 3m labels."""
    prediction_month = pd.Timestamp(prediction_month)
    realization_column = f"target_realization_end_{horizon_months}m"
    fully_realized = history[pd.to_datetime(history[realization_column]) < prediction_month].copy()
    embargo_months = 3 if horizon_months == 3 else 0
    validation_end = prediction_month - pd.offsets.MonthEnd(embargo_months + 1)
    validation_start = validation_end - pd.offsets.MonthEnd(validation_months - 1)
    validation = fully_realized[
        fully_realized["month"].between(validation_start, validation_end)
    ].copy()
    purge_cutoff = validation_start - pd.offsets.MonthEnd(3 if horizon_months == 3 else 0)
    training = fully_realized[
        pd.to_datetime(fully_realized[realization_column]) < purge_cutoff
    ].copy()
    audit = {
        "prediction_month": prediction_month,
        "target_horizon_months": horizon_months,
        "training_signal_end": training["month"].max() if not training.empty else pd.NaT,
        "training_realization_end": (
            pd.to_datetime(training[realization_column]).max() if not training.empty else pd.NaT
        ),
        "validation_signal_start": validation["month"].min() if not validation.empty else pd.NaT,
        "validation_signal_end": validation["month"].max() if not validation.empty else pd.NaT,
        "validation_realization_end": (
            pd.to_datetime(validation[realization_column]).max() if not validation.empty else pd.NaT
        ),
        "purge_months": 3 if horizon_months == 3 else 0,
        "embargo_months": embargo_months,
    }
    audit["training_precedes_validation"] = bool(
        training.empty
        or validation.empty
        or pd.Timestamp(audit["training_realization_end"])
        < pd.Timestamp(audit["validation_signal_start"]) - pd.offsets.MonthEnd(2)
    )
    audit["validation_precedes_prediction"] = bool(
        validation.empty
        or pd.Timestamp(audit["validation_realization_end"]) < prediction_month
    )
    return training, validation, audit


def decile_labels(frame: pd.DataFrame, score_column: str) -> pd.Series:
    percentile = frame.groupby("industry", dropna=False)[score_column].rank(pct=True)
    percentile = percentile.fillna(frame[score_column].rank(pct=True))
    return np.ceil(percentile * 10.0).clip(1, 10).astype("Int64")


def quantile_bucket(values: pd.Series, buckets: int = 5) -> pd.Series:
    ranked = pd.to_numeric(values, errors="coerce").rank(method="first", pct=True)
    return np.ceil(ranked * buckets).clip(1, buckets).astype("Int64")


@dataclass(frozen=True)
class DataLimitations:
    market_beta: str = (
        "12-month monthly-return beta is used; daily beta was not added as a new predictor."
    )
    benchmark_weights: str = (
        "Historical index constituent weights are absent; the eligible monthly pool is the exposure reference."
    )
    market_impact: str = (
        "The square-root cost term is a liquidity stratification proxy, not a calibrated impact model."
    )
    shorting: str = (
        "Borrow availability, borrow fees and short-sale execution are absent; short legs remain diagnostic."
    )
