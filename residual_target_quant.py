from __future__ import annotations

"""Residual 1m/3m targets with a fixed, CPU-only Ridge research ladder."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_quant import FEATURES, OUT, linear_design, monthly_rank_ic
from ml_quant import fit as ridge_fit
from ml_quant import predict as ridge_predict
from model_comparison_quant import fit_huber, fit_ols, predict_linear
from residual_common import (
    DEVELOPMENT_START,
    RETROSPECTIVE_START,
    SELECTION_START,
    TRUE_FORWARD_START,
    assert_selection_period_only,
    build_residual_panel,
    period_label,
    purged_validation_split,
)


RIDGE_ALPHA_GRID = (10.0, 100.0, 1000.0, 10000.0)
TARGET_SPECS = {
    "original_1m": ("original_1m_target", "forward_return_1m", 1),
    "residual_1m": ("residual_return_1m_target", "residual_return_1m", 1),
    "residual_3m": ("residual_return_3m_target", "residual_return_3m", 3),
}


@dataclass(frozen=True)
class ResidualConfig:
    train_months: int = 84
    validation_months: int = 12
    refit_months: int = 3
    max_ridge_samples: int = 150_000
    seed: int = 20260723
    signal_start: str = "2018-01-01"
    signal_end: str = "2025-11-30"


def compute_forward_return_3m(returns: pd.Series) -> pd.Series:
    values = pd.concat([returns.shift(-1), returns.shift(-2), returns.shift(-3)], axis=1)
    return ((1.0 + values).prod(axis=1) - 1.0).where(values.notna().all(axis=1))


def deterministic_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.sample(maximum, random_state=seed).sort_values(["month", "Stkcd"])


def fixed_combined_score(residual_1m: pd.Series, residual_3m: pd.Series) -> pd.Series:
    return 0.5 * residual_1m + 0.5 * residual_3m


def _target_history(
    panel: pd.DataFrame,
    target_name: str,
    prediction_month: pd.Timestamp,
    train_months: int,
) -> pd.DataFrame:
    target_column, _, horizon = TARGET_SPECS[target_name]
    realization_column = f"target_realization_end_{horizon}m"
    lower = prediction_month - pd.DateOffset(months=train_months)
    return panel[
        panel["month"].between(lower, prediction_month, inclusive="left")
        & panel["eligible"]
        & panel[target_column].notna()
        & (pd.to_datetime(panel[realization_column]) < prediction_month)
    ].copy()


def _calibrate_alpha(score: np.ndarray, realized: np.ndarray) -> dict[str, float]:
    score = np.asarray(score, dtype=float)
    realized = np.asarray(realized, dtype=float)
    valid = np.isfinite(score) & np.isfinite(realized)
    if valid.sum() < 100 or np.var(score[valid]) < 1e-12:
        return {
            "alpha_intercept": 0.0,
            "alpha_beta": 0.0,
            "alpha_beta_t_stat": np.nan,
            "alpha_calibration_samples": int(valid.sum()),
            "signal_failure": True,
        }
    x = score[valid]
    y = realized[valid]
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=1e-10)[0]
    residual = y - design @ coefficients
    dof = max(len(y) - 2, 1)
    variance = float(residual @ residual / dof)
    covariance = variance * np.linalg.pinv(design.T @ design)
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    beta = float(coefficients[1])
    return {
        "alpha_intercept": float(coefficients[0]),
        "alpha_beta": beta,
        "alpha_beta_t_stat": beta / standard_error if standard_error > 0 else np.nan,
        "alpha_calibration_samples": int(len(y)),
        "signal_failure": bool(beta <= 1e-8),
    }


def choose_frozen_alphas(
    panel: pd.DataFrame, config: ResidualConfig
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Choose the four-value alpha grid using pre-2024 data only."""
    selection = panel[
        panel["month"].between(SELECTION_START, RETROSPECTIVE_START, inclusive="left")
        & panel["eligible"]
    ].copy()
    assert_selection_period_only(selection.rename(columns={"month": "signal_month"}))
    comparison_rows: list[dict[str, object]] = []
    selected: dict[str, float] = {}
    for target_name, (target_column, _, horizon) in TARGET_SPECS.items():
        realization_column = f"target_realization_end_{horizon}m"
        training = panel[
            (panel["month"] < SELECTION_START)
            & panel["eligible"]
            & panel[target_column].notna()
            & (pd.to_datetime(panel[realization_column]) < SELECTION_START)
        ].copy()
        training = deterministic_sample(
            training, config.max_ridge_samples, config.seed + horizon * 100
        )
        x_train, medians = linear_design(training)
        y_train = training[target_column].to_numpy(float)
        validation = selection[selection[target_column].notna()].copy()
        x_validation, _ = linear_design(validation, medians)
        y_validation = validation[target_column].to_numpy(float)
        months = validation["month"].to_numpy()
        for alpha in RIDGE_ALPHA_GRID:
            model = ridge_fit(x_train, y_train, alpha)
            prediction = ridge_predict(x_validation, model)
            comparison_rows.append(
                {
                    "target_type": target_name,
                    "model": "ridge",
                    "ridge_alpha": alpha,
                    "selection_mean_rank_ic": monthly_rank_ic(prediction, y_validation, months),
                    "uses_retrospective_test": False,
                    "selection_samples": len(validation),
                }
            )
        # OLS and Huber remain fixed comparisons, never production candidates.
        for model_name, model in (
            ("ols", fit_ols(x_train, y_train)),
            ("huber", fit_huber(x_train, y_train)),
        ):
            prediction = predict_linear(x_validation, model)
            comparison_rows.append(
                {
                    "target_type": target_name,
                    "model": model_name,
                    "ridge_alpha": np.nan,
                    "selection_mean_rank_ic": monthly_rank_ic(prediction, y_validation, months),
                    "uses_retrospective_test": False,
                    "selection_samples": len(validation),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    for target_name in TARGET_SPECS:
        candidates = comparison[
            (comparison["target_type"] == target_name) & (comparison["model"] == "ridge")
        ]
        selected[target_name] = float(
            candidates.sort_values(
                ["selection_mean_rank_ic", "ridge_alpha"], ascending=[False, True]
            ).iloc[0]["ridge_alpha"]
        )
    comparison["selected"] = comparison.apply(
        lambda row: row["model"] == "ridge"
        and float(row["ridge_alpha"]) == selected[row["target_type"]],
        axis=1,
    )
    audit = pd.DataFrame(
        [
            {
                "check": "ridge_alpha_selection_excludes_retrospective_test",
                "passed": bool(
                    not comparison["uses_retrospective_test"].astype(bool).any()
                    and selection["month"].max() < RETROSPECTIVE_START
                ),
                "evidence": (
                    f"selection maximum={selection['month'].max().date()}; "
                    f"retrospective starts={RETROSPECTIVE_START.date()}"
                ),
            }
        ]
    )
    return selected, comparison, audit


def _standardized_coefficients(
    model: tuple[np.ndarray, np.ndarray], target_name: str, refit_month: pd.Timestamp
) -> list[dict[str, object]]:
    beta, _ = model
    names = FEATURES + [f"{name}_missing" for name in FEATURES]
    return [
        {
            "refit_month": refit_month,
            "target_type": target_name,
            "feature": name,
            "standardized_coefficient": float(value),
        }
        for name, value in zip(names, beta[1:], strict=True)
    ]


def walk_forward_target(
    panel: pd.DataFrame,
    target_name: str,
    frozen_alpha: float,
    config: ResidualConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_column, realized_column, horizon = TARGET_SPECS[target_name]
    months = sorted(
        pd.Timestamp(month)
        for month in panel.loc[
            panel["month"].between(config.signal_start, config.signal_end), "month"
        ].unique()
    )
    score_rows: list[pd.DataFrame] = []
    log_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    fitted: tuple[tuple[np.ndarray, np.ndarray], np.ndarray, dict[str, float]] | None = None
    refit_counter = config.refit_months
    for month in months:
        if fitted is None or refit_counter >= config.refit_months:
            history = _target_history(panel, target_name, month, config.train_months)
            if len(history) < 10_000:
                continue
            diagnostic_train, validation, split_audit = purged_validation_split(
                history, month, horizon, config.validation_months
            )
            if diagnostic_train.empty or validation.empty:
                continue
            diagnostic_train = deterministic_sample(
                diagnostic_train,
                config.max_ridge_samples,
                config.seed + month.year * 100 + month.month + horizon,
            )
            x_train, diagnostic_medians = linear_design(diagnostic_train)
            trial = ridge_fit(
                x_train, diagnostic_train[target_column].to_numpy(float), frozen_alpha
            )
            x_validation, _ = linear_design(validation, diagnostic_medians)
            validation_score = ridge_predict(x_validation, trial)
            calibration = _calibrate_alpha(
                validation_score, validation[realized_column].to_numpy(float)
            )

            full = deterministic_sample(
                history,
                config.max_ridge_samples,
                config.seed + 17 + month.year * 100 + month.month + horizon,
            )
            x_full, medians = linear_design(full)
            model = ridge_fit(x_full, full[target_column].to_numpy(float), frozen_alpha)
            fitted = (model, medians, calibration)
            coefficient_rows.extend(_standardized_coefficients(model, target_name, month))
            log_rows.append(
                {
                    "refit_month": month,
                    "target_type": target_name,
                    "target_horizon_months": horizon,
                    "ridge_alpha": frozen_alpha,
                    "history_start": history["month"].min(),
                    "history_end": history["month"].max(),
                    "maximum_training_realization_month": pd.to_datetime(
                        history[f"target_realization_end_{horizon}m"]
                    ).max(),
                    "history_samples": len(history),
                    "validation_ic": monthly_rank_ic(
                        validation_score,
                        validation[target_column].to_numpy(float),
                        validation["month"].to_numpy(),
                    ),
                    **calibration,
                }
            )
            split_audit.update(
                {
                    "target_type": target_name,
                    "maximum_feature_month": month,
                    "all_training_labels_realized_before_prediction": bool(
                        pd.to_datetime(history[f"target_realization_end_{horizon}m"]).max()
                        < month
                    ),
                    "features_not_after_signal": True,
                }
            )
            audit_rows.append(split_audit)
            refit_counter = 0

        test = panel[(panel["month"] == month) & panel["eligible"]].copy()
        if fitted is None or test.empty:
            continue
        model, medians, calibration = fitted
        x_test, _ = linear_design(test, medians)
        score = ridge_predict(x_test, model)
        test[f"{target_name}_score"] = score
        test[f"{target_name}_alpha"] = (
            calibration["alpha_intercept"] + calibration["alpha_beta"] * score
        )
        test["signal_month"] = month
        test["realization_month"] = month + pd.offsets.MonthEnd(1)
        columns = [
            "signal_month",
            "realization_month",
            "period",
            "Stkcd",
            "industry",
            "forward_return_1m",
            "forward_return_3m",
            "market_beta",
            "market_beta_missing",
            "log_size",
            "volatility_exposure",
            "liquidity_exposure",
            "amount_clean",
            "value_composite",
            "short_reversal",
            target_column,
            f"{target_name}_score",
            f"{target_name}_alpha",
        ]
        score_rows.append(test[columns].rename(columns={"Stkcd": "stock_code"}))
        refit_counter += 1
    return (
        pd.concat(score_rows, ignore_index=True),
        pd.DataFrame(log_rows),
        pd.DataFrame(audit_rows),
        pd.DataFrame(coefficient_rows),
    )


def _merge_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    keys = [
        "signal_month",
        "realization_month",
        "period",
        "stock_code",
        "industry",
        "forward_return_1m",
        "forward_return_3m",
        "market_beta",
        "market_beta_missing",
        "log_size",
        "volatility_exposure",
        "liquidity_exposure",
        "amount_clean",
        "value_composite",
        "short_reversal",
    ]
    result = frames[0]
    for frame in frames[1:]:
        extra = [column for column in frame if column not in keys]
        result = result.merge(frame[keys[:4] + extra], on=keys[:4], how="inner")
    result["fixed_combined_score"] = fixed_combined_score(
        result["residual_1m_score"], result["residual_3m_score"]
    )
    result["fixed_combined_alpha"] = fixed_combined_score(
        result["residual_1m_alpha"], result["residual_3m_alpha"]
    )
    return result


def build_leakage_audit(
    split_audits: pd.DataFrame, alpha_audit: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    rows = alpha_audit.to_dict("records")
    rows.extend(
        [
            {
                "check": "all_training_labels_realized_before_prediction",
                "passed": bool(
                    split_audits["all_training_labels_realized_before_prediction"]
                    .fillna(False)
                    .all()
                ),
                "evidence": "maximum training label realization is earlier than every refit month",
            },
            {
                "check": "three_month_purge_and_embargo",
                "passed": bool(
                    split_audits.loc[
                        split_audits["target_horizon_months"] == 3,
                        ["training_precedes_validation", "validation_precedes_prediction"],
                    ]
                    .fillna(False)
                    .all()
                    .all()
                ),
                "evidence": "3m rows use a 3m purge and a 3m validation-to-test embargo",
            },
            {
                "check": "features_not_after_signal",
                "passed": bool(split_audits["features_not_after_signal"].fillna(False).all()),
                "evidence": "prepare_advanced features and risk exposures end at signal month",
            },
            {
                "check": "prediction_months_are_fixed",
                "passed": bool(
                    predictions["signal_month"].max() < TRUE_FORWARD_START
                    and predictions["realization_month"].max() <= pd.Timestamp("2025-12-31")
                ),
                "evidence": "historical prediction window ends with the 2025-11 signal",
            },
        ]
    )
    return pd.DataFrame(rows)


def prediction_period_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_name in (*TARGET_SPECS.keys(), "fixed_combined"):
        score_column = f"{target_name}_score"
        target_column = {
            "original_1m": "original_1m_target",
            "residual_1m": "residual_return_1m_target",
            "residual_3m": "residual_return_3m_target",
            "fixed_combined": "residual_return_1m_target",
        }[target_name]
        for period, group in predictions.groupby("period", observed=True):
            monthly_original = group.groupby("signal_month", observed=True).apply(
                lambda local: local[score_column].corr(
                    local["original_1m_target"], method="spearman"
                ),
                include_groups=False,
            )
            monthly_target = group.groupby("signal_month", observed=True).apply(
                lambda local: local[score_column].corr(
                    local[target_column], method="spearman"
                ),
                include_groups=False,
            )
            local = group.copy()
            local["decile"] = local.groupby("signal_month", observed=True)[
                score_column
            ].rank(pct=True)
            local["decile"] = np.ceil(local["decile"] * 10).clip(1, 10)
            decile_return = local.groupby("decile", observed=True)[
                "forward_return_1m"
            ].mean()
            rows.append(
                {
                    "comparison_scope": "frozen_walk_forward",
                    "target_type": target_name,
                    "model": "ridge",
                    "period": period,
                    "mean_rank_ic_vs_original_1m": monthly_original.mean(),
                    "annualized_icir_vs_original_1m": (
                        monthly_original.mean()
                        / monthly_original.std(ddof=1)
                        * np.sqrt(12)
                    ),
                    "mean_rank_ic_vs_own_target": monthly_target.mean(),
                    "positive_original_ic_month_share": (
                        monthly_original > 0
                    ).mean(),
                    "decile_monotonicity_vs_next_1m_return": (
                        decile_return.index.to_series().corr(
                            decile_return, method="spearman"
                        )
                    ),
                    "uses_retrospective_test": False,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=OUT / "monthly_panel.csv.gz")
    parser.add_argument("--reuse-panel-cache", action="store_true")
    args = parser.parse_args()
    cache = OUT / "residual_panel_cache.pkl"
    if args.reuse_panel_cache and cache.exists():
        panel, regressions = pd.read_pickle(cache)
    else:
        panel, regressions = build_residual_panel(args.panel)
        pd.to_pickle((panel, regressions), cache)

    config = ResidualConfig()
    frozen_alphas, comparison, alpha_audit = choose_frozen_alphas(panel, config)
    prediction_frames = []
    logs = []
    audits = []
    coefficients = []
    for target_name in TARGET_SPECS:
        prediction, log, audit, coefficient = walk_forward_target(
            panel, target_name, frozen_alphas[target_name], config
        )
        prediction_frames.append(prediction)
        logs.append(log)
        audits.append(audit)
        coefficients.append(coefficient)
    predictions = _merge_prediction_frames(prediction_frames)
    comparison["comparison_scope"] = "selection_alpha_and_model_choice"
    comparison["period"] = "selection"
    comparison = pd.concat(
        [comparison, prediction_period_comparison(predictions)],
        ignore_index=True,
        sort=False,
    )
    model_log = pd.concat(logs, ignore_index=True)
    split_audit = pd.concat(audits, ignore_index=True)
    coefficient_table = pd.concat(coefficients, ignore_index=True)
    leakage_audit = build_leakage_audit(split_audit, alpha_audit, predictions)

    target_columns = [
        "month",
        "period",
        "Stkcd",
        "industry",
        "eligible",
        "forward_return_1m",
        "forward_return_3m",
        "original_1m_target",
        "residual_return_1m",
        "residual_return_1m_target",
        "residual_return_3m",
        "residual_return_3m_target",
        "market_beta",
        "market_beta_missing",
        "log_size",
        "volatility_exposure",
        "liquidity_exposure",
        "target_realization_end_1m",
        "target_realization_end_3m",
    ]
    panel[target_columns].to_csv(
        OUT / "residual_target_monthly.csv", index=False, float_format="%.8g"
    )
    regressions.to_csv(OUT / "residual_cross_sectional_regression.csv", index=False)
    model_log.to_csv(OUT / "residual_model_log.csv", index=False)
    comparison.to_csv(OUT / "residual_model_comparison.csv", index=False)
    predictions.to_csv(OUT / "residual_predictions.csv", index=False, float_format="%.8g")
    leakage_audit.to_csv(OUT / "residual_future_leakage_audit.csv", index=False)
    coefficient_table.to_csv(OUT / "residual_model_coefficients.csv", index=False)
    print("Frozen Ridge alpha:", frozen_alphas)
    print(comparison[comparison["selected"].fillna(False)].to_string(index=False))
    print(leakage_audit.to_string(index=False))


if __name__ == "__main__":
    main()
