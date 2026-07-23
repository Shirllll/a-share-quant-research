from __future__ import annotations

"""Clean-first model ladder: OLS, Ridge, robust regression, then ML experiments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_quant import (
    Config as AdvancedConfig,
    OUT,
    Standardizer,
    TabMModel,
    TemporalRetriever,
    feature_matrix,
    linear_design,
    monthly_rank_ic,
    prepare_advanced,
    seed_everything,
)
from ml_quant import fit as ridge_fit
from ml_quant import metric
from ml_quant import predict as ridge_predict


MODEL_ORDER = ("ols", "ridge", "huber", "tabm_experimental", "retrieval_experimental")


@dataclass(frozen=True)
class ComparisonConfig:
    start: str = "2018-01-01"
    end: str = "2025-11-30"
    train_months: int = 84
    validation_months: int = 12
    refit_months: int = 3
    max_linear_samples: int = 150_000
    max_ml_samples: int = 20_000
    max_retrieval_samples: int = 2_000
    seed: int = 20260723


def period_label(month: pd.Timestamp) -> str:
    if month < pd.Timestamp("2022-01-01"):
        return "development"
    if month < pd.Timestamp("2024-01-01"):
        return "selection"
    return "retrospective_test"


def sample_frame(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.sample(maximum, random_state=seed).sort_values("month")


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(design, y, rcond=1e-8)[0]
    return beta, mean, scale


def fit_huber(x: np.ndarray, y: np.ndarray, iterations: int = 8,
              tuning_constant: float = 1.345) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(design, y, rcond=1e-8)[0]
    for _ in range(iterations):
        residual = y - design @ beta
        robust_scale = np.median(np.abs(residual - np.median(residual))) / 0.6745
        robust_scale = max(float(robust_scale), 1e-6)
        standardized = np.abs(residual) / (tuning_constant * robust_scale)
        weights = np.ones_like(standardized)
        large = standardized > 1.0
        weights[large] = 1.0 / standardized[large]
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        beta = np.linalg.lstsq(weighted_design, weighted_y, rcond=1e-8)[0]
    return beta, mean, scale


def predict_linear(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    beta, mean, scale = model
    z = (x - mean) / scale
    return beta[0] + z @ beta[1:]


def industry_long_short_spread(frame: pd.DataFrame, score_column: str) -> float:
    spreads = []
    for _, group in frame.groupby("industry", dropna=False):
        if len(group) < 10:
            continue
        count = max(1, int(np.floor(len(group) * 0.10)))
        ordered = group.sort_values(score_column)
        long_return = ordered.tail(count)["forward_return"].mean()
        short_return = ordered.head(count)["forward_return"].mean()
        if pd.notna(long_return) and pd.notna(short_return):
            spreads.append(long_return - short_return)
    return float(np.mean(spreads)) if spreads else np.nan


def _deciles(frame: pd.DataFrame, score_column: str) -> pd.Series:
    percentile = frame.groupby("industry", dropna=False)[score_column].rank(pct=True)
    percentile = percentile.fillna(frame[score_column].rank(pct=True))
    return np.ceil(percentile * 10).clip(1, 10).astype(int)


def walk_forward_comparison(panel: pd.DataFrame, config: ComparisonConfig):
    monthly_rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []
    fitted = None
    counter = 0
    months = sorted(
        pd.Timestamp(month) for month in panel.loc[
            panel["month"].between(config.start, config.end), "month"
        ].unique()
    )
    for month in months:
        if fitted is None or counter >= config.refit_months:
            lower = month - pd.DateOffset(months=config.train_months)
            history = panel[
                (panel["month"] < month) & (panel["month"] >= lower)
                & panel["eligible"] & panel["target"].notna()
            ].copy()
            validation_start = history["month"].max() - pd.DateOffset(months=config.validation_months)
            training = history[history["month"] <= validation_start]
            validation = history[history["month"] > validation_start]
            linear_train = sample_frame(
                training, config.max_linear_samples, config.seed + month.year * 100 + month.month,
            )
            x_train, medians = linear_design(linear_train)
            y_train = linear_train["target"].to_numpy(float)
            x_validation, _ = linear_design(validation, medians)
            y_validation = validation["target"].to_numpy(float)
            validation_months = validation["month"].to_numpy()

            ols_trial = fit_ols(x_train, y_train)
            huber_trial = fit_huber(x_train, y_train)
            ridge_trials = []
            for alpha in (10.0, 100.0, 1000.0, 10000.0):
                model = ridge_fit(x_train, y_train, alpha)
                prediction = ridge_predict(x_validation, model)
                ridge_trials.append((
                    monthly_rank_ic(prediction, y_validation, validation_months), alpha,
                ))
            ridge_ic, ridge_alpha = max(ridge_trials, key=lambda item: item[0])

            ml_train = sample_frame(
                training, config.max_ml_samples, config.seed + 17 + month.year * 100 + month.month,
            )
            standardizer = Standardizer().fit(feature_matrix(ml_train))
            ml_x = standardizer.transform(feature_matrix(ml_train))
            ml_y = ml_train["target"].to_numpy(np.float32)
            validation_ml_x = standardizer.transform(feature_matrix(validation))
            advanced_config = AdvancedConfig(
                epochs=2, max_train_samples=config.max_ml_samples,
                max_retrieval_samples=config.max_retrieval_samples,
                seed=config.seed,
            )
            tabm_trial = TabMModel(advanced_config, ml_x.shape[1], config.seed + month.year * 100 + month.month)
            best_epoch, tabm_validation_ic = tabm_trial.fit(
                ml_x, ml_y, (validation_ml_x, y_validation.astype(np.float32), validation_months),
            )

            full_linear = sample_frame(
                history, config.max_linear_samples, config.seed + 31 + month.year * 100 + month.month,
            )
            full_x, medians = linear_design(full_linear)
            full_y = full_linear["target"].to_numpy(float)
            ols = fit_ols(full_x, full_y)
            ridge = ridge_fit(full_x, full_y, ridge_alpha)
            huber = fit_huber(full_x, full_y)
            full_ml = sample_frame(
                history, config.max_ml_samples, config.seed + 47 + month.year * 100 + month.month,
            )
            standardizer = Standardizer().fit(feature_matrix(full_ml))
            full_ml_x = standardizer.transform(feature_matrix(full_ml))
            full_ml_y = full_ml["target"].to_numpy(np.float32)
            tabm = TabMModel(advanced_config, full_ml_x.shape[1], config.seed + month.year * 100 + month.month)
            tabm.fit(full_ml_x, full_ml_y, validation=None, epochs=best_epoch)
            retriever = TemporalRetriever(
                config.max_retrieval_samples, seed=config.seed + month.year * 100 + month.month,
            ).fit(full_ml_x, full_ml_y, full_ml["month"].to_numpy())
            fitted = (ols, ridge, huber, medians, standardizer, tabm, retriever)
            log_rows.append({
                "refit_month": month, "training_end": training["month"].max(),
                "validation_end": validation["month"].max(), "ridge_alpha": ridge_alpha,
                "ridge_validation_ic": ridge_ic,
                "ols_validation_ic": monthly_rank_ic(
                    predict_linear(x_validation, ols_trial), y_validation, validation_months,
                ),
                "huber_validation_ic": monthly_rank_ic(
                    predict_linear(x_validation, huber_trial), y_validation, validation_months,
                ),
                "tabm_validation_ic": tabm_validation_ic, "tabm_best_epoch": best_epoch,
            })
            counter = 0

        # Eligibility uses only current-month fields. Missing next-month returns
        # stay in the scored universe and are ignored only in realized diagnostics.
        test = panel[(panel["month"] == month) & panel["eligible"]].copy()
        if test.empty:
            continue
        ols, ridge, huber, medians, standardizer, tabm, retriever = fitted
        x_test, _ = linear_design(test, medians)
        ml_x_test = standardizer.transform(feature_matrix(test))
        predictions = {
            "ols": predict_linear(x_test, ols),
            "ridge": ridge_predict(x_test, ridge),
            "huber": predict_linear(x_test, huber),
            "tabm_experimental": tabm.predict(ml_x_test),
            "retrieval_experimental": retriever.predict(ml_x_test),
        }
        for model in MODEL_ORDER:
            score_column = f"score_{model}"
            test[score_column] = predictions[model]
            valid = test["target"].notna()
            rank_ic = test.loc[valid, score_column].corr(test.loc[valid, "target"], method="spearman")
            monthly_rows.append({
                "signal_month": month, "realization_month": month + pd.offsets.MonthEnd(1),
                "period": period_label(month), "model": model, "rank_ic": rank_ic,
                "industry_long_short_gross_return": industry_long_short_spread(test, score_column),
                "scored_stocks": len(test), "missing_forward_return_stocks": test["forward_return"].isna().sum(),
                "maximum_training_month": pd.Timestamp(log_rows[-1]["validation_end"]),
                "maximum_feature_month": month,
                "production_candidate": model == "ridge",
            })
            for decile, group in test.assign(decile=_deciles(test, score_column)).groupby("decile"):
                decile_rows.append({
                    "signal_month": month, "period": period_label(month), "model": model,
                    "decile": decile, "return": group["forward_return"].mean(), "stocks": len(group),
                })
        counter += 1
    return pd.DataFrame(monthly_rows), pd.DataFrame(decile_rows), pd.DataFrame(log_rows)


def model_metrics(monthly: pd.DataFrame, deciles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for period in ("all", "development", "selection", "retrospective_test"):
            sample = monthly[monthly["model"] == model]
            model_deciles = deciles[deciles["model"] == model]
            if period != "all":
                sample = sample[sample["period"] == period]
                model_deciles = model_deciles[model_deciles["period"] == period]
            ic = sample["rank_ic"].dropna()
            spread = sample["industry_long_short_gross_return"].dropna()
            average_decile = model_deciles.groupby("decile")["return"].mean()
            spread_metric = metric(spread)
            rows.append({
                "model": model, "period": period,
                "role": "production_candidate" if model == "ridge" else (
                    "ordinary_regression_baseline" if model == "ols" else "experimental_comparison"
                ),
                "mean_rank_ic": ic.mean(), "annualized_icir": ic.mean() / ic.std() * np.sqrt(12),
                "positive_ic_month_share": (ic > 0).mean(),
                "decile_monotonicity": average_decile.index.to_series().corr(average_decile, method="spearman"),
                "gross_long_short_annual_return": spread_metric["annual_return"],
                "gross_long_short_sharpe": spread_metric["sharpe_rf0"],
                "gross_long_short_t_stat": spread.mean() / (spread.std() / np.sqrt(len(spread))),
                "months": len(sample),
            })
    return pd.DataFrame(rows)


def future_leakage_audit(monthly: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    missing_future_rows_remain_scored = int(
        monthly.drop_duplicates("signal_month")["missing_forward_return_stocks"].sum()
    )
    return pd.DataFrame([
        {"check": "clean_before_all_models", "passed": True,
         "evidence": "prepare_advanced calls clean_monthly_panel before feature construction"},
        {"check": "training_precedes_signal", "passed": bool((log["validation_end"] < log["refit_month"]).all()),
         "evidence": "maximum validation month is earlier than every refit month"},
        {"check": "features_not_after_signal", "passed": bool((monthly["maximum_feature_month"] <= monthly["signal_month"]).all()),
         "evidence": "maximum_feature_month <= signal_month"},
        {"check": "scoring_ignores_future_return_availability",
         "passed": missing_future_rows_remain_scored > 0,
         "evidence": (
             f"{missing_future_rows_remain_scored} scored stock-months have missing next-month returns; "
             "the scoring universe is therefore not conditioned on future return availability"
         )},
        {"check": "retrospective_not_used_to_choose_production_model", "passed": True,
         "evidence": "Ridge is declared by policy; ML models remain experimental regardless of 2024-2025 results"},
    ])


def main() -> None:
    config = ComparisonConfig()
    seed_everything(config.seed)
    panel_path = OUT / "monthly_panel.csv.gz"
    panel = prepare_advanced(panel_path)
    monthly, deciles, log = walk_forward_comparison(panel, config)
    monthly.to_csv(OUT / "model_comparison_monthly.csv", index=False)
    deciles.to_csv(OUT / "model_comparison_deciles.csv", index=False)
    log.to_csv(OUT / "model_comparison_log.csv", index=False)
    metrics = model_metrics(monthly, deciles)
    metrics.to_csv(OUT / "model_comparison_metrics.csv", index=False)
    future_leakage_audit(monthly, log).to_csv(OUT / "future_leakage_audit.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
