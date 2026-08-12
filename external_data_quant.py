from __future__ import annotations

"""Point-in-time sell-side research data extension for the frozen A-share model.

The external source is Eastmoney's public research-report endpoint.  Raw data are
cached locally under ``data/`` and are intentionally excluded from Git.  Report
features are formed strictly from reports whose publication date is no later than
the signal month-end.
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from deep_learning_quant import MLP, ridge_fit, ridge_predict, sampled
from optimized_quant import (
    FEATURES,
    OUT,
    ROOT,
    ResearchConfig,
    build_portfolio,
    cost_stress,
    metric,
    monthly_rank_ic,
    period_metric,
    prepare_clean,
    select_parameters,
    signal_for_model,
)


REPORT_API = "https://reportapi.eastmoney.com/report/list"
REPORT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/127.0 Safari/537.36",
    "Referer": "https://data.eastmoney.com/report/",
}
RAW_REPORTS = ROOT / "data" / "external_analyst_reports.csv.gz"
EXTERNAL_FEATURES = [
    "analyst_attention",
    "analyst_breadth",
    "analyst_rating",
    "analyst_eps_growth",
    "analyst_eps_revision",
]


@dataclass(frozen=True)
class ExternalConfig:
    source_start: str = "2017-01-01"
    source_end: str = "2025-12-31"
    workers: int = 1
    request_timeout_seconds: int = 30
    request_retries: int = 8
    attention_days: int = 90
    detail_days: int = 180
    minimum_report_coverage: float = 0.10
    random_seed: int = 20260812


def _request_params(start: str, end: str, page: int) -> dict[str, str]:
    return {
        "industryCode": "*", "pageSize": "100", "industry": "*",
        "rating": "*", "ratingChange": "*", "beginTime": start,
        "endTime": end, "pageNo": str(page), "fields": "", "qType": "0",
        "orgCode": "", "code": "", "rcode": "", "p": str(page),
        "pageNum": str(page), "pageNumber": str(page),
    }


def _fetch_page(start: str, end: str, page: int, timeout: int,
                retries: int) -> list[dict[str, object]]:
    for attempt in range(retries):
        try:
            response = requests.get(
                REPORT_API, params=_request_params(start, end, page),
                headers=REPORT_HEADERS, timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            return list(payload.get("data") or [])
        except (requests.RequestException, ValueError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(20.0, 0.75 * (2 ** attempt)) + 0.10)
    raise RuntimeError("unreachable")


def _normalise_reports(records: list[dict[str, object]]) -> pd.DataFrame:
    raw = pd.DataFrame(records)
    columns = {
        "stockCode": "Stkcd", "publishDate": "publish_date",
        "orgCode": "org_code", "orgName": "org_name",
        "emRatingValue": "rating_value", "emRatingName": "rating_name",
        "ratingChange": "rating_change",
        "predictThisYearEps": "eps_current",
        "predictNextYearEps": "eps_next",
        "predictNextTwoYearEps": "eps_next_two",
        "predictThisYearPe": "pe_current", "predictNextYearPe": "pe_next",
        "infoCode": "report_id", "title": "report_title",
    }
    missing = set(columns) - set(raw.columns)
    if missing:
        raise RuntimeError(f"External report response missing fields: {sorted(missing)}")
    reports = raw[list(columns)].rename(columns=columns)
    reports["Stkcd"] = reports["Stkcd"].astype("string").str.zfill(6)
    reports["publish_date"] = pd.to_datetime(reports["publish_date"], errors="coerce")
    for column in [
        "rating_value", "rating_change", "eps_current", "eps_next",
        "eps_next_two", "pe_current", "pe_next",
    ]:
        reports[column] = pd.to_numeric(reports[column], errors="coerce")
    reports = reports.dropna(subset=["Stkcd", "publish_date", "report_id"])
    return reports.drop_duplicates("report_id", keep="last").sort_values(
        ["Stkcd", "publish_date", "report_id"]
    )


def _year_cache(destination: Path, year: int) -> Path:
    return destination.parent / f"external_analyst_reports_{year}.csv.gz"


def _page_cache(destination: Path, year: int, page: int) -> Path:
    return destination.parent / "external_report_pages" / f"{year}_{page:04d}.json"


def download_reports(config: ExternalConfig, destination: Path = RAW_REPORTS) -> pd.DataFrame:
    """Download annual partitions with checkpoints, then combine the snapshot."""
    start = pd.Timestamp(config.source_start)
    end = pd.Timestamp(config.source_end)
    annual_frames = []
    for year in range(start.year, end.year + 1):
        cache = _year_cache(destination, year)
        if cache.exists():
            annual_frames.append(load_reports(cache))
            print(f"reused report checkpoint {year}", flush=True)
            continue
        year_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        first_response = requests.get(
            REPORT_API,
            params=_request_params(str(year_start.date()), str(year_end.date()), 1),
            headers=REPORT_HEADERS,
            timeout=config.request_timeout_seconds,
        )
        first_response.raise_for_status()
        first = first_response.json()
        total_pages = int(first["TotalPage"])
        records = []
        for page in range(1, total_pages + 1):
            page_path = _page_cache(destination, year, page)
            if page_path.exists():
                page_records = json.loads(page_path.read_text(encoding="utf-8"))
            else:
                page_records = list(first.get("data") or []) if page == 1 else _fetch_page(
                    str(year_start.date()), str(year_end.date()), page,
                    config.request_timeout_seconds, config.request_retries,
                )
                page_path.parent.mkdir(parents=True, exist_ok=True)
                page_path.write_text(
                    json.dumps(page_records, ensure_ascii=False), encoding="utf-8"
                )
                time.sleep(0.35)
            records.extend(page_records)
            if page % 25 == 0 or page == total_pages:
                print(f"{year}: completed {page}/{total_pages} pages", flush=True)
        annual = _normalise_reports(records)
        if len(annual) != int(first["hits"]):
            raise RuntimeError(
                f"{year} report count mismatch: {len(annual)} != {first['hits']}"
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        annual.to_csv(cache, index=False, compression="gzip")
        annual_frames.append(annual)
        print(f"saved report checkpoint {year}: {len(annual)} records", flush=True)

    reports = pd.concat(annual_frames, ignore_index=True)
    reports = reports.drop_duplicates("report_id", keep="last").sort_values(
        ["Stkcd", "publish_date", "report_id"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    reports.to_csv(destination, index=False, compression="gzip")
    return reports


def load_reports(path: Path = RAW_REPORTS) -> pd.DataFrame:
    reports = pd.read_csv(
        path, compression="gzip", parse_dates=["publish_date"], dtype={"Stkcd": "string"}
    )
    reports["Stkcd"] = reports["Stkcd"].str.zfill(6)
    return reports.sort_values(["Stkcd", "publish_date", "report_id"])


def _bounded_growth(current: pd.Series, future: pd.Series) -> pd.Series:
    valid = current.abs().ge(0.05) & current.notna() & future.notna()
    growth = (future / current.abs() - 1).where(valid)
    return growth.clip(-1.0, 3.0)


def point_in_time_features(monthly_universe: pd.DataFrame, reports: pd.DataFrame,
                           config: ExternalConfig) -> pd.DataFrame:
    """Build stock-month features with an explicit source-date audit column."""
    universe = monthly_universe[["month", "Stkcd"]].drop_duplicates().copy()
    universe["Stkcd"] = universe["Stkcd"].astype("string").str.zfill(6)
    universe["month"] = pd.to_datetime(universe["month"])
    work = reports.copy()
    work["eps_growth"] = _bounded_growth(work["eps_current"], work["eps_next"])
    rows: list[pd.DataFrame] = []
    grouped_reports = {code: group for code, group in work.groupby("Stkcd", sort=False)}
    for code, months in universe.groupby("Stkcd", sort=False):
        stock_reports = grouped_reports.get(code)
        month_values = months["month"].sort_values().to_numpy(dtype="datetime64[ns]")
        result = pd.DataFrame({"month": month_values, "Stkcd": code})
        if stock_reports is None:
            result["report_count_90d"] = 0.0
            result["org_count_180d"] = 0.0
            result["rating_mean_180d"] = np.nan
            result["eps_growth_180d"] = np.nan
            result["external_max_publish_date"] = pd.NaT
        else:
            stock_reports = stock_reports.sort_values("publish_date")
            dates = stock_reports["publish_date"].to_numpy(dtype="datetime64[ns]")
            right = np.searchsorted(dates, month_values, side="right")
            left90 = np.searchsorted(
                dates, month_values - np.timedelta64(config.attention_days, "D"),
                side="right",
            )
            left180 = np.searchsorted(
                dates, month_values - np.timedelta64(config.detail_days, "D"),
                side="right",
            )
            rating = stock_reports["rating_value"].to_numpy(float)
            eps_growth = stock_reports["eps_growth"].to_numpy(float)
            result["report_count_90d"] = (right - left90).astype(float)
            organisations = stock_reports["org_code"].astype("string").to_numpy()
            result["org_count_180d"] = [
                float(pd.unique(organisations[left:stop][pd.notna(organisations[left:stop])]).size)
                for left, stop in zip(left180, right)
            ]
            result["rating_mean_180d"] = [
                np.nanmean(rating[left:stop]) if np.isfinite(rating[left:stop]).any() else np.nan
                for left, stop in zip(left180, right)
            ]
            result["eps_growth_180d"] = [
                np.nanmedian(eps_growth[left:stop])
                if np.isfinite(eps_growth[left:stop]).any() else np.nan
                for left, stop in zip(left180, right)
            ]
            max_dates = np.full(
                len(month_values), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
            )
            valid = right > 0
            max_dates[valid] = dates[right[valid] - 1]
            result["external_max_publish_date"] = max_dates
        rows.append(result)
    features = pd.concat(rows, ignore_index=True).sort_values(["Stkcd", "month"])
    features["eps_growth_revision_3m"] = features.groupby("Stkcd", sort=False)[
        "eps_growth_180d"
    ].diff(3).clip(-2.0, 2.0)
    features["external_leakage_pass"] = (
        features["external_max_publish_date"].isna()
        | features["external_max_publish_date"].le(features["month"])
    )
    if not features["external_leakage_pass"].all():
        raise RuntimeError("External data point-in-time audit failed")
    return features


def attach_external_features(panel: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    augmented = panel.copy()
    augmented["Stkcd"] = augmented["Stkcd"].astype("string").str.zfill(6)
    augmented = augmented.merge(features, on=["month", "Stkcd"], how="left")
    raw = {
        "analyst_attention": np.log1p(augmented["report_count_90d"].fillna(0)),
        "analyst_breadth": np.log1p(augmented["org_count_180d"].fillna(0)),
        "analyst_rating": augmented["rating_mean_180d"],
        "analyst_eps_growth": augmented["eps_growth_180d"],
        "analyst_eps_revision": augmented["eps_growth_revision_3m"],
    }
    for name, values in raw.items():
        within = values.groupby(
            [augmented["month"], augmented["industry"]], dropna=False
        ).rank(pct=True)
        fallback = values.groupby(augmented["month"]).rank(pct=True)
        ranked = within.fillna(fallback) - 0.5
        augmented[f"{name}_missing"] = ranked.isna().astype(np.float32)
        augmented[name] = ranked.fillna(0).astype(np.float32)
    return augmented


def augmented_matrix(frame: pd.DataFrame) -> np.ndarray:
    names = FEATURES + EXTERNAL_FEATURES
    columns = names + [f"{name}_missing" for name in names]
    return frame[columns].to_numpy(np.float32)


def generate_external_signals(panel: pd.DataFrame, research: ResearchConfig,
                              seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    logs: list[dict[str, object]] = []
    ridge = mlp = None
    months = sorted(panel.loc[panel["month"].ge(research.start), "month"].unique())
    for month_value in months:
        month = pd.Timestamp(month_value)
        if ridge is None or month.month in (1, 4, 7, 10):
            lower = month - pd.DateOffset(months=research.train_months)
            history = panel[
                panel["month"].lt(month)
                & panel["month"].ge(lower)
                & panel["eligible"]
                & panel["target"].notna()
                & panel["realization_month"].le(month)
            ]
            if len(history) < 10_000:
                continue
            split = history["month"].max() - pd.DateOffset(
                months=research.validation_months
            )
            train = sampled(
                history[history["month"].le(split)], 120_000,
                seed + month.year * 100 + month.month,
            )
            validation = history[history["month"].gt(split)]
            x, y = augmented_matrix(train), train["target"].to_numpy(np.float32)
            xv, yv = augmented_matrix(validation), validation["target"].to_numpy(np.float32)
            ridge_trial = ridge_fit(x, y)
            trial = MLP(x.shape[1], seed=seed + month.year * 100 + month.month)
            best_epoch, validation_ic = trial.fit(x, y, validation=(xv, yv))
            full = sampled(
                history, 150_000, seed + 1_000_000 + month.year * 100 + month.month
            )
            xf, yf = augmented_matrix(full), full["target"].to_numpy(np.float32)
            ridge = ridge_fit(xf, yf)
            mlp = MLP(xf.shape[1], seed=seed + month.year * 100 + month.month)
            mlp.fit(xf, yf, epochs=best_epoch, validation=None)
            logs.append({
                "refit_month": month,
                "history_start": history["month"].min(),
                "history_end": history["month"].max(),
                "max_label_realization_month": history["realization_month"].max(),
                "max_external_publish_date": history["external_max_publish_date"].max(),
                "future_label_pass": history["realization_month"].max() <= month,
                "future_external_data_pass": (
                    history["external_max_publish_date"].dropna().max() <= month
                    if history["external_max_publish_date"].notna().any() else True
                ),
                "samples": len(full), "epochs": best_epoch,
                "mlp_validation_ic": validation_ic,
            })
        test = panel[panel["month"].eq(month) & panel["eligible"]].copy()
        if test.empty or ridge is None or mlp is None:
            continue
        x_test = augmented_matrix(test)
        test["ridge_score"] = ridge_predict(x_test, ridge)
        test["mlp_score"] = mlp.predict(x_test)
        test["raw_score"] = test.groupby("month")["mlp_score"].rank(pct=True) - 0.5
        rows.append(test[[
            "month", "realization_month", "Stkcd", "industry", "forward_return",
            "volatility_clean", "amount_clean", "ridge_score", "mlp_score", "raw_score",
        ]])
    signals = pd.concat(rows, ignore_index=True)
    audit = pd.DataFrame(logs)
    if audit.empty or not audit[["future_label_pass", "future_external_data_pass"]].all().all():
        raise RuntimeError("External signal generation failed leakage audit")
    return signals, audit


def external_factor_ic(panel: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "development": ("2018-01-01", "2021-12-31"),
        "selection": ("2022-01-01", "2023-12-31"),
        "retrospective_test": ("2024-01-01", "2025-12-31"),
    }
    rows = []
    for period, (start, end) in periods.items():
        sample = panel[
            panel["eligible"] & panel["target"].notna()
            & panel["month"].between(start, end)
        ]
        for feature in EXTERNAL_FEATURES:
            monthly = sample.groupby("month", observed=True).apply(
                lambda group: group[[feature, "target"]].corr(method="spearman").iloc[0, 1]
                if group[[feature, "target"]].dropna().shape[0] >= 30 else np.nan,
                include_groups=False,
            )
            rows.append({
                "period": period, "feature": feature,
                "mean_rank_ic": monthly.mean(),
                "annualized_icir": monthly.mean() / monthly.std() * np.sqrt(12)
                if monthly.std() else np.nan,
                "positive_ic_month_ratio": monthly.gt(0).mean(),
                "months": monthly.count(),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--reports", type=Path, default=RAW_REPORTS)
    args = parser.parse_args()
    external = ExternalConfig()
    research = ResearchConfig()
    reports = download_reports(external, args.reports) if args.refresh or not args.reports.exists() else load_reports(args.reports)
    panel = prepare_clean(OUT / "monthly_panel.csv.gz")
    features = point_in_time_features(panel[["month", "Stkcd"]], reports, external)
    coverage = features["report_count_90d"].gt(0).groupby(features["month"]).mean()
    development_coverage = coverage[
        (coverage.index >= pd.Timestamp(research.start))
        & (coverage.index <= pd.Timestamp(research.selection_end))
    ].mean()
    if development_coverage < external.minimum_report_coverage:
        raise RuntimeError(
            f"External report coverage {development_coverage:.2%} is below the minimum "
            f"{external.minimum_report_coverage:.2%}"
        )
    augmented = attach_external_features(panel, features)
    signals, audit = generate_external_signals(augmented, research, external.random_seed)
    chosen, selection = select_parameters(signals, research)
    chosen_signals = signal_for_model(signals, str(chosen["signal_model"]))
    backtest = build_portfolio(
        chosen_signals, research.top_fraction, float(chosen["exit_fraction"]),
        float(chosen["smoothing_weight"]), research.report_cost_bps,
    )
    metrics = []
    periods = {
        "development": (research.start, research.development_end),
        "selection": (research.selection_start, research.selection_end),
        "retrospective_test": (research.retrospective_start, research.retrospective_end),
        "full": (research.start, research.retrospective_end),
    }
    for period, (start, end) in periods.items():
        metrics.append({"period": period, **period_metric(backtest, start, end, 20)})
    OUT.mkdir(exist_ok=True)
    backtest.to_csv(OUT / "external_backtest.csv", index=False)
    signals.to_csv(OUT / "external_signal_panel.csv.gz", index=False, compression="gzip")
    audit.to_csv(OUT / "external_leakage_audit.csv", index=False)
    selection.to_csv(OUT / "external_parameter_selection.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "external_subperiod.csv", index=False)
    cost_stress(backtest).to_csv(OUT / "external_cost_stress.csv", index=False)
    external_factor_ic(augmented).to_csv(OUT / "external_feature_ic.csv", index=False)
    pd.DataFrame([{
        "source": "Eastmoney research report API",
        "source_start": reports["publish_date"].min(),
        "source_end": reports["publish_date"].max(),
        "reports": len(reports), "covered_stocks": reports["Stkcd"].nunique(),
        "development_selection_mean_monthly_coverage": development_coverage,
        "point_in_time_audit_pass": features["external_leakage_pass"].all(),
    }]).to_csv(OUT / "external_data_quality.csv", index=False)
    lock = {
        "status": "external_candidate_selected_without_retrospective",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrospective_not_used_for_selection": True,
        "selected_parameters": chosen,
        "external_config": asdict(external),
        "research_config": asdict(research),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "external_research_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
