from __future__ import annotations

import argparse
import csv
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from data_cleaning import attach_security_states


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"


def csv_members(z: zipfile.ZipFile, prefix: str) -> list[str]:
    return [n for n in z.namelist() if Path(n).name.startswith(prefix) and n.lower().endswith(".csv")]


def read_zip_parts(path: Path, prefix: str, usecols: list[str], start: str, end: str,
                   date_col: str, chunksize: int = 300_000):
    with zipfile.ZipFile(path) as z:
        for member in csv_members(z, prefix):
            with z.open(member) as fh:
                for chunk in pd.read_csv(fh, usecols=usecols, dtype={usecols[0]: "string"},
                                         chunksize=chunksize, low_memory=False):
                    d = pd.to_datetime(chunk[date_col], errors="coerce")
                    mask = d.between(start, end)
                    if mask.any():
                        chunk = chunk.loc[mask].copy()
                        chunk[date_col] = d.loc[mask]
                        yield chunk


def monthly_prices(start: str, end: str) -> pd.DataFrame:
    cols = ["Stkcd", "Trddt", "Dretwd", "Dnvaltrd", "Dsmvosd", "Markettype", "Trdsta"]
    pieces = []
    files = sorted(DATA.glob("02_daily_return_*.zip"))
    for path in files:
        for x in read_zip_parts(path, "TRD_Dalyr", cols, start, end, "Trddt"):
            x = x[x["Markettype"].isin([1, 4, 16, 32, 64])]
            x["month"] = x["Trddt"].dt.to_period("M").dt.to_timestamp("M")
            x["Dretwd"] = pd.to_numeric(x["Dretwd"], errors="coerce")
            x["Dnvaltrd"] = pd.to_numeric(x["Dnvaltrd"], errors="coerce")
            x["Dsmvosd"] = pd.to_numeric(x["Dsmvosd"], errors="coerce")
            x["ret_sq"] = x["Dretwd"] ** 2
            x["illiq_day"] = x["Dretwd"].abs() / x["Dnvaltrd"].replace(0, np.nan)
            pieces.append(x.groupby(["Stkcd", "month"], observed=True).agg(
                ret=("Dretwd", lambda s: (1 + s.dropna()).prod() - 1),
                amount=("Dnvaltrd", "mean"), size=("Dsmvosd", "last"),
                ret_sum=("Dretwd", "sum"), ret_sq_sum=("ret_sq", "sum"),
                illiq=("illiq_day", "mean"), max_ret=("Dretwd", "max"),
                trdsta=("Trdsta", "last"), trading_days=("Dretwd", "count")).reset_index())
    if not pieces:
        raise RuntimeError("指定日期范围内没有日行情数据")
    # 同一股票月份可能横跨压缩包分片；复合收益并合并统计量。
    p = pd.concat(pieces, ignore_index=True)
    p = p.groupby(["Stkcd", "month"], observed=True).agg(
        ret=("ret", lambda s: (1 + s).prod() - 1), amount=("amount", "mean"),
        size=("size", "last"), trdsta=("trdsta", "last"),
        ret_sum=("ret_sum", "sum"), ret_sq_sum=("ret_sq_sum", "sum"),
        illiq=("illiq", "mean"), max_ret=("max_ret", "max"),
        trading_days=("trading_days", "sum")).reset_index()
    variance = (p["ret_sq_sum"] - p["ret_sum"] ** 2 / p["trading_days"].clip(lower=1)) / (p["trading_days"] - 1).clip(lower=1)
    p["volatility"] = np.sqrt(variance.clip(lower=0)) * np.sqrt(252)
    return p.drop(columns=["ret_sum", "ret_sq_sum"])


def monthly_valuation(start: str, end: str) -> pd.DataFrame:
    cols = ["Symbol", "TradingDate", "PE1TTM", "PBV1B", "PSTTM"]
    pieces = []
    for path in sorted(DATA.glob("03_valuation_*.zip")):
        for x in read_zip_parts(path, "STK_MKT_ValuationMetrics", cols, start, end, "TradingDate"):
            x["month"] = x["TradingDate"].dt.to_period("M").dt.to_timestamp("M")
            x = x.sort_values("TradingDate").groupby(["Symbol", "month"], observed=True).tail(1)
            pieces.append(x)
    v = pd.concat(pieces, ignore_index=True)
    v = v.sort_values("TradingDate").groupby(["Symbol", "month"], observed=True).tail(1)
    return v.rename(columns={"Symbol": "Stkcd"})[["Stkcd", "month", "PE1TTM", "PBV1B", "PSTTM"]]


def read_single_csv(zip_path: Path, prefix: str, usecols: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        member = csv_members(z, prefix)[0]
        with z.open(member) as fh:
            return pd.read_csv(fh, usecols=usecols, dtype={usecols[0]: "string"}, low_memory=False)


def quality_by_announcement(start: str, end: str) -> pd.DataFrame:
    qcols = ["Stkcd", "Accper", "Typrep", "Source", "F050204C", "F050504C", "F053301C", "F052901C"]
    q = read_single_csv(DATA / "11_profitability_indicators.csv.zip", "FI_T5", qcols)
    lev = read_single_csv(DATA / "10_solvency_indicators.csv.zip", "FI_T1",
                          ["Stkcd", "Accper", "Typrep", "Source", "F011201A"])
    a = read_single_csv(DATA / "12_report_publication_date.csv.zip", "IAR_Rept", ["Stkcd", "Accper", "Annodt"])
    q = q[(q["Typrep"] == "A") & (q["Source"] == 0)].copy()
    lev = lev[(lev["Typrep"] == "A") & (lev["Source"] == 0)].copy()
    q["Accper"] = pd.to_datetime(q["Accper"], errors="coerce")
    lev["Accper"] = pd.to_datetime(lev["Accper"], errors="coerce")
    lev = lev.sort_values("Accper").drop_duplicates(["Stkcd", "Accper"], keep="last")
    q = q.merge(lev[["Stkcd", "Accper", "F011201A"]], on=["Stkcd", "Accper"], how="left")
    a["Accper"] = pd.to_datetime(a["Accper"], errors="coerce")
    a["Annodt"] = pd.to_datetime(a["Annodt"], errors="coerce")
    a = a.dropna(subset=["Annodt"]).groupby(["Stkcd", "Accper"], as_index=False)["Annodt"].min()
    q = q.merge(a, on=["Stkcd", "Accper"], how="inner").dropna(subset=["Annodt"])
    q = q[q["Annodt"].between(start, end)].sort_values(["Stkcd", "Annodt", "Accper"])
    return q.drop_duplicates(["Stkcd", "Annodt"], keep="last")


def company_info() -> pd.DataFrame:
    c = read_single_csv(DATA / "01_company_file.csv.zip", "TRD_Co", ["Stkcd", "Listdt", "Markettype"])
    c["Listdt"] = pd.to_datetime(c["Listdt"], errors="coerce")
    return c


def industry_history() -> pd.DataFrame:
    i = read_single_csv(DATA / "08_industry_classification_history.csv.zip", "STK_INDUSTRYCLASS",
                        ["Symbol", "IndustryClassificationID", "ImplementDate", "IndustryCode"])
    # 证监会2012版覆盖历史较长，适合全样本做粗行业中性化。
    i = i[i["IndustryClassificationID"] == "P0207"].copy()
    i["ImplementDate"] = pd.to_datetime(i["ImplementDate"], errors="coerce")
    i = i.dropna(subset=["ImplementDate"]).rename(columns={"Symbol": "Stkcd", "IndustryCode": "industry"})
    return i.sort_values(["ImplementDate", "Stkcd"])


def rank_z(s: pd.Series, higher_better: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo, hi = x.quantile([0.01, 0.99])
    x = x.clip(lo, hi)
    z = (x - x.mean()) / x.std(ddof=0)
    return z if higher_better else -z


def positive_inverse(s: pd.Series, upper: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return 1 / x.where(x.between(0.01, upper))


def build_panel(start: str, end: str, use_quality: bool) -> pd.DataFrame:
    p = monthly_prices(start, end).merge(monthly_valuation(start, end), on=["Stkcd", "month"], how="left")
    p = p.merge(company_info()[["Stkcd", "Listdt"]], on="Stkcd", how="left")
    p = p.sort_values(["Stkcd", "month"])
    p["mom_12_1"] = p.groupby("Stkcd")["ret"].transform(
        lambda s: (1 + s.shift(1)).rolling(11, min_periods=8).apply(np.prod, raw=True) - 1)
    if use_quality:
        q = quality_by_announcement(start, end)
        left = p[["Stkcd", "month"]].copy().sort_values(["month", "Stkcd"])
        right = q.sort_values(["Annodt", "Stkcd"])
        p = pd.merge_asof(left, right, left_on="month", right_on="Annodt", by="Stkcd", direction="backward").merge(
            p, on=["Stkcd", "month"], how="right")
    left = p[["Stkcd", "month"]].sort_values(["month", "Stkcd"])
    ind = pd.merge_asof(left, industry_history(), left_on="month", right_on="ImplementDate",
                        by="Stkcd", direction="backward")[["Stkcd", "month", "industry"]]
    p = p.merge(ind, on=["Stkcd", "month"], how="left")
    p["listed_days"] = (p["month"] - p["Listdt"]).dt.days
    return attach_security_states(p)


def neutralize(g: pd.DataFrame, col: str) -> pd.Series:
    """行业内标准化；行业缺失时退回全市场标准化。"""
    out = g.groupby("industry", dropna=False)[col].transform(lambda s: rank_z(s))
    return out.fillna(rank_z(g[col]))


def run_backtest(p: pd.DataFrame, cost_bps: float, top_frac: float, buffer_frac: float,
                 use_quality: bool, target_vol: float) -> pd.DataFrame:
    quality_cols = ["F050204C", "F050504C", "F053301C", "F052901C"]
    p = p.sort_values(["Stkcd", "month"]).copy()
    p["forward_return"] = p.groupby("Stkcd")["ret"].shift(-1)
    rows, prev = [], set()
    for month, g in p.groupby("month", sort=True):
        g = g.copy()
        liquid_cut = g["amount"].quantile(.20)
        g = g[(g["trdsta"] == 1) & (g["listed_days"] >= 180) & (g["amount"] >= liquid_cut)
              & (g["trading_days"] >= 15) & (g["special_state"] == "A") & (g["listing_state"] == "A")]
        value_parts = [positive_inverse(g["PE1TTM"], 500), positive_inverse(g["PBV1B"], 50),
                       positive_inverse(g["PSTTM"], 100)]
        g["value"] = pd.concat([rank_z(x) for x in value_parts], axis=1).mean(axis=1)
        g["momentum"] = rank_z(g["mom_12_1"])
        g["low_volatility"] = rank_z(g["volatility"], higher_better=False)
        g["small_size"] = rank_z(np.log(g["size"].where(g["size"] > 0)), higher_better=False)
        g["reversal"] = rank_z(g["ret"], higher_better=False)
        g["illiquidity_premium"] = rank_z(g["illiq"])
        factors = ["value", "momentum", "low_volatility", "small_size", "reversal", "illiquidity_premium"]
        if use_quality:
            cleaned_quality = [pd.to_numeric(g[c], errors="coerce").where(pd.to_numeric(g[c], errors="coerce").between(-1, 1.5)) for c in quality_cols]
            quality_parts = [rank_z(x) for x in cleaned_quality]
            leverage = pd.to_numeric(g["F011201A"], errors="coerce").where(pd.to_numeric(g["F011201A"], errors="coerce").between(0, 1.5))
            quality_parts.append(rank_z(leverage, higher_better=False))
            g["quality"] = pd.concat(quality_parts, axis=1).mean(axis=1)
            factors.append("quality")
        for f in factors:
            g[f] = neutralize(g, f)
        g["score"] = g[factors].mean(axis=1)
        eligible = g.dropna(subset=["score"])
        n = max(1, math.ceil(len(eligible) * top_frac))
        buffer_n = max(n, math.ceil(len(eligible) * buffer_frac))
        buffer_names = set(eligible.nlargest(buffer_n, "score")["Stkcd"])
        retained = prev & buffer_names
        additions = eligible[~eligible["Stkcd"].isin(retained)].nlargest(max(0, n - len(retained)), "score")
        chosen = pd.concat([eligible[eligible["Stkcd"].isin(retained)], additions]).drop_duplicates("Stkcd").head(n)
        names = set(chosen["Stkcd"])
        turnover = 1.0 if not prev else 1 - len(names & prev) / max(len(names), 1)
        realized_month = month + pd.offsets.MonthEnd(1)
        rows.append({"month": realized_month, "gross_return": chosen["forward_return"].mean(),
                     "turnover": turnover, "holdings": len(names)})
        prev = names
    r = pd.DataFrame(rows).sort_values("month")
    # 当月月末选中的股票赚取各自下一月收益。
    r["net_return"] = r["gross_return"] - r["turnover"].fillna(0) * cost_bps / 10_000
    r = r.dropna(subset=["gross_return"])
    trailing_vol = r["net_return"].rolling(12, min_periods=6).std().shift(1) * np.sqrt(12)
    r["exposure"] = (target_vol / trailing_vol).clip(lower=0, upper=1).fillna(1)
    r["risk_managed_return"] = r["exposure"] * r["net_return"]
    return r


def benchmark(start: str, end: str) -> pd.DataFrame:
    parts = []
    for path in sorted(DATA.glob("09_index_daily_*.zip")):
        for x in read_zip_parts(path, "IDX_Idxtrd", ["Indexcd", "Idxtrd01", "Idxtrd08"], start, end, "Idxtrd01"):
            x = x[x["Indexcd"] == "000300"]
            x["month"] = x["Idxtrd01"].dt.to_period("M").dt.to_timestamp("M")
            x["r"] = pd.to_numeric(x["Idxtrd08"], errors="coerce") / 100
            parts.append(x.groupby("month")["r"].apply(lambda s: (1 + s.dropna()).prod() - 1).reset_index())
    b = pd.concat(parts).groupby("month")["r"].apply(lambda s: (1 + s).prod() - 1).rename("benchmark_return").reset_index()
    return b


def metrics(r: pd.DataFrame) -> pd.DataFrame:
    out = []
    for col in ["net_return", "risk_managed_return", "benchmark_return"]:
        s = r[col].dropna()
        nav = (1 + s).cumprod()
        years = len(s) / 12
        ann = nav.iloc[-1] ** (1 / years) - 1
        vol = s.std(ddof=1) * np.sqrt(12)
        dd = nav / nav.cummax() - 1
        out.append({"series": col, "annual_return": ann, "annual_volatility": vol,
                    "sharpe_rf0": ann / vol if vol else np.nan, "max_drawdown": dd.min(),
                    "total_return": nav.iloc[-1] - 1, "months": len(s)})
    return pd.DataFrame(out)


def factor_ic(p: pd.DataFrame, use_quality: bool) -> pd.DataFrame:
    """月度 Spearman IC：本月末因子与下月收益的截面相关。"""
    p = p.sort_values(["Stkcd", "month"]).copy()
    p["forward_return"] = p.groupby("Stkcd")["ret"].shift(-1)
    raw = {"momentum": p["mom_12_1"], "value_pb": 1 / pd.to_numeric(p["PBV1B"], errors="coerce"),
           "low_volatility": -p["volatility"], "small_size": -np.log(p["size"].where(p["size"] > 0)),
           "reversal": -p["ret"], "illiquidity_premium": p["illiq"]}
    if use_quality:
        raw.update({"roe": p["F050504C"], "roa": p["F050204C"], "low_leverage": -p["F011201A"]})
    rows = []
    for name, values in raw.items():
        tmp = pd.DataFrame({"month": p["month"], "x": pd.to_numeric(values, errors="coerce"), "y": p["forward_return"]})
        ics = tmp.groupby("month").apply(lambda g: g[["x", "y"]].corr(method="spearman").iloc[0, 1] if g[["x", "y"]].dropna().shape[0] >= 30 else np.nan)
        rows.append({"factor": name, "mean_ic": ics.mean(), "ic_std": ics.std(),
                     "ic_ir": ics.mean() / ics.std() if ics.std() else np.nan, "months": ics.count()})
    return pd.DataFrame(rows).sort_values("mean_ic", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--cost-bps", type=float, default=20)
    ap.add_argument("--top-frac", type=float, default=.10)
    ap.add_argument("--buffer-frac", type=float, default=.15)
    ap.add_argument("--target-vol", type=float, default=.15)
    ap.add_argument("--no-quality", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    p = build_panel(args.start, args.end, not args.no_quality)
    p.to_csv(OUT / "monthly_panel.csv.gz", index=False, compression="gzip", quoting=csv.QUOTE_MINIMAL)
    r = run_backtest(p, args.cost_bps, args.top_frac, args.buffer_frac, not args.no_quality,
                     args.target_vol).merge(benchmark(args.start, args.end), on="month", how="left")
    r["strategy_nav"] = (1 + r["net_return"]).cumprod()
    r["risk_managed_nav"] = (1 + r["risk_managed_return"]).cumprod()
    r["benchmark_nav"] = (1 + r["benchmark_return"].fillna(0)).cumprod()
    r.to_csv(OUT / "backtest_monthly.csv", index=False)
    m = metrics(r)
    m.to_csv(OUT / "metrics.csv", index=False)
    factor_ic(p, not args.no_quality).to_csv(OUT / "factor_ic.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = r.plot(x="month", y=["strategy_nav", "risk_managed_nav", "benchmark_nav"], figsize=(10, 5), grid=True)
        ax.set_ylabel("NAV"); ax.figure.tight_layout(); ax.figure.savefig(OUT / "nav.png", dpi=160); plt.close(ax.figure)
    except (ImportError, RuntimeError):
        pass
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
