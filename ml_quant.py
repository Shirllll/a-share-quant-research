from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
FEATURES = ["momentum", "value_pe", "value_pb", "value_ps", "low_volatility",
            "small_size", "reversal", "liquidity", "roe", "roa", "gross_margin",
            "cash_quality", "low_leverage"]


def num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def prepare(path):
    p = pd.read_csv(path, parse_dates=["month", "Listdt"], low_memory=False).sort_values(["Stkcd", "month"])
    p["forward_return"] = p.groupby("Stkcd")["ret"].shift(-1)
    raw = {"momentum": num(p["mom_12_1"]), "value_pe": 1/num(p["PE1TTM"]),
           "value_pb": 1/num(p["PBV1B"]), "value_ps": 1/num(p["PSTTM"]),
           "low_volatility": -num(p["volatility"]),
           "small_size": -np.log(num(p["size"]).where(num(p["size"]) > 0)),
           "reversal": -num(p["ret"]), "liquidity": -num(p["illiq"]),
           "roe": num(p["F050504C"]), "roa": num(p["F050204C"]),
           "gross_margin": num(p["F053301C"]), "cash_quality": num(p["F052901C"]),
           "low_leverage": -num(p["F011201A"])}
    for name, values in raw.items():
        p[name] = values
        r = p.groupby(["month", "industry"], dropna=False)[name].rank(pct=True)
        p[name] = r.fillna(p.groupby("month")[name].rank(pct=True)) - .5
    p["target"] = p.groupby("month")["forward_return"].rank(pct=True) - .5
    cut = p.groupby("month")["amount"].transform(lambda s: s.quantile(.20))
    p["eligible"] = ((p["trdsta"] == 1) & (p["listed_days"] >= 180) &
                     (p["amount"] >= cut) & (p["trading_days"] >= 10))
    return p


def design(df, medians=None):
    x = df[FEATURES].to_numpy(float)
    if medians is None:
        medians = np.nan_to_num(np.nanmedian(x, axis=0), nan=0.0)
    x = np.where(np.isnan(x), medians, x)
    pairs = [(0,1), (0,2), (1,4), (2,9), (4,5), (5,6), (8,12), (9,12)]
    return np.column_stack([x, x*x] + [(x[:,i]*x[:,j])[:,None] for i,j in pairs]), medians


def fit(x, y, alpha):
    mean, scale = x.mean(0), x.std(0); scale[scale < 1e-8] = 1
    z = (x-mean)/scale
    beta = np.linalg.solve(z.T@z + alpha*np.eye(z.shape[1]), z.T@y)
    return np.r_[y.mean(), beta], np.vstack([mean, scale])


def predict(x, model):
    b, stats = model
    return b[0] + ((x-stats[0])/stats[1])@b[1:]


def choose_alpha(train, x, y):
    split = train["month"].max() - pd.DateOffset(months=12)
    val = (train["month"] > split).to_numpy()
    if val.sum() < 1000 or (~val).sum() < 1000: return 100.0
    best = (100.0, -np.inf)
    for a in [1., 10., 100., 1000., 10000.]:
        pred = predict(x[val], fit(x[~val], y[~val], a))
        pr = pd.Series(pred).rank().to_numpy(); yr = pd.Series(y[val]).rank().to_numpy()
        corr = np.corrcoef(pr, yr)[0, 1]
        if pd.notna(corr) and corr > best[1]: best = (a, corr)
    return best[0]


def walk_forward(p, start, train_months, top_frac, buffer_frac, cost_bps):
    rows, logs, prev, model, medians = [], [], set(), None, None
    for month in sorted(p.loc[p["month"] >= start, "month"].unique()):
        month = pd.Timestamp(month)
        if model is None or month.month in [1,4,7,10]:
            lo = month - pd.DateOffset(months=train_months)
            tr = p[(p["month"] < month) & (p["month"] >= lo) & p["eligible"] & p["target"].notna()]
            if len(tr) < 5000: continue
            x, medians = design(tr); y = tr["target"].to_numpy(float)
            alpha = choose_alpha(tr, x, y); model = fit(x, y, alpha)
            logs.append({"refit_month":month, "samples":len(tr), "alpha":alpha})
        test = p[(p["month"] == month) & p["eligible"]].copy()
        if model is None or test.empty: continue
        xt, _ = design(test, medians); test["prediction"] = predict(xt, model)
        n = max(1, int(np.ceil(len(test)*top_frac))); bn = max(n, int(np.ceil(len(test)*buffer_frac)))
        retained = prev & set(test.nlargest(bn, "prediction")["Stkcd"])
        add = test[~test["Stkcd"].isin(retained)].nlargest(max(0,n-len(retained)), "prediction")
        chosen = pd.concat([test[test["Stkcd"].isin(retained)], add]).drop_duplicates("Stkcd").head(n)
        names = set(chosen["Stkcd"]); turnover = 1 if not prev else 1-len(names&prev)/max(len(names),1)
        rows.append({"month":month+pd.offsets.MonthEnd(1), "gross_return":chosen["forward_return"].mean(),
                     "turnover":turnover, "holdings":len(names), "mean_prediction":chosen["prediction"].mean()})
        prev = names
    r = pd.DataFrame(rows).dropna(subset=["gross_return"])
    r["net_return"] = r["gross_return"] - r["turnover"]*cost_bps/10000
    r["ml_nav"] = (1+r["net_return"]).cumprod()
    return r, pd.DataFrame(logs)


def metric(s):
    s=s.dropna(); nav=(1+s).cumprod(); ann=nav.iloc[-1]**(12/len(s))-1; vol=s.std()*np.sqrt(12)
    return {"annual_return":ann, "annual_volatility":vol, "sharpe_rf0":ann/vol,
            "max_drawdown":(nav/nav.cummax()-1).min(), "total_return":nav.iloc[-1]-1, "months":len(s)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",default="2018-01-01")
    ap.add_argument("--train-months",type=int,default=60); ap.add_argument("--top-frac",type=float,default=.10)
    ap.add_argument("--buffer-frac",type=float,default=.15); ap.add_argument("--cost-bps",type=float,default=20)
    a=ap.parse_args(); p=prepare(OUT/"monthly_panel.csv.gz")
    r,logs=walk_forward(p,a.start,a.train_months,a.top_frac,a.buffer_frac,a.cost_bps)
    base=pd.read_csv(OUT/"backtest_monthly.csv",parse_dates=["month"])
    r=r.merge(base[["month","net_return","benchmark_return"]].rename(columns={"net_return":"linear_return"}),on="month",how="left")
    r["linear_nav"]=(1+r["linear_return"]).cumprod(); r["benchmark_nav"]=(1+r["benchmark_return"]).cumprod()
    r.to_csv(OUT/"ml_backtest_monthly.csv",index=False); logs.to_csv(OUT/"ml_model_log.csv",index=False)
    m=pd.DataFrame([{"series":"ml_nonlinear_ridge",**metric(r["net_return"])},
                    {"series":"linear_multifactor",**metric(r["linear_return"])},
                    {"series":"benchmark",**metric(r["benchmark_return"])}])
    m.to_csv(OUT/"ml_metrics.csv",index=False); print(m.to_string(index=False))

if __name__ == "__main__": main()
