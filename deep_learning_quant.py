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


def bounded(s, lower, upper):
    x = num(s)
    return x.where(x.between(lower, upper))


def positive_inverse(s, upper):
    x = bounded(s, 0.01, upper)
    return 1 / x


def prepare():
    p = pd.read_csv(OUT / "monthly_panel.csv.gz", parse_dates=["month", "Listdt"], low_memory=False)
    p = p.sort_values(["Stkcd", "month"])
    p["forward_return"] = p.groupby("Stkcd")["ret"].shift(-1)
    raw = {"momentum": num(p["mom_12_1"]), "value_pe": positive_inverse(p["PE1TTM"], 500),
           "value_pb": positive_inverse(p["PBV1B"], 50), "value_ps": positive_inverse(p["PSTTM"], 100),
           "low_volatility": -bounded(p["volatility"], .03, 3),
           "small_size": -np.log(num(p["size"]).where(num(p["size"]) > 0)),
           "reversal": -num(p["ret"]), "liquidity": num(p["illiq"]),
           "roe": bounded(p["F050504C"], -1, 1), "roa": bounded(p["F050204C"], -.5, .5),
           "gross_margin": bounded(p["F053301C"], -1, 1.5),
           "cash_quality": bounded(p["F052901C"], 0, 5),
           "low_leverage": -bounded(p["F011201A"], 0, 1.5)}
    for name, values in raw.items():
        p[name] = values
        ranks = p.groupby(["month", "industry"], dropna=False)[name].rank(pct=True)
        p[name] = ranks.fillna(p.groupby("month")[name].rank(pct=True)) - .5
        p[name + "_missing"] = p[name].isna().astype(float)
        p[name] = p[name].fillna(0)
    p["target"] = p.groupby("month")["forward_return"].rank(pct=True) - .5
    cut = p.groupby("month")["amount"].transform(lambda s: s.quantile(.20))
    p["eligible"] = ((p["trdsta"] == 1) & (p["listed_days"] >= 180) &
                     (p["amount"] >= cut) & (p["trading_days"] >= 10))
    return p


def matrix(df):
    cols = FEATURES + [x + "_missing" for x in FEATURES]
    return df[cols].to_numpy(np.float32)


def rank_corr(a, b):
    ar = pd.Series(a).rank().to_numpy(); br = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])


def ridge_fit(x, y, alpha=1000.0):
    mean, scale = x.mean(0), x.std(0); scale[scale < 1e-6] = 1
    z = (x-mean)/scale
    beta = np.linalg.solve(z.T@z + alpha*np.eye(z.shape[1]), z.T@y)
    return mean, scale, y.mean(), beta


def ridge_predict(x, m):
    mean, scale, intercept, beta = m
    return intercept + ((x-mean)/scale)@beta


class MLP:
    def __init__(self, d, seed=20260705):
        rng = np.random.default_rng(seed)
        self.w1 = (rng.normal(size=(d, 32))*np.sqrt(2/d)).astype(np.float32); self.b1 = np.zeros(32, np.float32)
        self.w2 = (rng.normal(size=(32, 16))*np.sqrt(2/32)).astype(np.float32); self.b2 = np.zeros(16, np.float32)
        self.w3 = (rng.normal(size=(16, 1))*np.sqrt(1/16)).astype(np.float32); self.b3 = np.zeros(1, np.float32)

    def predict(self, x):
        h1 = np.maximum(x@self.w1+self.b1, 0); h2 = np.maximum(h1@self.w2+self.b2, 0)
        return (h2@self.w3+self.b3).ravel()

    def copy_state(self):
        return [a.copy() for a in [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]]

    def load_state(self, s):
        self.w1,self.b1,self.w2,self.b2,self.w3,self.b3 = [a.copy() for a in s]

    def fit(self, x, y, epochs=20, batch=2048, lr=.002, l2=1e-4, val=None, patience=4, seed=7):
        pars = [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]
        m = [np.zeros_like(a) for a in pars]; v = [np.zeros_like(a) for a in pars]
        rng = np.random.default_rng(seed); best, best_state, best_epoch, stale, step = -np.inf, self.copy_state(), 1, 0, 0
        for epoch in range(1, epochs+1):
            order = rng.permutation(len(x))
            for lo in range(0, len(x), batch):
                ix = order[lo:lo+batch]; xb=x[ix]; yb=y[ix,None]
                z1=xb@self.w1+self.b1; h1=np.maximum(z1,0); z2=h1@self.w2+self.b2; h2=np.maximum(z2,0); pred=h2@self.w3+self.b3
                dp=2*(pred-yb)/len(ix); gw3=h2.T@dp+l2*self.w3; gb3=dp.sum(0)
                dh2=dp@self.w3.T; dz2=dh2*(z2>0); gw2=h1.T@dz2+l2*self.w2; gb2=dz2.sum(0)
                dh1=dz2@self.w2.T; dz1=dh1*(z1>0); gw1=xb.T@dz1+l2*self.w1; gb1=dz1.sum(0)
                grads=[gw1,gb1,gw2,gb2,gw3,gb3]; step+=1
                for j,(par,g) in enumerate(zip(pars,grads)):
                    m[j]=.9*m[j]+.1*g; v[j]=.999*v[j]+.001*g*g
                    par -= lr*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
            if val is not None:
                score=rank_corr(self.predict(val[0]),val[1])
                if score>best+1e-4: best,best_state,best_epoch,stale=score,self.copy_state(),epoch,0
                else: stale+=1
                if stale>=patience: break
        if val is not None: self.load_state(best_state)
        return best_epoch, best


def sampled(df, n, seed):
    return df if len(df)<=n else df.sample(n=n, random_state=seed)


def backtest(p, start="2018-01-01", train_months=60, cost_bps=20):
    rows=[]; logs=[]; prev=set(); mlp=ridge=None; weight=.5
    for month in sorted(p.loc[p["month"]>=start,"month"].unique()):
        month=pd.Timestamp(month)
        if mlp is None or month.month in [1,4,7,10]:
            lo=month-pd.DateOffset(months=train_months)
            hist=p[(p["month"]<month)&(p["month"]>=lo)&p["eligible"]&p["target"].notna()]
            split=hist["month"].max()-pd.DateOffset(months=12)
            tr=sampled(hist[hist["month"]<=split],120000,100+month.year+month.month)
            va=hist[hist["month"]>split]
            x,y=matrix(tr),tr["target"].to_numpy(np.float32); xv,yv=matrix(va),va["target"].to_numpy(np.float32)
            trial=MLP(x.shape[1],seed=month.year*100+month.month); best_epoch,mlp_ic=trial.fit(x,y,val=(xv,yv))
            rtrial=ridge_fit(x,y); rp=ridge_predict(xv,rtrial); mp=trial.predict(xv)
            choices=[]
            for w in [0,.25,.5,.75,1.]: choices.append((rank_corr(w*mp+(1-w)*rp,yv),w))
            val_ic,weight=max(choices)
            full=sampled(hist,150000,200+month.year+month.month); xf,yf=matrix(full),full["target"].to_numpy(np.float32)
            mlp=MLP(xf.shape[1],seed=month.year*100+month.month); mlp.fit(xf,yf,epochs=best_epoch,val=None)
            ridge=ridge_fit(xf,yf)
            logs.append({"refit_month":month,"samples":len(full),"epochs":best_epoch,"mlp_val_ic":mlp_ic,
                         "ensemble_val_ic":val_ic,"mlp_weight":weight})
        test=p[(p["month"]==month)&p["eligible"]].copy(); xt=matrix(test)
        test["prediction"]=weight*mlp.predict(xt)+(1-weight)*ridge_predict(xt,ridge)
        n=max(1,int(np.ceil(len(test)*.10))); bn=max(n,int(np.ceil(len(test)*.15)))
        retained=prev&set(test.nlargest(bn,"prediction")["Stkcd"])
        add=test[~test["Stkcd"].isin(retained)].nlargest(max(0,n-len(retained)),"prediction")
        chosen=pd.concat([test[test["Stkcd"].isin(retained)],add]).drop_duplicates("Stkcd").head(n)
        names=set(chosen["Stkcd"]); turnover=1 if not prev else 1-len(names&prev)/max(len(names),1)
        rows.append({"month":month+pd.offsets.MonthEnd(1),"gross_return":chosen["forward_return"].mean(),
                     "turnover":turnover,"holdings":len(names),"mlp_weight":weight})
        prev=names
    r=pd.DataFrame(rows).dropna(subset=["gross_return"]); r["net_return"]=r["gross_return"]-r["turnover"]*cost_bps/10000
    return r,pd.DataFrame(logs)


def metric(s):
    s=s.dropna(); nav=(1+s).cumprod(); ann=nav.iloc[-1]**(12/len(s))-1; vol=s.std()*np.sqrt(12)
    return {"annual_return":ann,"annual_volatility":vol,"sharpe_rf0":ann/vol,
            "max_drawdown":(nav/nav.cummax()-1).min(),"total_return":nav.iloc[-1]-1,"months":len(s)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",default="2018-01-01"); a=ap.parse_args()
    p=prepare(); r,logs=backtest(p,a.start)
    old=pd.read_csv(OUT/"ml_backtest_monthly.csv",parse_dates=["month"])
    r=r.merge(old[["month","net_return","benchmark_return"]].rename(columns={"net_return":"ridge_return"}),on="month",how="left")
    r.to_csv(OUT/"deep_learning_backtest.csv",index=False); logs.to_csv(OUT/"deep_learning_model_log.csv",index=False)
    m=pd.DataFrame([{"series":"mlp_ridge_ensemble",**metric(r["net_return"])},
                    {"series":"polynomial_ridge",**metric(r["ridge_return"])},
                    {"series":"benchmark",**metric(r["benchmark_return"])}])
    m.to_csv(OUT/"deep_learning_metrics.csv",index=False); print(m.to_string(index=False))

if __name__=="__main__": main()
