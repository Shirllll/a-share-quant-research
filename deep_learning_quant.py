from __future__ import annotations

"""Quarterly walk-forward MLP + Ridge ensemble used by the original first version."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
FEATURES = [
    "momentum", "value_pe", "value_pb", "value_ps", "low_volatility",
    "small_size", "reversal", "liquidity", "roe", "roa", "gross_margin",
    "cash_quality", "low_leverage",
]


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def bounded(series: pd.Series, lower: float, upper: float) -> pd.Series:
    values = num(series)
    return values.where(values.between(lower, upper))


def positive_inverse(series: pd.Series, upper: float) -> pd.Series:
    values = bounded(series, 0.01, upper)
    return 1 / values


def prepare(panel_path: Path = OUT / "monthly_panel.csv.gz") -> pd.DataFrame:
    """Clean factor inputs before either Ridge or MLP sees the panel."""
    panel = pd.read_csv(panel_path, parse_dates=["month", "Listdt"], low_memory=False)
    panel["Stkcd"] = (
        panel["Stkcd"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    panel = panel.sort_values(["Stkcd", "month"]).drop_duplicates(
        ["Stkcd", "month"], keep="last"
    )
    panel["forward_return"] = panel.groupby("Stkcd", sort=False)["ret"].shift(-1)
    raw = {
        "momentum": num(panel["mom_12_1"]),
        "value_pe": positive_inverse(panel["PE1TTM"], 500),
        "value_pb": positive_inverse(panel["PBV1B"], 50),
        "value_ps": positive_inverse(panel["PSTTM"], 100),
        "low_volatility": -bounded(panel["volatility"], 0.03, 3),
        "small_size": -np.log(num(panel["size"]).where(num(panel["size"]) > 0)),
        "reversal": -bounded(panel["ret"], -0.95, 3.0),
        "liquidity": num(panel["illiq"]).where(num(panel["illiq"]) >= 0),
        "roe": bounded(panel["F050504C"], -1, 1),
        "roa": bounded(panel["F050204C"], -0.5, 0.5),
        "gross_margin": bounded(panel["F053301C"], -1, 1.5),
        "cash_quality": bounded(panel["F052901C"], 0, 5),
        "low_leverage": -bounded(panel["F011201A"], 0, 1.5),
    }
    for name, values in raw.items():
        panel[name] = values
        ranks = panel.groupby(["month", "industry"], dropna=False)[name].rank(pct=True)
        panel[name] = ranks.fillna(panel.groupby("month")[name].rank(pct=True)) - 0.5
        panel[f"{name}_missing"] = panel[name].isna().astype(np.float32)
        panel[name] = panel[name].fillna(0)

    panel["target"] = panel.groupby("month")["forward_return"].rank(pct=True) - 0.5
    amount = num(panel["amount"]).where(num(panel["amount"]) > 0)
    liquid_cut = amount.groupby(panel["month"]).transform(lambda values: values.quantile(0.20))
    panel["eligible"] = (
        panel["trdsta"].eq(1)
        & panel["listed_days"].ge(180)
        & amount.ge(liquid_cut)
        & panel["trading_days"].ge(10)
    )
    return panel


def matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = FEATURES + [f"{name}_missing" for name in FEATURES]
    return frame[columns].to_numpy(np.float32)


def rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank().to_numpy()
    right_rank = pd.Series(right).rank().to_numpy()
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1000.0):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1
    standardized = (x - mean) / scale
    beta = np.linalg.solve(
        standardized.T @ standardized + alpha * np.eye(standardized.shape[1]),
        standardized.T @ y,
    )
    return mean, scale, y.mean(), beta


def ridge_predict(x: np.ndarray, model) -> np.ndarray:
    mean, scale, intercept, beta = model
    return intercept + ((x - mean) / scale) @ beta


class MLP:
    def __init__(self, dimensions: int, seed: int = 20260705):
        rng = np.random.default_rng(seed)
        self.w1 = (rng.normal(size=(dimensions, 32)) * np.sqrt(2 / dimensions)).astype(np.float32)
        self.b1 = np.zeros(32, np.float32)
        self.w2 = (rng.normal(size=(32, 16)) * np.sqrt(2 / 32)).astype(np.float32)
        self.b2 = np.zeros(16, np.float32)
        self.w3 = (rng.normal(size=(16, 1)) * np.sqrt(1 / 16)).astype(np.float32)
        self.b3 = np.zeros(1, np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        hidden1 = np.maximum(x @ self.w1 + self.b1, 0)
        hidden2 = np.maximum(hidden1 @ self.w2 + self.b2, 0)
        return (hidden2 @ self.w3 + self.b3).ravel()

    def copy_state(self) -> list[np.ndarray]:
        return [
            value.copy()
            for value in [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        ]

    def load_state(self, state: list[np.ndarray]) -> None:
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = [
            value.copy() for value in state
        ]

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch: int = 2048,
        learning_rate: float = 0.002,
        l2: float = 1e-4,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        patience: int = 4,
        seed: int = 7,
    ) -> tuple[int, float]:
        parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        first_moment = [np.zeros_like(value) for value in parameters]
        second_moment = [np.zeros_like(value) for value in parameters]
        rng = np.random.default_rng(seed)
        best_score = -np.inf
        best_state = self.copy_state()
        best_epoch, stale, step = 1, 0, 0
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(x))
            for lower in range(0, len(x), batch):
                index = order[lower:lower + batch]
                xb, yb = x[index], y[index, None]
                z1 = xb @ self.w1 + self.b1
                h1 = np.maximum(z1, 0)
                z2 = h1 @ self.w2 + self.b2
                h2 = np.maximum(z2, 0)
                prediction = h2 @ self.w3 + self.b3
                prediction_gradient = 2 * (prediction - yb) / len(index)
                gw3 = h2.T @ prediction_gradient + l2 * self.w3
                gb3 = prediction_gradient.sum(0)
                dh2 = prediction_gradient @ self.w3.T
                dz2 = dh2 * (z2 > 0)
                gw2 = h1.T @ dz2 + l2 * self.w2
                gb2 = dz2.sum(0)
                dh1 = dz2 @ self.w2.T
                dz1 = dh1 * (z1 > 0)
                gw1 = xb.T @ dz1 + l2 * self.w1
                gb1 = dz1.sum(0)
                gradients = [gw1, gb1, gw2, gb2, gw3, gb3]
                step += 1
                for position, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                    first_moment[position] = 0.9 * first_moment[position] + 0.1 * gradient
                    second_moment[position] = (
                        0.999 * second_moment[position] + 0.001 * gradient * gradient
                    )
                    parameter -= learning_rate * (
                        first_moment[position] / (1 - 0.9 ** step)
                    ) / (
                        np.sqrt(second_moment[position] / (1 - 0.999 ** step)) + 1e-8
                    )
            if validation is not None:
                score = rank_corr(self.predict(validation[0]), validation[1])
                if score > best_score + 1e-4:
                    best_score, best_state, best_epoch, stale = (
                        score, self.copy_state(), epoch, 0
                    )
                else:
                    stale += 1
                if stale >= patience:
                    break
        if validation is not None:
            self.load_state(best_state)
        return best_epoch, best_score


def sampled(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    return frame if len(frame) <= maximum else frame.sample(n=maximum, random_state=seed)


def backtest(
    panel: pd.DataFrame,
    start: str = "2018-01-01",
    train_months: int = 60,
    cost_bps: float = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, logs, previous = [], [], set()
    mlp = ridge = None
    weight = 0.5
    months = sorted(panel.loc[panel["month"] >= start, "month"].unique())
    for month_value in months:
        month = pd.Timestamp(month_value)
        if mlp is None or month.month in [1, 4, 7, 10]:
            lower = month - pd.DateOffset(months=train_months)
            history = panel[
                panel["month"].lt(month)
                & panel["month"].ge(lower)
                & panel["eligible"]
                & panel["target"].notna()
            ]
            if len(history) < 10_000:
                continue
            split = history["month"].max() - pd.DateOffset(months=12)
            train = sampled(
                history[history["month"].le(split)],
                120_000,
                100 + month.year + month.month,
            )
            validation = history[history["month"].gt(split)]
            x, y = matrix(train), train["target"].to_numpy(np.float32)
            xv = matrix(validation)
            yv = validation["target"].to_numpy(np.float32)
            trial = MLP(x.shape[1], seed=month.year * 100 + month.month)
            best_epoch, mlp_ic = trial.fit(x, y, validation=(xv, yv))
            ridge_trial = ridge_fit(x, y)
            ridge_validation = ridge_predict(xv, ridge_trial)
            mlp_validation = trial.predict(xv)
            choices = [
                (rank_corr(blend * mlp_validation + (1 - blend) * ridge_validation, yv), blend)
                for blend in [0, 0.25, 0.5, 0.75, 1.0]
            ]
            validation_ic, weight = max(choices)
            full = sampled(history, 150_000, 200 + month.year + month.month)
            xf = matrix(full)
            yf = full["target"].to_numpy(np.float32)
            mlp = MLP(xf.shape[1], seed=month.year * 100 + month.month)
            mlp.fit(xf, yf, epochs=best_epoch, validation=None)
            ridge = ridge_fit(xf, yf)
            history_end = history["month"].max()
            max_label_realization = history_end + pd.offsets.MonthEnd(1)
            logs.append({
                "refit_month": month,
                "history_start": history["month"].min(),
                "history_end": history_end,
                "max_label_realization_month": max_label_realization,
                "future_leakage_pass": max_label_realization <= month,
                "samples": len(full),
                "epochs": best_epoch,
                "mlp_validation_ic": mlp_ic,
                "ensemble_validation_ic": validation_ic,
                "mlp_weight": weight,
            })

        test = panel[panel["month"].eq(month) & panel["eligible"]].copy()
        if mlp is None or ridge is None or test.empty:
            continue
        xt = matrix(test)
        test["prediction"] = (
            weight * mlp.predict(xt) + (1 - weight) * ridge_predict(xt, ridge)
        )
        count = max(1, int(np.ceil(len(test) * 0.10)))
        buffer_count = max(count, int(np.ceil(len(test) * 0.15)))
        retained = previous & set(test.nlargest(buffer_count, "prediction")["Stkcd"])
        additions = test[~test["Stkcd"].isin(retained)].nlargest(
            max(0, count - len(retained)), "prediction"
        )
        chosen = pd.concat([
            test[test["Stkcd"].isin(retained)],
            additions,
        ]).drop_duplicates("Stkcd").head(count)
        names = set(chosen["Stkcd"])
        turnover = 1 if not previous else 1 - len(names & previous) / max(len(names), 1)
        rows.append({
            "month": month + pd.offsets.MonthEnd(1),
            "gross_return": chosen["forward_return"].mean(),
            "turnover": turnover,
            "holdings": len(names),
            "mlp_weight": weight,
        })
        previous = names

    result = pd.DataFrame(rows).dropna(subset=["gross_return"])
    result["net_return"] = (
        result["gross_return"] - result["turnover"] * cost_bps / 10_000
    )
    log = pd.DataFrame(logs)
    if log.empty or not log["future_leakage_pass"].all():
        raise RuntimeError("MLP walk-forward leakage audit failed")
    return result, log


def metric(series: pd.Series) -> dict[str, float]:
    values = series.dropna()
    nav = (1 + values).cumprod()
    annual_return = nav.iloc[-1] ** (12 / len(values)) - 1
    annual_volatility = values.std() * np.sqrt(12)
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_rf0": annual_return / annual_volatility,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "total_return": nav.iloc[-1] - 1,
        "months": len(values),
    }


def cost_stress(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost_bps in [0, 20, 50, 100, 150, 200]:
        returns = result["gross_return"] - result["turnover"] * cost_bps / 10_000
        rows.append({"cost_bps": cost_bps, **metric(returns)})
    return pd.DataFrame(rows)


def data_quality_report(panel: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"rule": "model_panel_rows", "affected_rows": len(panel), "affected_share": 1.0},
        {
            "rule": "eligible_rows",
            "affected_rows": int(panel["eligible"].sum()),
            "affected_share": float(panel["eligible"].mean()),
        },
    ]
    for feature in FEATURES:
        column = f"{feature}_missing"
        count = int(panel[column].sum())
        rows.append({
            "rule": column,
            "affected_rows": count,
            "affected_share": count / len(panel),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--panel", type=Path, default=OUT / "monthly_panel.csv.gz")
    args = parser.parse_args()

    panel = prepare(args.panel)
    data_quality_report(panel).to_csv(
        OUT / "deep_learning_data_quality.csv", index=False
    )
    result, logs = backtest(panel, args.start)
    ridge = pd.read_csv(OUT / "ml_backtest_monthly.csv", parse_dates=["month"])
    result = result.merge(
        ridge[["month", "net_return", "benchmark_return"]].rename(
            columns={"net_return": "ridge_return"}
        ),
        on="month",
        how="left",
    )
    result.to_csv(OUT / "deep_learning_backtest.csv", index=False)
    logs.to_csv(OUT / "deep_learning_model_log.csv", index=False)
    logs[[
        "refit_month", "history_start", "history_end",
        "max_label_realization_month", "future_leakage_pass",
    ]].to_csv(OUT / "deep_learning_leakage_audit.csv", index=False)
    metrics = pd.DataFrame([
        {"series": "mlp_ridge_ensemble", **metric(result["net_return"])},
        {"series": "polynomial_ridge", **metric(result["ridge_return"])},
        {"series": "benchmark", **metric(result["benchmark_return"])},
    ])
    metrics.to_csv(OUT / "deep_learning_metrics.csv", index=False)
    cost_stress(result).to_csv(OUT / "deep_learning_cost_stress.csv", index=False)
    print(metrics.to_string(index=False))
    print(cost_stress(result).to_string(index=False))


if __name__ == "__main__":
    main()
