from __future__ import annotations

"""Walk-forward A-share ranking with recent tabular-learning ideas.

The implementation deliberately keeps all model selection inside the historical
window.  It combines:

* a TabM-style parameter-efficient neural ensemble;
* a lightweight TabR-style temporal neighbour retriever;
* the existing polynomial ridge model as a low-variance anchor.

This is an independent research implementation, not a verbatim reproduction of
the TabM or TabR reference packages.
"""

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml_quant import design as ridge_design
from ml_quant import fit as ridge_fit
from ml_quant import metric
from ml_quant import predict as ridge_predict
from ml_quant import prepare as prepare_base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
BASE_FEATURES = [
    "momentum", "value_pe", "value_pb", "value_ps", "low_volatility",
    "small_size", "reversal", "liquidity", "roe", "roa", "gross_margin",
    "cash_quality", "low_leverage",
]
DYNAMIC_FEATURES = [
    "momentum_3m", "momentum_6m", "return_stability_6m", "return_stability_12m",
    "size_growth_3m", "turnover_proxy", "max_return_quality", "value_composite",
    "quality_composite", "momentum_acceleration",
]
FEATURES = BASE_FEATURES + DYNAMIC_FEATURES


@dataclass(frozen=True)
class Config:
    start: str = "2018-01-01"
    train_months: int = 84
    top_frac: float = 0.10
    buffer_frac: float = 0.15
    cost_bps: float = 20.0
    ensemble_size: int = 4
    hidden_dim: int = 48
    max_train_samples: int = 20_000
    max_ridge_samples: int = 150_000
    max_retrieval_samples: int = 2_000
    epochs: int = 3
    patience: int = 3
    batch_size: int = 2048
    learning_rate: float = 2e-3
    pairwise_weight: float = 0.20
    signal_memory: float = 0.25
    refit_months: int = 3
    seed: int = 20260717


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _compound_return(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    return (1 + s).rolling(window, min_periods=min_periods).apply(np.prod, raw=True) - 1


def _cross_sectional_rank(p: pd.DataFrame, name: str, values: pd.Series) -> None:
    """Winsorise implicitly through ranks, then neutralise within industry."""
    p[name] = _num(values)
    within_industry = p.groupby(["month", "industry"], dropna=False)[name].rank(pct=True)
    market = p.groupby("month")[name].rank(pct=True)
    p[name] = within_industry.fillna(market) - 0.5


def prepare_advanced(panel_path: Path) -> pd.DataFrame:
    p = prepare_base(panel_path).sort_values(["Stkcd", "month"]).copy()
    stock_return = p.groupby("Stkcd", sort=False)["ret"]
    stock_size = p.groupby("Stkcd", sort=False)["size"]

    raw: dict[str, pd.Series] = {
        "momentum_3m": stock_return.transform(lambda s: _compound_return(s, 3, 2)),
        "momentum_6m": stock_return.transform(lambda s: _compound_return(s, 6, 4)),
        "return_stability_6m": -stock_return.transform(lambda s: s.rolling(6, min_periods=4).std()),
        "return_stability_12m": -stock_return.transform(lambda s: s.rolling(12, min_periods=8).std()),
        "size_growth_3m": -stock_size.pct_change(3, fill_method=None),
        "turnover_proxy": _num(p["amount"]) / _num(p["size"]).replace(0, np.nan),
        "max_return_quality": -_num(p["max_ret"]),
        "value_composite": p[["value_pe", "value_pb", "value_ps"]].mean(axis=1),
        "quality_composite": p[["roe", "roa", "gross_margin", "cash_quality", "low_leverage"]].mean(axis=1),
    }
    raw["momentum_acceleration"] = raw["momentum_3m"] - raw["momentum_6m"]
    for name, values in raw.items():
        _cross_sectional_rank(p, name, values)

    for name in FEATURES:
        p[f"{name}_missing"] = p[name].isna().astype(np.float32)
        p[name] = p[name].fillna(0).astype(np.float32)
    return p


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = FEATURES + [f"{name}_missing" for name in FEATURES]
    return df[cols].to_numpy(np.float32, copy=True)


def monthly_rank_ic(prediction: np.ndarray, target: np.ndarray, months: np.ndarray) -> float:
    frame = pd.DataFrame({"prediction": prediction, "target": target, "month": months})
    ic = frame.groupby("month", observed=True).apply(
        lambda g: g["prediction"].corr(g["target"], method="spearman"),
        include_groups=False,
    )
    return float(ic.mean())


class Standardizer:
    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
        self.scale[self.scale < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)


class TemporalRetriever:
    """A small leakage-safe analogue retriever inspired by TabR.

    Reference rows always precede the prediction month. Cosine neighbours vote
    on the cross-sectional rank target, with a similarity softmax.
    """

    def __init__(self, max_samples: int, neighbors: int = 32, seed: int = 0):
        self.max_samples = max_samples
        self.neighbors = neighbors
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray, months: np.ndarray) -> "TemporalRetriever":
        order = np.argsort(months)
        recent = order[-min(len(order), self.max_samples * 3):]
        if len(recent) > self.max_samples:
            rng = np.random.default_rng(self.seed)
            recent = rng.choice(recent, self.max_samples, replace=False)
        ref = x[recent].astype(np.float32)
        norm = np.linalg.norm(ref, axis=1, keepdims=True).clip(min=1e-6)
        self.reference = ref / norm
        self.target = y[recent].astype(np.float32)
        return self

    def predict(self, x: np.ndarray, batch_size: int = 768) -> np.ndarray:
        out = []
        k = min(self.neighbors, len(self.target))
        for lo in range(0, len(x), batch_size):
            query = x[lo:lo + batch_size]
            query = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-6)
            sim = query @ self.reference.T
            idx = np.argpartition(sim, -k, axis=1)[:, -k:]
            top_sim = np.take_along_axis(sim, idx, axis=1)
            top_y = self.target[idx]
            logits = (top_sim - top_sim.max(axis=1, keepdims=True)) / 0.05
            weight = np.exp(logits)
            weight /= weight.sum(axis=1, keepdims=True)
            out.append((weight * top_y).sum(axis=1))
        return np.concatenate(out).astype(np.float32)


def _torch_components():
    import torch
    from torch import nn

    class BatchEnsembleLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, members: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.r = nn.Parameter(torch.ones(members, in_features))
            self.s = nn.Parameter(torch.ones(members, out_features))
            self.bias = nn.Parameter(torch.zeros(members, out_features))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.normal_(self.r, 1.0, 0.08)
            nn.init.normal_(self.s, 1.0, 0.08)

        def forward(self, x):
            # x: [batch, member, feature]
            return torch.nn.functional.linear(x * self.r, self.weight) * self.s + self.bias

    class TabMRanker(nn.Module):
        def __init__(self, features: int, hidden: int, members: int):
            super().__init__()
            self.members = members
            self.first = BatchEnsembleLinear(features, hidden, members)
            self.second = BatchEnsembleLinear(hidden, hidden, members)
            self.head = BatchEnsembleLinear(hidden, 1, members)
            self.norm1 = nn.LayerNorm(hidden)
            self.norm2 = nn.LayerNorm(hidden)
            self.activation = nn.SiLU()

        def forward(self, x):
            x = x[:, None, :].expand(-1, self.members, -1)
            h = self.activation(self.norm1(self.first(x)))
            h = h + self.activation(self.norm2(self.second(h)))
            return self.head(h).squeeze(-1)

    return torch, TabMRanker


class TabMModel:
    def __init__(self, config: Config, input_dim: int, seed: int):
        torch, TabMRanker = _torch_components()
        torch.manual_seed(seed)
        self.torch = torch
        self.config = config
        self.model = TabMRanker(input_dim, config.hidden_dim, config.ensemble_size)

    def _loss(self, prediction, target):
        torch = self.torch
        pointwise = torch.nn.functional.smooth_l1_loss(prediction, target[:, None].expand_as(prediction))
        mean_prediction = prediction.mean(dim=1)
        permutation = torch.randperm(len(target))
        target_difference = target - target[permutation]
        useful = target_difference.abs() >= 0.10
        if useful.any():
            signed_margin = target_difference[useful].sign() * (
                mean_prediction[useful] - mean_prediction[permutation][useful]
            )
            pairwise = torch.nn.functional.softplus(-signed_margin / 0.10).mean()
        else:
            pairwise = pointwise.new_zeros(())
        return pointwise + self.config.pairwise_weight * pairwise

    def fit(self, x: np.ndarray, y: np.ndarray, validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
            epochs: int | None = None) -> tuple[int, float]:
        torch = self.torch
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y.astype(np.float32))
        best_epoch, best_ic, stale, best_state = 1, -np.inf, 0, None
        total_epochs = epochs or self.config.epochs
        generator = torch.Generator().manual_seed(self.config.seed + len(x))
        for epoch in range(1, total_epochs + 1):
            self.model.train()
            order = torch.randperm(len(x_tensor), generator=generator)
            for lo in range(0, len(order), self.config.batch_size):
                idx = order[lo:lo + self.config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(self.model(x_tensor[idx]), y_tensor[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            if validation is None:
                continue
            vx, vy, vm = validation
            score = monthly_rank_ic(self.predict(vx), vy, vm)
            if score > best_ic + 1e-4:
                best_epoch, best_ic, stale = epoch, score, 0
                best_state = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.config.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_epoch, best_ic

    def predict(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        result = []
        with torch.inference_mode():
            for lo in range(0, len(x), batch_size):
                result.append(self.model(torch.from_numpy(x[lo:lo + batch_size])).mean(dim=1).numpy())
        return np.concatenate(result).astype(np.float32)


def _sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.sample(maximum, random_state=seed).sort_values("month")


def _select_blend(predictions: list[np.ndarray], target: np.ndarray, months: np.ndarray) -> tuple[np.ndarray, float]:
    # Small simplex grid: enough flexibility without turning validation into a large search.
    candidates = []
    for neural_weight in [0.0, 0.25, 0.50, 0.75, 1.0]:
        for retrieval_weight in [0.0, 0.15, 0.30]:
            if neural_weight + retrieval_weight > 1.0:
                continue
            ridge_weight = 1.0 - neural_weight - retrieval_weight
            blend = ridge_weight * predictions[0] + neural_weight * predictions[1] + retrieval_weight * predictions[2]
            candidates.append((monthly_rank_ic(blend, target, months), np.array([ridge_weight, neural_weight, retrieval_weight])))
    score, weight = max(candidates, key=lambda item: item[0])
    return weight, score


def walk_forward(p: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    logs: list[dict] = []
    previous_names: set = set()
    previous_signal: dict = {}
    fitted = None
    refit_counter = 0

    months = sorted(pd.Timestamp(m) for m in p.loc[p["month"] >= config.start, "month"].unique())
    for month in months:
        should_refit = fitted is None or refit_counter >= config.refit_months
        if should_refit:
            lower = month - pd.DateOffset(months=config.train_months)
            history = p[(p["month"] < month) & (p["month"] >= lower) & p["eligible"] & p["target"].notna()].copy()
            if len(history) < 10_000:
                continue
            validation_start = history["month"].max() - pd.DateOffset(months=12)
            train_frame = history[history["month"] <= validation_start]
            validation_frame = history[history["month"] > validation_start]
            sampled_train = _sample(train_frame, config.max_train_samples, config.seed + month.year * 100 + month.month)

            standardizer = Standardizer().fit(feature_matrix(sampled_train))
            x_train = standardizer.transform(feature_matrix(sampled_train))
            y_train = sampled_train["target"].to_numpy(np.float32)
            x_validation = standardizer.transform(feature_matrix(validation_frame))
            y_validation = validation_frame["target"].to_numpy(np.float32)
            validation_months = validation_frame["month"].to_numpy()

            neural_trial = TabMModel(config, x_train.shape[1], config.seed + month.year * 100 + month.month)
            best_epoch, neural_ic = neural_trial.fit(
                x_train, y_train, (x_validation, y_validation, validation_months)
            )

            ridge_train = _sample(
                train_frame, config.max_ridge_samples, config.seed + 31 + month.year * 100 + month.month
            )
            ridge_x, medians = ridge_design(ridge_train)
            ridge_y = ridge_train["target"].to_numpy(np.float32)
            ridge = ridge_fit(ridge_x, ridge_y, alpha=1000.0)
            ridge_validation_x, _ = ridge_design(validation_frame, medians)
            ridge_validation = ridge_predict(ridge_validation_x, ridge)

            retriever = TemporalRetriever(
                config.max_retrieval_samples, seed=config.seed + month.year * 100 + month.month
            ).fit(x_train, y_train, sampled_train["month"].to_numpy())
            retrieval_validation = retriever.predict(x_validation)
            neural_validation = neural_trial.predict(x_validation)
            weights, validation_ic = _select_blend(
                [ridge_validation, neural_validation, retrieval_validation], y_validation, validation_months
            )

            full_frame = _sample(history, config.max_train_samples, config.seed + 17 + month.year * 100 + month.month)
            standardizer = Standardizer().fit(feature_matrix(full_frame))
            x_full = standardizer.transform(feature_matrix(full_frame))
            y_full = full_frame["target"].to_numpy(np.float32)
            neural = TabMModel(config, x_full.shape[1], config.seed + month.year * 100 + month.month)
            neural.fit(x_full, y_full, validation=None, epochs=best_epoch)
            ridge_full = _sample(
                history, config.max_ridge_samples, config.seed + 47 + month.year * 100 + month.month
            )
            ridge_x, medians = ridge_design(ridge_full)
            ridge = ridge_fit(ridge_x, ridge_full["target"].to_numpy(np.float32), alpha=1000.0)
            retriever = TemporalRetriever(
                config.max_retrieval_samples, seed=config.seed + 17 + month.year * 100 + month.month
            ).fit(x_full, y_full, full_frame["month"].to_numpy())
            fitted = (standardizer, neural, ridge, medians, retriever, weights)
            logs.append({
                "refit_month": month, "history_samples": len(history), "train_samples": len(full_frame),
                "ridge_samples": len(ridge_full),
                "best_epoch": best_epoch, "tabm_validation_ic": neural_ic,
                "ensemble_validation_ic": validation_ic, "ridge_weight": weights[0],
                "tabm_weight": weights[1], "retrieval_weight": weights[2],
            })
            refit_counter = 0

        test = p[(p["month"] == month) & p["eligible"]].copy()
        if fitted is None or test.empty:
            continue
        standardizer, neural, ridge, medians, retriever, weights = fitted
        x_test = standardizer.transform(feature_matrix(test))
        ridge_test_x, _ = ridge_design(test, medians)
        components = [ridge_predict(ridge_test_x, ridge), neural.predict(x_test), retriever.predict(x_test)]
        raw_signal = sum(weight * prediction for weight, prediction in zip(weights, components))
        test["raw_prediction"] = raw_signal
        old = test["Stkcd"].map(previous_signal)
        test["prediction"] = np.where(
            old.notna(), (1 - config.signal_memory) * test["raw_prediction"] + config.signal_memory * old,
            test["raw_prediction"],
        )
        previous_signal = dict(zip(test["Stkcd"], test["prediction"]))

        count = max(1, int(np.ceil(len(test) * config.top_frac)))
        buffer_count = max(count, int(np.ceil(len(test) * config.buffer_frac)))
        retained = previous_names & set(test.nlargest(buffer_count, "prediction")["Stkcd"])
        additions = test[~test["Stkcd"].isin(retained)].nlargest(max(0, count - len(retained)), "prediction")
        chosen = pd.concat([test[test["Stkcd"].isin(retained)], additions]).drop_duplicates("Stkcd").head(count)
        names = set(chosen["Stkcd"])
        turnover = 1.0 if not previous_names else 1 - len(names & previous_names) / max(len(names), 1)
        rows.append({
            "month": month + pd.offsets.MonthEnd(1), "gross_return": chosen["forward_return"].mean(),
            "turnover": turnover, "holdings": len(names), "mean_prediction": chosen["prediction"].mean(),
            "ridge_weight": weights[0], "tabm_weight": weights[1], "retrieval_weight": weights[2],
        })
        previous_names = names
        refit_counter += 1

    result = pd.DataFrame(rows).dropna(subset=["gross_return"])
    result["net_return"] = result["gross_return"] - result["turnover"] * config.cost_bps / 10_000
    result["advanced_nav"] = (1 + result["net_return"]).cumprod()
    return result, pd.DataFrame(logs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=OUT / "monthly_panel.csv.gz")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--train-months", type=int, default=84)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=20_000)
    parser.add_argument("--refit-months", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--quick", action="store_true", help="Fast smoke test with fewer samples and epochs")
    args = parser.parse_args()

    config = Config(
        start=args.start, train_months=args.train_months, epochs=args.epochs,
        max_train_samples=args.max_train_samples, refit_months=args.refit_months, cost_bps=args.cost_bps,
    )
    if args.quick:
        config = Config(**{**asdict(config), "epochs": 2, "max_train_samples": 8_000,
                           "max_retrieval_samples": 1_000, "ensemble_size": 3})
    seed_everything(config.seed)
    OUT.mkdir(exist_ok=True)
    panel = prepare_advanced(args.panel)
    result, log = walk_forward(panel, config)

    baseline = pd.read_csv(OUT / "deep_learning_backtest.csv", parse_dates=["month"])
    result = result.merge(
        baseline[["month", "net_return", "benchmark_return"]].rename(columns={"net_return": "previous_dl_return"}),
        on="month", how="left",
    )
    result["previous_dl_nav"] = (1 + result["previous_dl_return"]).cumprod()
    result["benchmark_nav"] = (1 + result["benchmark_return"]).cumprod()
    result.to_csv(OUT / "advanced_backtest.csv", index=False)
    log.to_csv(OUT / "advanced_model_log.csv", index=False)
    metrics = pd.DataFrame([
        {"series": "tabm_rank_retrieval_ensemble", **metric(result["net_return"]),
         "avg_turnover": result["turnover"].mean()},
        {"series": "previous_mlp_ridge", **metric(result["previous_dl_return"]),
         "avg_turnover": baseline["turnover"].mean()},
        {"series": "benchmark", **metric(result["benchmark_return"]), "avg_turnover": np.nan},
    ])
    metrics.to_csv(OUT / "advanced_metrics.csv", index=False)
    (OUT / "advanced_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
