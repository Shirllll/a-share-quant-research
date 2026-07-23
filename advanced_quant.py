from __future__ import annotations

"""Walk-forward A-share cross-sectional alpha research.

The primary model is an auditable linear ridge baseline. A TabM-style neural
ensemble and a temporal neighbour retriever are admitted only when they improve
historical validation IC by a pre-declared margin. The long-short portfolio is
an industry-balanced research diagnostic, not an executable A-share short book.
"""

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml_quant import fit as ridge_fit
from ml_quant import metric
from ml_quant import predict as ridge_predict
from data_cleaning import clean_monthly_panel, cleaning_report


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
BASE_FEATURES = [
    "long_term_reversal_12_1", "earnings_yield", "book_to_market", "sales_to_price",
    "low_volatility", "large_cap_stability", "short_reversal", "illiquidity_premium",
    "roe", "roa", "gross_margin", "cash_conversion",
]
DYNAMIC_FEATURES = [
    "medium_reversal_3m", "medium_reversal_6m", "return_stability_6m",
    "return_stability_12m", "downside_stability_12m", "low_turnover",
    "lottery_avoidance", "residual_reversal", "profitability_momentum",
    "value_composite", "quality_composite",
]
FEATURES = BASE_FEATURES + DYNAMIC_FEATURES
FACTOR_DEFINITIONS = {
    "long_term_reversal_12_1": ("lower_raw_better", "negative 12-1 return; direction fixed on 2014-2017 formation data"),
    "earnings_yield": ("higher_raw_better", "1 / positive PE"),
    "book_to_market": ("higher_raw_better", "1 / positive PB"),
    "sales_to_price": ("higher_raw_better", "1 / positive PS"),
    "low_volatility": ("lower_raw_better", "negative annualised realised volatility"),
    "large_cap_stability": ("higher_raw_better", "positive log free-float market value; small-cap sign failed formation audit"),
    "short_reversal": ("lower_raw_better", "negative current-month return"),
    "illiquidity_premium": ("higher_raw_better", "positive Amihud-style illiquidity"),
    "roe": ("higher_raw_better", "clean ROE TTM"),
    "roa": ("higher_raw_better", "clean ROA TTM"),
    "gross_margin": ("higher_raw_better", "clean gross margin TTM"),
    "cash_conversion": ("higher_raw_better", "clean CFO / total profit ratio"),
    "medium_reversal_3m": ("lower_raw_better", "negative three-month compounded return"),
    "medium_reversal_6m": ("lower_raw_better", "negative six-month compounded return"),
    "return_stability_6m": ("lower_raw_better", "negative six-month return volatility"),
    "return_stability_12m": ("lower_raw_better", "negative twelve-month return volatility"),
    "downside_stability_12m": ("lower_raw_better", "negative twelve-month downside volatility"),
    "low_turnover": ("lower_raw_better", "negative traded amount / free-float market value"),
    "lottery_avoidance": ("lower_raw_better", "negative maximum daily return"),
    "residual_reversal": ("lower_raw_better", "negative current return scaled by volatility"),
    "profitability_momentum": ("higher_raw_better", "year-over-year ROE and ROA improvement"),
    "value_composite": ("higher_score_better", "average clean value rank"),
    "quality_composite": ("higher_score_better", "average clean quality rank"),
}


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
    min_nonlinear_ic_gain: float = 0.005
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
    p = clean_monthly_panel(panel_path).sort_values(["Stkcd", "month"]).copy()
    stock_return = p.groupby("Stkcd", sort=False)["ret_clean"]
    momentum_3m = stock_return.transform(lambda s: _compound_return(s, 3, 2))
    momentum_6m = stock_return.transform(lambda s: _compound_return(s, 6, 4))
    momentum_12_1 = stock_return.transform(
        lambda s: (1 + s.shift(1)).rolling(11, min_periods=8).apply(np.prod, raw=True) - 1
    )
    downside = stock_return.transform(
        lambda s: s.where(s < 0).rolling(12, min_periods=5).std()
    )
    roe_change = p.groupby("Stkcd", sort=False)["F050504C_clean"].diff(12)
    roa_change = p.groupby("Stkcd", sort=False)["F050204C_clean"].diff(12)

    raw: dict[str, pd.Series] = {
        "long_term_reversal_12_1": -momentum_12_1,
        "earnings_yield": 1 / p["PE1TTM_clean"],
        "book_to_market": 1 / p["PBV1B_clean"],
        "sales_to_price": 1 / p["PSTTM_clean"],
        "low_volatility": -p["volatility_clean"],
        "large_cap_stability": np.log(p["size_clean"]),
        "short_reversal": -p["ret_clean"],
        "illiquidity_premium": p["illiq_clean"],
        "roe": p["F050504C_clean"],
        "roa": p["F050204C_clean"],
        "gross_margin": p["F053301C_clean"],
        "cash_conversion": p["F052901C_clean"],
        "medium_reversal_3m": -momentum_3m,
        "medium_reversal_6m": -momentum_6m,
        "return_stability_6m": -stock_return.transform(lambda s: s.rolling(6, min_periods=4).std()),
        "return_stability_12m": -stock_return.transform(lambda s: s.rolling(12, min_periods=8).std()),
        "downside_stability_12m": -downside,
        "low_turnover": -(p["amount_clean"] / p["size_clean"]),
        "lottery_avoidance": -p["max_ret_clean"],
        "residual_reversal": -p["ret_clean"] / p["volatility_clean"],
        "profitability_momentum": pd.concat([roe_change, roa_change], axis=1).mean(axis=1),
    }
    for name, values in raw.items():
        _cross_sectional_rank(p, name, values)
    p["value_composite"] = p[["earnings_yield", "book_to_market", "sales_to_price"]].mean(axis=1)
    p["quality_composite"] = p[["roe", "roa", "gross_margin", "cash_conversion"]].mean(axis=1)

    p["forward_return"] = p.groupby("Stkcd", sort=False)["ret_clean"].shift(-1)
    market_target = p.groupby("month")["forward_return"].rank(pct=True) - 0.5
    industry_target = p.groupby(["month", "industry"], dropna=False)["forward_return"].rank(pct=True) - 0.5
    p["target"] = industry_target.fillna(market_target)
    amount_cut = p.groupby("month")["amount_clean"].transform(lambda s: s.quantile(0.20))
    feature_coverage = p[FEATURES].notna().sum(axis=1)
    history_months = p.groupby("Stkcd", sort=False).cumcount() + 1
    p["eligible"] = (
        (p["trdsta"] == 1) & (p["listed_days"] >= 180) & (p["trading_days"] >= 10)
        & (p["amount_clean"] >= amount_cut)
        & ~p["flag_market_data_invalid"] & ~p["flag_special_treatment"]
        & ~p["flag_abnormal_listing"] & (feature_coverage >= 10) & (history_months >= 8)
    )

    # Compatibility aliases used only by the legacy polynomial ridge design.
    aliases = {
        "momentum": "long_term_reversal_12_1", "value_pe": "earnings_yield",
        "value_pb": "book_to_market", "value_ps": "sales_to_price",
        "reversal": "short_reversal", "liquidity": "illiquidity_premium",
        "cash_quality": "cash_conversion",
    }
    for legacy, corrected in aliases.items():
        p[legacy] = p[corrected]
    p["small_size"] = p["large_cap_stability"]
    p["low_leverage"] = 0.0
    for name in FEATURES:
        p[f"{name}_missing"] = p[name].isna().astype(np.float32)
        p[name] = p[name].fillna(0).astype(np.float32)
    return p


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = FEATURES + [f"{name}_missing" for name in FEATURES]
    return df[cols].to_numpy(np.float32, copy=True)


def linear_feature_names() -> list[str]:
    return FEATURES + [f"{name}_missing" for name in FEATURES]


def linear_design(df: pd.DataFrame, medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Auditable linear baseline: cleaned factors and missing flags, without interactions."""
    x = feature_matrix(df).astype(float)
    if medians is None:
        medians = np.nan_to_num(np.nanmedian(x, axis=0), nan=0.0)
    return np.where(np.isnan(x), medians, x), medians


def monthly_rank_ic(prediction: np.ndarray, target: np.ndarray, months: np.ndarray) -> float:
    frame = pd.DataFrame({"prediction": prediction, "target": target, "month": months})
    ic = frame.groupby("month", observed=True).apply(
        lambda g: g["prediction"].corr(g["target"], method="spearman"),
        include_groups=False,
    )
    return float(ic.mean())


def factor_direction_audit(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = {
        "formation_2014_2017": panel["month"] < "2018-01-01",
        "out_of_sample_2018_2025": panel["month"] >= "2018-01-01",
    }
    for factor in FEATURES:
        definition = FACTOR_DEFINITIONS[factor]
        row = {"factor": factor, "economic_preference": definition[0], "implemented_transform": definition[1]}
        values = panel[factor].mask(panel[f"{factor}_missing"].eq(1))
        for label, period in periods.items():
            sample = panel[period & panel["eligible"] & panel["target"].notna()].copy()
            sample["factor_value"] = values.loc[sample.index]
            ic = sample.groupby("month", observed=True).apply(
                lambda group: group["factor_value"].corr(group["target"], method="spearman"),
                include_groups=False,
            )
            row[f"mean_ic_{label}"] = ic.mean()
            row[f"positive_month_share_{label}"] = (ic > 0).mean()
            row[f"months_{label}"] = ic.count()
        rows.append(row)
    return pd.DataFrame(rows)


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


def _select_blend(predictions: list[np.ndarray], target: np.ndarray, months: np.ndarray,
                  minimum_gain: float) -> tuple[np.ndarray, float, float, bool]:
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
    ridge_score = monthly_rank_ic(predictions[0], target, months)
    enabled = bool(score >= ridge_score + minimum_gain and weight[0] < 1.0)
    if not enabled:
        return np.array([1.0, 0.0, 0.0]), ridge_score, ridge_score, False
    return weight, score, ridge_score, True


def industry_balanced_diagnostic(test: pd.DataFrame, tail_fraction: float = 0.10) -> tuple[pd.Series, pd.Series]:
    """Return exact industry-balanced long and short weights.

    Industries with fewer than ten eligible names are excluded. Every included
    industry receives equal capital on each leg, with equal weights inside the
    industry's upper or lower prediction tail.
    """
    groups: list[tuple[pd.Index, pd.Index]] = []
    for _, group in test.groupby("industry", dropna=False):
        if len(group) < 10:
            continue
        tail_count = max(1, int(np.floor(len(group) * tail_fraction)))
        ordered = group.sort_values("prediction")
        groups.append((ordered.tail(tail_count).index, ordered.head(tail_count).index))
    if not groups:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    industry_weight = 1.0 / len(groups)
    long_weights: dict[int, float] = {}
    short_weights: dict[int, float] = {}
    for long_index, short_index in groups:
        for index in long_index:
            long_weights[index] = industry_weight / len(long_index)
        for index in short_index:
            short_weights[index] = industry_weight / len(short_index)
    return pd.Series(long_weights, dtype=float), pd.Series(short_weights, dtype=float)


def leg_turnover(current: dict[str, float], previous: dict[str, float]) -> float:
    """One-way turnover for a fully invested leg."""
    if not previous:
        return 1.0
    names = set(current) | set(previous)
    return 0.5 * sum(abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in names)


def walk_forward(p: pd.DataFrame, config: Config):
    rows, logs, coefficient_rows, decile_rows, trade_rows = [], [], [], [], []
    previous_names, previous_weights, previous_signal = set(), {}, {}
    previous_amount: dict[str, float] = {}
    previous_diag_long: dict[str, float] = {}
    previous_diag_short: dict[str, float] = {}
    fitted, refit_counter = None, 0

    months = sorted(pd.Timestamp(m) for m in p.loc[p["month"] >= config.start, "month"].unique())
    for month in months:
        if fitted is None or refit_counter >= config.refit_months:
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
            best_epoch, neural_ic = neural_trial.fit(x_train, y_train, (x_validation, y_validation, validation_months))

            ridge_train = _sample(train_frame, config.max_ridge_samples, config.seed + 31 + month.year * 100 + month.month)
            ridge_x, medians = linear_design(ridge_train)
            ridge_y = ridge_train["target"].to_numpy(np.float32)
            ridge_validation_x, _ = linear_design(validation_frame, medians)
            ridge_candidates = []
            for alpha in [10.0, 100.0, 1000.0, 10000.0]:
                candidate = ridge_fit(ridge_x, ridge_y, alpha=alpha)
                prediction = ridge_predict(ridge_validation_x, candidate)
                score = monthly_rank_ic(prediction, y_validation, validation_months)
                ridge_candidates.append((score, alpha, candidate, prediction))
            ridge_ic, ridge_alpha, ridge, ridge_validation = max(ridge_candidates, key=lambda item: item[0])

            retriever = TemporalRetriever(config.max_retrieval_samples, seed=config.seed + month.year * 100 + month.month).fit(
                x_train, y_train, sampled_train["month"].to_numpy()
            )
            neural_validation = neural_trial.predict(x_validation)
            retrieval_validation = retriever.predict(x_validation)
            weights, validation_ic, ridge_ic, nonlinear_enabled = _select_blend(
                [ridge_validation, neural_validation, retrieval_validation], y_validation,
                validation_months, config.min_nonlinear_ic_gain,
            )

            full_frame = _sample(history, config.max_train_samples, config.seed + 17 + month.year * 100 + month.month)
            standardizer = Standardizer().fit(feature_matrix(full_frame))
            x_full = standardizer.transform(feature_matrix(full_frame))
            y_full = full_frame["target"].to_numpy(np.float32)
            neural = TabMModel(config, x_full.shape[1], config.seed + month.year * 100 + month.month)
            neural.fit(x_full, y_full, validation=None, epochs=best_epoch)
            ridge_full = _sample(history, config.max_ridge_samples, config.seed + 47 + month.year * 100 + month.month)
            ridge_x, medians = linear_design(ridge_full)
            ridge = ridge_fit(ridge_x, ridge_full["target"].to_numpy(np.float32), alpha=ridge_alpha)
            retriever = TemporalRetriever(config.max_retrieval_samples, seed=config.seed + 17 + month.year * 100 + month.month).fit(
                x_full, y_full, full_frame["month"].to_numpy()
            )
            fitted = (standardizer, neural, ridge, medians, retriever, weights, nonlinear_enabled)
            history_start = history["month"].min()
            history_end = history["month"].max()
            max_label_realization_month = history_end + pd.offsets.MonthEnd(1)
            logs.append({
                "refit_month": month, "prediction_month": month,
                "history_start": history_start, "history_end": history_end,
                "max_label_realization_month": max_label_realization_month,
                "future_leakage_pass": max_label_realization_month <= month,
                "history_samples": len(history), "ridge_samples": len(ridge_full),
                "ridge_alpha": ridge_alpha, "best_epoch": best_epoch, "ridge_validation_ic": ridge_ic,
                "tabm_validation_ic": neural_ic, "selected_validation_ic": validation_ic,
                "incremental_validation_ic": validation_ic - ridge_ic, "nonlinear_enabled": nonlinear_enabled,
                "ridge_weight": weights[0], "tabm_weight": weights[1], "retrieval_weight": weights[2],
            })
            for feature, coefficient in zip(linear_feature_names(), ridge[0][1:]):
                coefficient_rows.append({"refit_month": month, "feature": feature, "standardized_coefficient": coefficient})
            refit_counter = 0

        all_month = p[p["month"] == month]
        test = all_month[all_month["eligible"]].copy()
        if fitted is None or test.empty:
            continue
        standardizer, neural, ridge, medians, retriever, weights, nonlinear_enabled = fitted
        x_test = standardizer.transform(feature_matrix(test))
        ridge_test_x, _ = linear_design(test, medians)
        components = [ridge_predict(ridge_test_x, ridge), neural.predict(x_test), retriever.predict(x_test)]
        test["linear_prediction"] = components[0]
        test["raw_prediction"] = sum(weight * prediction for weight, prediction in zip(weights, components))
        old = test["Stkcd"].map(previous_signal)
        test["prediction"] = np.where(
            old.notna(), (1 - config.signal_memory) * test["raw_prediction"] + config.signal_memory * old,
            test["raw_prediction"],
        )
        previous_signal = dict(zip(test["Stkcd"], test["prediction"]))

        industry_percentile = test.groupby("industry", dropna=False)["prediction"].rank(pct=True)
        industry_percentile = industry_percentile.fillna(test["prediction"].rank(pct=True))
        test["diagnostic_decile"] = np.ceil(industry_percentile * 10).clip(1, 10).astype(int)
        for decile, group in test.groupby("diagnostic_decile"):
            decile_rows.append({"month": month + pd.offsets.MonthEnd(1), "decile": decile,
                                "return": group["forward_return"].mean(), "stocks": len(group)})
        diag_long_weights, diag_short_weights = industry_balanced_diagnostic(test, config.top_frac)
        diag_long = dict(zip(test.loc[diag_long_weights.index, "Stkcd"], diag_long_weights))
        diag_short = dict(zip(test.loc[diag_short_weights.index, "Stkcd"], diag_short_weights))
        diag_long_turnover = leg_turnover(diag_long, previous_diag_long)
        diag_short_turnover = leg_turnover(diag_short, previous_diag_short)
        long_short_gross = (
            float(np.dot(diag_long_weights, test.loc[diag_long_weights.index, "forward_return"]))
            - float(np.dot(diag_short_weights, test.loc[diag_short_weights.index, "forward_return"]))
        )
        long_short_turnover = diag_long_turnover + diag_short_turnover

        count = max(1, int(np.ceil(len(test) * config.top_frac)))
        buffer_count = max(count, int(np.ceil(len(test) * config.buffer_frac)))
        retained = previous_names & set(test.nlargest(buffer_count, "prediction")["Stkcd"])
        additions = test[~test["Stkcd"].isin(retained)].nlargest(max(0, count - len(retained)), "prediction")
        chosen = pd.concat([test[test["Stkcd"].isin(retained)], additions]).drop_duplicates("Stkcd").head(count)
        names = set(chosen["Stkcd"])
        portfolio_weights = pd.Series(1 / len(chosen), index=chosen.index)
        current_weights = dict(zip(chosen["Stkcd"], portfolio_weights))
        union = set(previous_weights) | set(current_weights)
        turnover = 1.0 if not previous_weights else 0.5 * sum(
            abs(current_weights.get(name, 0) - previous_weights.get(name, 0)) for name in union
        )
        amount_map = dict(zip(all_month["Stkcd"], all_month["amount_clean"]))
        for name in union:
            trade_weight = abs(current_weights.get(name, 0) - previous_weights.get(name, 0))
            if trade_weight > 1e-10:
                trade_rows.append({
                    "month": month + pd.offsets.MonthEnd(1), "Stkcd": name, "trade_weight": trade_weight,
                    "direction": "buy" if current_weights.get(name, 0) > previous_weights.get(name, 0) else "sell",
                    "average_daily_amount": amount_map.get(name, previous_amount.get(name, np.nan)),
                    "position_weight_after": current_weights.get(name, 0),
                })
        weighted_return = float(np.dot(portfolio_weights, chosen["forward_return"]))
        rows.append({
            "month": month + pd.offsets.MonthEnd(1), "gross_return": weighted_return,
            "turnover": turnover, "holdings": len(names), "mean_prediction": chosen["prediction"].mean(),
            "cross_sectional_ic": test["prediction"].corr(test["target"], method="spearman"),
            "linear_ic": test["linear_prediction"].corr(test["target"], method="spearman"),
            "long_short_gross_return": long_short_gross, "long_short_turnover": long_short_turnover,
            "diagnostic_long_names": len(diag_long), "diagnostic_short_names": len(diag_short),
            "nonlinear_enabled": nonlinear_enabled, "ridge_weight": weights[0],
            "tabm_weight": weights[1], "retrieval_weight": weights[2],
        })
        previous_names, previous_weights = names, current_weights
        previous_amount = {name: amount_map.get(name, previous_amount.get(name, np.nan)) for name in names}
        previous_diag_long, previous_diag_short = diag_long, diag_short
        refit_counter += 1

    result = pd.DataFrame(rows).dropna(subset=["gross_return"])
    result["net_return"] = result["gross_return"] - result["turnover"] * config.cost_bps / 10_000
    result["long_short_net_return"] = result["long_short_gross_return"] - result["long_short_turnover"] * config.cost_bps / 10_000
    result["advanced_nav"] = (1 + result["net_return"]).cumprod()
    result["long_short_nav"] = (1 + result["long_short_net_return"]).cumprod()
    return result, pd.DataFrame(logs), pd.DataFrame(coefficient_rows), pd.DataFrame(decile_rows), pd.DataFrame(trade_rows)


def leakage_audit(logs: pd.DataFrame) -> pd.DataFrame:
    """Machine-check that every training label was realised by the signal month."""
    columns = [
        "refit_month", "prediction_month", "history_start", "history_end",
        "max_label_realization_month", "future_leakage_pass",
    ]
    audit = logs[columns].copy()
    for column in columns[:-1]:
        audit[column] = pd.to_datetime(audit[column])
    audit["future_leakage_pass"] = (
        audit["future_leakage_pass"].astype(bool)
        & audit["history_end"].lt(audit["prediction_month"])
        & audit["max_label_realization_month"].le(audit["prediction_month"])
    )
    if not audit["future_leakage_pass"].all():
        raise RuntimeError("Future leakage audit failed: a training label extends beyond its prediction month")
    return audit


def alpha_diagnostics(result: pd.DataFrame, deciles: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    monthly_ic = result["cross_sectional_ic"].dropna()
    linear_ic = result["linear_ic"].dropna()
    long_short = result["long_short_net_return"].dropna()
    decile_average = deciles.groupby("decile")["return"].mean()
    monotonicity = decile_average.index.to_series().corr(decile_average, method="spearman")
    rows = [
        {"metric": "mean_cross_sectional_ic", "value": monthly_ic.mean()},
        {"metric": "cross_sectional_icir_annualized", "value": monthly_ic.mean() / monthly_ic.std() * np.sqrt(12)},
        {"metric": "positive_ic_month_share", "value": (monthly_ic > 0).mean()},
        {"metric": "mean_linear_ic", "value": linear_ic.mean()},
        {"metric": "mean_final_minus_linear_ic", "value": (monthly_ic - linear_ic).mean()},
        {"metric": "long_short_monthly_t_stat", "value": long_short.mean() / (long_short.std() / np.sqrt(len(long_short)))},
        {"metric": "decile_monotonicity_spearman", "value": monotonicity},
        {"metric": "nonlinear_refit_share", "value": logs["nonlinear_enabled"].mean()},
        {"metric": "mean_validation_incremental_ic", "value": logs["incremental_validation_ic"].mean()},
    ]
    return pd.DataFrame(rows)


def capacity_analysis(trades: pd.DataFrame, trading_days: int = 5) -> pd.DataFrame:
    rows = []
    valid = trades.dropna(subset=["average_daily_amount"]).copy()
    for capital in [10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000]:
        participation = capital * valid["trade_weight"] / (valid["average_daily_amount"] * trading_days)
        rows.append({
            "portfolio_capital_rmb": capital, "execution_days": trading_days,
            "median_participation": participation.median(), "p90_participation": participation.quantile(0.90),
            "p95_participation": participation.quantile(0.95), "p99_participation": participation.quantile(0.99),
            "trades_over_10pct_adv_share": (participation > 0.10).mean(),
            "trades_over_20pct_adv_share": (participation > 0.20).mean(),
        })
    return pd.DataFrame(rows)


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
    cleaning_report(panel).to_csv(OUT / "data_quality_report.csv", index=False)
    factor_direction_audit(panel).to_csv(OUT / "factor_direction_audit.csv", index=False)
    result, log, coefficients, deciles, trades = walk_forward(panel, config)

    baseline = pd.read_csv(OUT / "backtest_monthly.csv", parse_dates=["month"])
    result = result.merge(
        baseline[["month", "benchmark_return"]],
        on="month", how="left",
    )
    result["benchmark_nav"] = (1 + result["benchmark_return"].fillna(0)).cumprod()
    result.to_csv(OUT / "advanced_backtest.csv", index=False)
    log.to_csv(OUT / "advanced_model_log.csv", index=False)
    leakage_audit(log).to_csv(OUT / "advanced_leakage_audit.csv", index=False)
    coefficients.to_csv(OUT / "linear_factor_coefficients.csv", index=False)
    deciles.to_csv(OUT / "decile_returns.csv", index=False)
    trades.to_csv(OUT / "trade_capacity_inputs.csv", index=False)
    capacity_analysis(trades).to_csv(OUT / "capacity_analysis.csv", index=False)
    alpha_diagnostics(result, deciles, log).to_csv(OUT / "alpha_diagnostics.csv", index=False)
    metrics = pd.DataFrame([
        {"series": "cleaned_gated_long_only", **metric(result["net_return"]),
         "avg_turnover": result["turnover"].mean()},
        {"series": "industry_neutral_long_short_diagnostic", **metric(result["long_short_net_return"]),
         "avg_turnover": result["long_short_turnover"].mean()},
        {"series": "benchmark", **metric(result["benchmark_return"]), "avg_turnover": np.nan},
    ])
    metrics.to_csv(OUT / "advanced_metrics.csv", index=False)
    (OUT / "advanced_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
