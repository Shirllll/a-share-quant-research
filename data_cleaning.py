from __future__ import annotations

"""Leakage-safe cleaning and eligibility rules for the monthly A-share panel."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def resolve_data_dir() -> Path | None:
    candidates = [ROOT / "data", ROOT.parents[1] / "data"]
    return next((path for path in candidates if path.exists()), None)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _read_zip_csv(path: Path, prefix: str, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if Path(name).name.startswith(prefix) and name.endswith(".csv"))
        with archive.open(member) as stream:
            return pd.read_csv(stream, usecols=columns, dtype={columns[0]: "string"}, low_memory=False)


def _event_state(panel: pd.DataFrame, path: Path, prefix: str) -> pd.Series:
    events = _read_zip_csv(path, prefix, ["Stkcd", "Chgtype", "Execudt"])
    events["Stkcd"] = events["Stkcd"].astype("string").str.zfill(6)
    events["effective_date"] = pd.to_datetime(events["Execudt"], errors="coerce")
    # CSMAR uses transitions such as AB and BA; the final character is the state after the event.
    events["state"] = events["Chgtype"].astype("string").str.strip().str[-1]
    events = events.dropna(subset=["effective_date"]).drop_duplicates(
        ["Stkcd", "effective_date"], keep="last"
    ).sort_values(["effective_date", "Stkcd"])
    left = panel[["Stkcd", "month"]].sort_values(["month", "Stkcd"])
    state = pd.merge_asof(
        left, events[["Stkcd", "effective_date", "state"]],
        left_on="month", right_on="effective_date", by="Stkcd", direction="backward",
    )
    return state.set_index(left.index)["state"].reindex(panel.index).fillna("A")


def attach_security_states(panel: pd.DataFrame, data_dir: Path | None = None) -> pd.DataFrame:
    panel = panel.copy()
    data_dir = data_dir or resolve_data_dir()
    panel["special_state"] = "A"
    panel["listing_state"] = "A"
    if data_dir is None:
        return panel
    special_path = data_dir / "05_special_treatment_change.csv.zip"
    listing_path = data_dir / "06_listing_status_change.csv.zip"
    if special_path.exists():
        panel["special_state"] = _event_state(panel, special_path, "SPT_Trdchg")
    if listing_path.exists():
        panel["listing_state"] = _event_state(panel, listing_path, "SPT_LTDSTACHG")
    return panel


def _valid_range(series: pd.Series, lower: float, upper: float) -> pd.Series:
    x = numeric(series)
    return x.where(x.between(lower, upper))


def clean_monthly_panel(path: Path) -> pd.DataFrame:
    date_columns = ["month", "Listdt", "Annodt", "Accper"]
    panel = pd.read_csv(path, parse_dates=date_columns, low_memory=False)
    panel["Stkcd"] = panel["Stkcd"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)
    panel = panel.sort_values(["Stkcd", "month"]).drop_duplicates(["Stkcd", "month"], keep="last")

    panel["ret_raw"] = numeric(panel["ret"])
    panel["flag_return_outlier"] = ~panel["ret_raw"].between(-0.95, 3.0)
    panel["ret_clean"] = panel["ret_raw"].clip(-0.95, 3.0)
    panel["amount_clean"] = numeric(panel["amount"]).where(numeric(panel["amount"]) > 0)
    panel["size_clean"] = numeric(panel["size"]).where(numeric(panel["size"]) > 0)
    panel["volatility_clean"] = _valid_range(panel["volatility"], 0.03, 3.0)
    panel["illiq_clean"] = numeric(panel["illiq"]).where(numeric(panel["illiq"]) >= 0)
    panel["max_ret_clean"] = _valid_range(panel["max_ret"], -0.50, 0.50)
    panel["flag_market_data_invalid"] = panel[
        ["amount_clean", "size_clean", "volatility_clean", "illiq_clean", "max_ret_clean"]
    ].isna().any(axis=1)

    # Negative or implausibly large multiples are not comparable value signals.
    valuation_rules = {"PE1TTM": (0.5, 500.0), "PBV1B": (0.05, 50.0), "PSTTM": (0.01, 100.0)}
    for column, bounds in valuation_rules.items():
        panel[f"{column}_clean"] = _valid_range(panel[column], *bounds)
    panel["flag_valuation_invalid"] = panel[
        [f"{name}_clean" for name in valuation_rules]
    ].isna().any(axis=1)

    quality_rules = {
        "F050204C": (-0.50, 0.50),       # ROA TTM
        "F050504C": (-1.00, 1.00),       # ROE TTM
        "F053301C": (-1.00, 1.50),       # gross margin TTM
        "F052901C": (0.00, 5.00),        # CFO / total profit; negatives are ambiguous
        "F011201A": (0.00, 1.50),        # liabilities / assets
    }
    for column, bounds in quality_rules.items():
        panel[f"{column}_clean"] = _valid_range(panel[column], *bounds)
    financial_age = (panel["month"] - panel["Annodt"]).dt.days
    panel["flag_financial_stale"] = financial_age.gt(370) | financial_age.lt(0)
    clean_quality = [f"{name}_clean" for name in quality_rules]
    panel.loc[panel["flag_financial_stale"], clean_quality] = np.nan
    panel["flag_financial_invalid"] = panel[clean_quality].isna().any(axis=1)

    panel = attach_security_states(panel)
    panel["flag_special_treatment"] = panel["special_state"].isin(["B", "C", "D", "S", "T", "X"])
    panel["flag_abnormal_listing"] = ~panel["listing_state"].isin(["A"])
    return panel


def cleaning_report(panel: pd.DataFrame) -> pd.DataFrame:
    flags = [column for column in panel if column.startswith("flag_")]
    rows = [{"rule": "input_rows", "affected_rows": len(panel), "affected_share": 1.0}]
    for column in flags:
        count = int(panel[column].fillna(False).sum())
        rows.append({"rule": column, "affected_rows": count, "affected_share": count / len(panel)})
    if "eligible" in panel:
        count = int(panel["eligible"].sum())
        rows.append({"rule": "final_eligible_rows", "affected_rows": count, "affected_share": count / len(panel)})
    return pd.DataFrame(rows)
