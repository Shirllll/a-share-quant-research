from __future__ import annotations

"""Independent tradable-return layer with suspension and one-price-limit rules."""

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_quant import OUT
from data_cleaning import attach_security_states
from quant_project import csv_members


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DAILY_COLUMNS = [
    "Stkcd",
    "Trddt",
    "Opnprc",
    "Hiprc",
    "Loprc",
    "Clsprc",
    "Dnvaltrd",
    "Dretwd",
    "Trdsta",
    "PreClosePrice",
    "LimitDown",
    "LimitUp",
    "LimitStatus",
]


def _trading_calendar() -> dict[pd.Period, list[pd.Timestamp]]:
    dates: list[pd.Timestamp] = []
    for path in sorted(DATA.glob("09_index_daily_*.zip")):
        with zipfile.ZipFile(path) as archive:
            for member in csv_members(archive, "IDX_Idxtrd"):
                with archive.open(member) as handle:
                    frame = pd.read_csv(
                        handle, usecols=["Indexcd", "Idxtrd01"], low_memory=False
                    )
                frame = frame[frame["Indexcd"].astype(str).str.zfill(6) == "000300"]
                dates.extend(pd.to_datetime(frame["Idxtrd01"], errors="coerce").dropna())
    unique = pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values()
    return {
        month: list(group.head(5))
        for month, group in unique.groupby(unique.dt.to_period("M"))
    }


def _one_price_limit(frame: pd.DataFrame, direction: int) -> pd.Series:
    prices_equal = (
        np.isclose(frame["Opnprc"], frame["Hiprc"], equal_nan=False)
        & np.isclose(frame["Opnprc"], frame["Loprc"], equal_nan=False)
        & np.isclose(frame["Opnprc"], frame["Clsprc"], equal_nan=False)
    )
    return prices_equal & pd.to_numeric(frame["LimitStatus"], errors="coerce").eq(direction)


def build_execution_state(
    start: str = "2018-01-01", end: str = "2025-12-31"
) -> pd.DataFrame:
    calendar = _trading_calendar()
    pieces: list[pd.DataFrame] = []
    for path in sorted(DATA.glob("02_daily_return_*.zip")):
        with zipfile.ZipFile(path) as archive:
            for member in csv_members(archive, "TRD_Dalyr"):
                with archive.open(member) as handle:
                    for frame in pd.read_csv(
                        handle,
                        usecols=DAILY_COLUMNS,
                        dtype={"Stkcd": "string"},
                        chunksize=300_000,
                        low_memory=False,
                    ):
                        frame["Trddt"] = pd.to_datetime(frame["Trddt"], errors="coerce")
                        frame = frame[frame["Trddt"].between(start, end)].copy()
                        if frame.empty:
                            continue
                        frame["stock_code"] = (
                            frame["Stkcd"]
                            .astype("string")
                            .str.replace(r"\.0$", "", regex=True)
                            .str.zfill(6)
                        )
                        frame["month_period"] = frame["Trddt"].dt.to_period("M")
                        frame["realization_month"] = (
                            frame["month_period"].dt.to_timestamp("M")
                        )
                        numeric_columns = [
                            column for column in DAILY_COLUMNS if column not in ("Stkcd", "Trddt")
                        ]
                        for column in numeric_columns:
                            frame[column] = pd.to_numeric(frame[column], errors="coerce")
                        frame["one_price_limit_up"] = _one_price_limit(frame, 1)
                        frame["one_price_limit_down"] = _one_price_limit(frame, -1)
                        frame["in_first_5_market_days"] = [
                            date in set(calendar.get(month, []))
                            for date, month in zip(
                                frame["Trddt"], frame["month_period"], strict=True
                            )
                        ]
                        frame["buyable_row"] = (
                            frame["in_first_5_market_days"]
                            & frame["Trdsta"].eq(1)
                            & ~frame["one_price_limit_up"]
                            & frame["Clsprc"].notna()
                        )
                        frame["sellable_row"] = (
                            frame["in_first_5_market_days"]
                            & frame["Trdsta"].eq(1)
                            & ~frame["one_price_limit_down"]
                            & frame["Clsprc"].notna()
                        )
                        first_day = frame["month_period"].map(
                            lambda month: calendar.get(month, [pd.NaT])[0]
                            if calendar.get(month)
                            else pd.NaT
                        )
                        fifth_day = frame["month_period"].map(
                            lambda month: calendar.get(month, [pd.NaT])[-1]
                            if calendar.get(month)
                            else pd.NaT
                        )
                        frame["return_after_1d"] = frame["Dretwd"].where(
                            frame["Trddt"] > first_day
                        )
                        frame["return_after_5d"] = frame["Dretwd"].where(
                            frame["Trddt"] > fifth_day
                        )
                        frame["_log_return"] = np.log1p(frame["Dretwd"])
                        frame["_log_return_after_1d"] = np.log1p(
                            frame["return_after_1d"]
                        )
                        frame["_log_return_after_5d"] = np.log1p(
                            frame["return_after_5d"]
                        )
                        frame["_limit_up_window"] = (
                            frame["in_first_5_market_days"]
                            & frame["one_price_limit_up"]
                        )
                        frame["_limit_down_window"] = (
                            frame["in_first_5_market_days"]
                            & frame["one_price_limit_down"]
                        )
                        frame = frame.sort_values(
                            ["stock_code", "realization_month", "Trddt"]
                        )
                        grouped = (
                            frame.groupby(
                                ["stock_code", "realization_month"], observed=True
                            )
                            .agg(
                                _log_return=("_log_return", "sum"),
                                _return_count=("Dretwd", "count"),
                                _log_return_after_1d=("_log_return_after_1d", "sum"),
                                _return_after_1d_count=("return_after_1d", "count"),
                                _log_return_after_5d=("_log_return_after_5d", "sum"),
                                _return_after_5d_count=("return_after_5d", "count"),
                                buyable_first_5d=("buyable_row", "any"),
                                sellable_first_5d=("sellable_row", "any"),
                                one_price_limit_up_first_5d=("_limit_up_window", "any"),
                                one_price_limit_down_first_5d=("_limit_down_window", "any"),
                                first_trade_date=("Trddt", "min"),
                                last_trade_date=("Trddt", "max"),
                                last_trade_status=("Trdsta", "last"),
                                last_close=("Clsprc", "last"),
                                last_preclose=("PreClosePrice", "last"),
                                last_limit_up=("LimitUp", "last"),
                                last_limit_down=("LimitDown", "last"),
                                trading_records=("Trddt", "count"),
                                average_daily_amount=("Dnvaltrd", "mean"),
                            )
                            .reset_index()
                        )
                        grouped["actual_month_return"] = np.expm1(
                            grouped.pop("_log_return")
                        ).where(grouped.pop("_return_count") > 0)
                        grouped["delayed_1d_return"] = np.expm1(
                            grouped.pop("_log_return_after_1d")
                        ).where(grouped.pop("_return_after_1d_count") > 0)
                        grouped["delayed_5d_return"] = np.expm1(
                            grouped.pop("_log_return_after_5d")
                        ).where(grouped.pop("_return_after_5d_count") > 0)
                        pieces.append(grouped)
    state = pd.concat(pieces, ignore_index=True)

    state["_log_return"] = np.log1p(state["actual_month_return"])
    state["_log_return_after_1d"] = np.log1p(state["delayed_1d_return"])
    state["_log_return_after_5d"] = np.log1p(state["delayed_5d_return"])
    state = state.sort_values(
        ["stock_code", "realization_month", "last_trade_date"]
    )
    state = (
        state.groupby(["stock_code", "realization_month"], observed=True)
        .agg(
            _log_return=("_log_return", "sum"),
            _return_count=("actual_month_return", "count"),
            _log_return_after_1d=("_log_return_after_1d", "sum"),
            _return_after_1d_count=("delayed_1d_return", "count"),
            _log_return_after_5d=("_log_return_after_5d", "sum"),
            _return_after_5d_count=("delayed_5d_return", "count"),
            buyable_first_5d=("buyable_first_5d", "any"),
            sellable_first_5d=("sellable_first_5d", "any"),
            one_price_limit_up_first_5d=("one_price_limit_up_first_5d", "any"),
            one_price_limit_down_first_5d=("one_price_limit_down_first_5d", "any"),
            first_trade_date=("first_trade_date", "min"),
            last_trade_date=("last_trade_date", "max"),
            last_trade_status=("last_trade_status", "last"),
            last_close=("last_close", "last"),
            last_preclose=("last_preclose", "last"),
            last_limit_up=("last_limit_up", "last"),
            last_limit_down=("last_limit_down", "last"),
            trading_records=("trading_records", "sum"),
            average_daily_amount=("average_daily_amount", "mean"),
        )
        .reset_index()
    )
    state["actual_month_return"] = np.expm1(state.pop("_log_return")).where(
        state.pop("_return_count") > 0
    )
    state["delayed_1d_return"] = np.expm1(
        state.pop("_log_return_after_1d")
    ).where(state.pop("_return_after_1d_count") > 0)
    state["delayed_5d_return"] = np.expm1(
        state.pop("_log_return_after_5d")
    ).where(state.pop("_return_after_5d_count") > 0)
    state["limit_execution_supported"] = True
    return state


def classify_execution_state(
    state: pd.Series | None,
    special_state: str = "A",
    listing_state: str = "A",
    previous_had_data: bool = True,
) -> str:
    def is_true(value: object) -> bool:
        return bool(pd.notna(value) and value is not False and bool(value))

    if state is None or state.empty:
        if listing_state != "A":
            return "unresolved_delisting"
        return "cross_month_suspension" if not previous_had_data else "temporary_suspension"
    if listing_state != "A":
        if pd.isna(state.get("actual_month_return", np.nan)):
            return f"unresolved_delisting_or_code_change_state_{listing_state}"
        return f"abnormal_listing_state_{listing_state}"
    if special_state != "A":
        return f"special_treatment_state_{special_state}"
    if not is_true(state.get("buyable_first_5d", False)) and not is_true(
        state.get("sellable_first_5d", False)
    ):
        return "suspended_or_no_executable_quote"
    if not previous_had_data:
        return "resumed_trading"
    if is_true(state.get("one_price_limit_up_first_5d", False)):
        return "one_price_limit_up"
    if is_true(state.get("one_price_limit_down_first_5d", False)):
        return "one_price_limit_down"
    return "normal_trading"


def execute_target_orders(
    previous: dict[str, float],
    desired: dict[str, float],
    state: pd.DataFrame,
    signal_month: pd.Timestamp,
    realization_month: pd.Timestamp,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    def is_true(value: object) -> bool:
        return bool(pd.notna(value) and value is not False and bool(value))

    indexed = state.set_index("stock_code") if not state.empty else pd.DataFrame()
    actual = previous.copy()
    records: list[dict[str, object]] = []
    cash = max(0.0, 1.0 - sum(actual.values()))
    names = sorted(set(previous) | set(desired))
    # Sells first: an untradeable old position remains in the actual portfolio.
    for name in names:
        old = float(actual.get(name, 0.0))
        target = float(desired.get(name, 0.0))
        if target >= old - 1e-12:
            continue
        row = indexed.loc[name] if name in indexed.index else pd.Series(dtype=float)
        allowed = is_true(row.get("sellable_first_5d", False))
        traded = old - target if allowed else 0.0
        if allowed:
            actual[name] = target
            cash += traded
        records.append(
            {
                "signal_month": signal_month,
                "realization_month": realization_month,
                "stock_code": name,
                "side": "sell",
                "old_weight": old,
                "desired_weight": target,
                "executed_weight": traded,
                "unfilled_weight": (old - target) - traded,
                "execution_status": "filled" if allowed else "unfilled_sell_carried",
                "one_price_limit_flag": is_true(
                    row.get("one_price_limit_down_first_5d", False)
                ),
                "unsupported_limit_execution": False,
            }
        )
    # Buys cannot use proceeds that were not actually realized.
    for name in names:
        old = float(actual.get(name, 0.0))
        target = float(desired.get(name, 0.0))
        if target <= old + 1e-12:
            continue
        row = indexed.loc[name] if name in indexed.index else pd.Series(dtype=float)
        allowed = is_true(row.get("buyable_first_5d", False))
        proposed = target - old
        traded = min(proposed, cash) if allowed else 0.0
        if traded > 0:
            actual[name] = old + traded
            cash -= traded
        records.append(
            {
                "signal_month": signal_month,
                "realization_month": realization_month,
                "stock_code": name,
                "side": "buy",
                "old_weight": old,
                "desired_weight": target,
                "executed_weight": traded,
                "unfilled_weight": proposed - traded,
                "execution_status": (
                    "filled"
                    if allowed and traded >= proposed - 1e-12
                    else ("partially_filled_cash_limit" if traded > 0 else "unfilled_buy")
                ),
                "one_price_limit_flag": is_true(
                    row.get("one_price_limit_up_first_5d", False)
                ),
                "unsupported_limit_execution": False,
            }
        )
    return {name: weight for name, weight in actual.items() if weight > 1e-12}, records


def portfolio_return_without_imputation(
    weights: dict[str, float], returns: pd.Series
) -> tuple[float, float]:
    known = 0.0
    unresolved = 0.0
    for name, weight in weights.items():
        value = returns.get(name, np.nan)
        if pd.isna(value):
            unresolved += weight
        else:
            known += weight * float(value)
    return (np.nan if unresolved > 1e-12 else float(known)), float(unresolved)


def run_engine(
    holdings: pd.DataFrame, execution_state: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    execution_state = execution_state.copy()
    execution_state["stock_code"] = (
        execution_state["stock_code"].astype(str).str.zfill(6)
    )
    execution_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    previous_actual: dict[str, float] = {}
    previous_state_names: set[str] = set()
    for signal_month, target_frame in holdings.groupby("signal_month", sort=True):
        realization_month = pd.Timestamp(target_frame["realization_month"].iloc[0])
        target_frame = target_frame.copy()
        target_frame["stock_code"] = (
            target_frame["stock_code"].astype(str).str.zfill(6)
        )
        desired = target_frame.set_index("stock_code")["target_weight"].to_dict()
        month_state = execution_state[
            execution_state["realization_month"] == realization_month
        ].copy()
        needed = pd.DataFrame(
            {
                "Stkcd": list(set(previous_actual) | set(desired)),
                "month": realization_month,
            }
        )
        needed["Stkcd"] = needed["Stkcd"].astype("string")
        security_state = attach_security_states(needed).rename(
            columns={"Stkcd": "stock_code"}
        )
        security_state["stock_code"] = (
            security_state["stock_code"].astype(str).str.zfill(6)
        )
        month_state = security_state.merge(
            month_state, on=["stock_code"], how="left"
        )
        actual, orders = execute_target_orders(
            previous_actual,
            desired,
            month_state,
            pd.Timestamp(signal_month),
            realization_month,
        )
        state_index = month_state.set_index("stock_code")
        for order in orders:
            name = order["stock_code"]
            row = state_index.loc[name] if name in state_index.index else pd.Series(dtype=float)
            order.update(
                {
                    "state_classification": classify_execution_state(
                        row,
                        str(row.get("special_state", "A")),
                        str(row.get("listing_state", "A")),
                        name in previous_state_names,
                    ),
                    "special_state": row.get("special_state", np.nan),
                    "listing_state": row.get("listing_state", np.nan),
                    "trading_status_code": row.get("last_trade_status", np.nan),
                    "actual_month_return": row.get("actual_month_return", np.nan),
                }
            )
            execution_rows.append(order)

        actual_returns = state_index["actual_month_return"]
        tradable_gross, unresolved_weight = portfolio_return_without_imputation(
            actual, actual_returns
        )
        theoretical_returns = target_frame.set_index("stock_code")["forward_return"]
        theoretical_return, theoretical_unresolved = portfolio_return_without_imputation(
            desired, theoretical_returns
        )
        traded_weight = sum(float(row["executed_weight"]) for row in orders)
        tradable_net = (
            tradable_gross - traded_weight * 20 / 10_000
            if pd.notna(tradable_gross)
            else np.nan
        )
        monthly_rows.append(
            {
                "signal_month": signal_month,
                "realization_month": realization_month,
                "theoretical_return": theoretical_return,
                "tradable_gross_return": tradable_gross,
                "tradable_return": tradable_net,
                "theoretical_unresolved_return_weight": theoretical_unresolved,
                "unresolved_return_weight": unresolved_weight,
                "executed_turnover": 0.5 * traded_weight,
                "unfilled_order_weight": sum(
                    float(row["unfilled_weight"]) for row in orders
                ),
                "actual_invested_weight": sum(actual.values()),
                "unsupported_limit_execution": False,
            }
        )
        for name, weight in actual.items():
            value = actual_returns.get(name, np.nan)
            if pd.isna(value):
                row = state_index.loc[name] if name in state_index.index else pd.Series(dtype=float)
                unresolved_rows.append(
                    {
                        "signal_month": signal_month,
                        "realization_month": realization_month,
                        "stock_code": name,
                        "weight": weight,
                        "reason": classify_execution_state(
                            row,
                            str(row.get("special_state", "A")),
                            str(row.get("listing_state", "A")),
                            name in previous_state_names,
                        ),
                        "unresolved_delisting": bool(
                            str(row.get("listing_state", "A")) != "A"
                        ),
                        "return_was_zero_imputed": False,
                    }
                )
        previous_actual = actual
        previous_state_names = set(
            month_state.loc[
                month_state["actual_month_return"].notna(), "stock_code"
            ]
        )
    return (
        pd.DataFrame(execution_rows),
        pd.DataFrame(monthly_rows),
        pd.DataFrame(unresolved_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdings", type=Path, default=OUT / "cost_aware_holdings.csv")
    parser.add_argument("--reuse-state-cache", action="store_true")
    args = parser.parse_args()
    holdings = pd.read_csv(
        args.holdings, parse_dates=["signal_month", "realization_month"]
    )
    cache = OUT / "execution_state_monthly.csv.gz"
    if args.reuse_state_cache and cache.exists():
        state = pd.read_csv(
            cache,
            parse_dates=[
                "realization_month",
                "first_trade_date",
                "last_trade_date",
            ],
        )
    else:
        state = build_execution_state()
        state.to_csv(cache, index=False, compression="gzip")
    execution, monthly, unresolved = run_engine(holdings, state)
    execution.to_csv(OUT / "tradable_execution_log.csv", index=False)
    monthly.to_csv(OUT / "tradable_return_monthly.csv", index=False)
    unresolved.to_csv(OUT / "unresolved_returns.csv", index=False)

    trade_path = OUT / "cost_aware_trade_log.csv"
    if trade_path.exists() and not execution.empty:
        trades = pd.read_csv(
            trade_path, parse_dates=["signal_month", "realization_month"]
        )
        status = (
            execution.sort_values("executed_weight")
            .drop_duplicates(
                ["signal_month", "realization_month", "stock_code"], keep="last"
            )[
                [
                    "signal_month",
                    "realization_month",
                    "stock_code",
                    "execution_status",
                ]
            ]
        )
        trades["stock_code"] = trades["stock_code"].astype(str).str.zfill(6)
        trades = trades.drop(columns=["execution_status"], errors="ignore").merge(
            status,
            on=["signal_month", "realization_month", "stock_code"],
            how="left",
        )
        trades["execution_status"] = trades["execution_status"].fillna(
            "no_order_required_or_carried"
        )
        trades.to_csv(trade_path, index=False)
    flagged = monthly[monthly["unresolved_return_weight"] > 0.005]
    print(monthly.to_string(index=False))
    if not flagged.empty:
        print("Months with unresolved return weight above 0.5%:")
        print(flagged.to_string(index=False))


if __name__ == "__main__":
    main()
