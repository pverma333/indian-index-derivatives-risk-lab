from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

ALLOWED_SYMBOLS = {"NIFTY", "BANKNIFTY"}
FUT_INSTRUMENT = "FUTIDX"
OPT_INSTRUMENT = "OPTIDX"


@dataclass(frozen=True)
class TradeCalendarPaths:
    input_csv: Path
    output_parquet: Path
    output_csv: Path


def _configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")


def _parse_datetime_series(s: pd.Series, col_name: str) -> pd.Series:
    """
    Robust datetime parsing. Not inferring weekly/monthly using DTE,
    only parse for comparisons and month grouping.
    """
    parsed = pd.to_datetime(s, errors="coerce", utc=False)
    n_bad = int(parsed.isna().sum())
    if n_bad > 0:
        LOGGER.warning("Datetime parse failures in %s: %d", col_name, n_bad)
    return parsed


def _normalize_trade_date(ts: pd.Series) -> pd.Series:
    # Normalize to midnight (datetime64[ns]) to match required output type
    return ts.dt.normalize()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _next_k_expiries(sorted_expiries: pd.DatetimeIndex, trade_date: pd.Timestamp, k: int) -> list[pd.Timestamp]:
    """
    Return next k expiries >= trade_date from a sorted DatetimeIndex.
    If insufficient, pads with pd.NaT.
    """
    if len(sorted_expiries) == 0 or pd.isna(trade_date):
        return [pd.NaT] * k

    idx = int(sorted_expiries.searchsorted(trade_date, side="left"))
    out: list[pd.Timestamp] = []
    for j in range(k):
        pos = idx + j
        out.append(sorted_expiries[pos] if pos < len(sorted_expiries) else pd.NaT)
    return out


def _next_expiry(sorted_expiries: pd.DatetimeIndex, trade_date: pd.Timestamp) -> pd.Timestamp:
    if len(sorted_expiries) == 0 or pd.isna(trade_date):
        return pd.NaT
    idx = int(sorted_expiries.searchsorted(trade_date, side="left"))
    return sorted_expiries[idx] if idx < len(sorted_expiries) else pd.NaT


def _build_futures_expiry_index(df: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    fut = df.loc[df["INSTRUMENT"].eq(FUT_INSTRUMENT), ["SYMBOL", "EXPIRY_DT"]].copy()
    fut = fut.dropna(subset=["SYMBOL", "EXPIRY_DT"])
    out: dict[str, pd.DatetimeIndex] = {}
    for sym, g in fut.groupby("SYMBOL", sort=False):
        expiries = pd.DatetimeIndex(sorted(g["EXPIRY_DT"].dropna().unique()))
        out[str(sym)] = expiries
        LOGGER.info("Futures expiries | %s | count=%d", sym, len(expiries))
    return out


def _build_option_expiry_indexes(df: pd.DataFrame) -> tuple[dict[str, pd.DatetimeIndex], dict[str, pd.DatetimeIndex]]:
    """
    Monthly expiry rule:
      For each (Symbol, ExpiryMonth), Monthly expiry = max(EXPIRY_DT) in that month.
      All other expiries in that month are Weekly.
    """
    opt = df.loc[df["INSTRUMENT"].eq(OPT_INSTRUMENT), ["SYMBOL", "EXPIRY_DT"]].copy()
    opt = opt.dropna(subset=["SYMBOL", "EXPIRY_DT"])

    weekly_idx: dict[str, pd.DatetimeIndex] = {}
    monthly_idx: dict[str, pd.DatetimeIndex] = {}

    for sym, g in opt.groupby("SYMBOL", sort=False):
        g = g.copy()
        # group by month (period) to apply max-of-month rule
        g["ExpiryMonth"] = g["EXPIRY_DT"].dt.to_period("M")
        month_max = g.groupby("ExpiryMonth")["EXPIRY_DT"].max()

        monthly_set = set(pd.to_datetime(month_max.values))
        all_expiries = pd.to_datetime(g["EXPIRY_DT"].unique())

        monthly_expiries = pd.DatetimeIndex(sorted(pd.Series(list(monthly_set)).unique()))
        weekly_expiries = pd.DatetimeIndex(sorted(pd.Series([d for d in all_expiries if d not in monthly_set]).unique()))

        weekly_idx[str(sym)] = weekly_expiries
        monthly_idx[str(sym)] = monthly_expiries

        LOGGER.info(
            "Options expiries | %s | weekly=%d monthly=%d",
            sym,
            len(weekly_expiries),
            len(monthly_expiries),
        )

    return weekly_idx, monthly_idx


def build_trade_calendar(
    input_csv: str | Path = Path("data/processed/Nifty_Historical_Derivatives.csv"),
    output_parquet: str | Path = Path("data/processed/trade_calendar.parquet"),
    output_csv: str | Path = Path("data/raw/trade_calendar.csv"),
    *,
    log_level: int = logging.INFO,
) -> pd.DataFrame:
    """
    Build a deterministic trade calendar from the derivatives dataset.

    Returns the calendar DataFrame (also persisted to disk).
    """
    _configure_logging(log_level)

    paths = TradeCalendarPaths(
        input_csv=Path(input_csv),
        output_parquet=Path(output_parquet),
        output_csv=Path(output_csv),
    )

    LOGGER.info("Reading input: %s", paths.input_csv)
    df = pd.read_csv(paths.input_csv)
    LOGGER.info("Input rows=%d cols=%d", len(df), df.shape[1])

    _require_columns(df, ["INSTRUMENT", "SYMBOL", "EXPIRY_DT", "TIMESTAMP"])

    # Filter scope symbols early
    df["SYMBOL"] = df["SYMBOL"].astype(str)
    df = df.loc[df["SYMBOL"].isin(ALLOWED_SYMBOLS)].copy()
    LOGGER.info("After symbol filter (%s): rows=%d", sorted(ALLOWED_SYMBOLS), len(df))

    # Parse dates
    df["TIMESTAMP"] = _parse_datetime_series(df["TIMESTAMP"], "TIMESTAMP")
    df["EXPIRY_DT"] = _parse_datetime_series(df["EXPIRY_DT"], "EXPIRY_DT")
    df["TradeDate"] = _normalize_trade_date(df["TIMESTAMP"])

    # Fail-fast if TradeDate is largely missing (critical field)
    trade_date_missing = float(df["TradeDate"].isna().mean())
    if trade_date_missing > 0.01:
        raise ValueError(f"TradeDate missing rate too high: {trade_date_missing:.2%}")

    # We only need rows with a TradeDate; keep logging explicit
    n_before = len(df)
    df = df.dropna(subset=["TradeDate"])
    LOGGER.info("Dropped rows with NaT TradeDate: %d -> %d", n_before, len(df))

    # Build grain (one row per unique (TradeDate, Symbol) present in input)
    grain = (
        df[["TradeDate", "SYMBOL"]]
        .drop_duplicates()
        .rename(columns={"SYMBOL": "Symbol"})
        .sort_values(["Symbol", "TradeDate"], kind="mergesort")
        .reset_index(drop=True)
    )
    LOGGER.info("Calendar grain rows=%d", len(grain))

    # Build expiry indexes
    fut_idx = _build_futures_expiry_index(df.rename(columns={"SYMBOL": "Symbol"}).rename(columns={"Symbol": "SYMBOL"}))
    # Above rename is a no-op; keep df column names consistent for helpers:
    fut_idx = _build_futures_expiry_index(df)

    weekly_idx, monthly_idx = _build_option_expiry_indexes(df)

    # Resolve expiries per row deterministically
    fut_near: list[pd.Timestamp] = []
    fut_next: list[pd.Timestamp] = []
    fut_far: list[pd.Timestamp] = []
    opt_weekly: list[pd.Timestamp] = []
    opt_monthly: list[pd.Timestamp] = []

    for row in grain.itertuples(index=False):
        trade_date: pd.Timestamp = row.TradeDate
        sym: str = row.Symbol

        fexp = fut_idx.get(sym, pd.DatetimeIndex([]))
        near, nxt, far = _next_k_expiries(fexp, trade_date, k=3)
        fut_near.append(near)
        fut_next.append(nxt)
        fut_far.append(far)

        wexp = weekly_idx.get(sym, pd.DatetimeIndex([]))
        mexp = monthly_idx.get(sym, pd.DatetimeIndex([]))
        opt_weekly.append(_next_expiry(wexp, trade_date))
        opt_monthly.append(_next_expiry(mexp, trade_date))

    out = grain.copy()
    out["Fut_Near_Expiry"] = pd.to_datetime(fut_near)
    out["Fut_Next_Expiry"] = pd.to_datetime(fut_next)
    out["Fut_Far_Expiry"] = pd.to_datetime(fut_far)
    out["Opt_Weekly_Expiry"] = pd.to_datetime(opt_weekly)
    out["Opt_Monthly_Expiry"] = pd.to_datetime(opt_monthly)

    # --- Validations / Assertions ---
    # 1) Futures ordering when all exist
    mask_all_fut = out[["Fut_Near_Expiry", "Fut_Next_Expiry", "Fut_Far_Expiry"]].notna().all(axis=1)
    bad_order = out.loc[
        mask_all_fut
        & ~(
            (out["Fut_Near_Expiry"] <= out["Fut_Next_Expiry"])
            & (out["Fut_Next_Expiry"] <= out["Fut_Far_Expiry"])
        )
    ]
    if len(bad_order) > 0:
        raise AssertionError(f"Futures expiry ordering violated on {len(bad_order)} rows.")

    # 2) Opt_Monthly_Expiry always belongs to monthly set
    monthly_sets: dict[str, set[pd.Timestamp]] = {
        sym: set(pd.to_datetime(idx).to_pydatetime() for idx in m_idx)
        for sym, m_idx in monthly_idx.items()
    }
    # Compare with timestamps; normalize to python datetimes for set membership robustness
    def _is_in_monthly_set(sym: str, dt: Optional[pd.Timestamp]) -> bool:
        if dt is pd.NaT or dt is None or pd.isna(dt):
            return True
        return dt.to_pydatetime() in monthly_sets.get(sym, set())

    bad_monthly = out.loc[
        ~out.apply(lambda r: _is_in_monthly_set(r["Symbol"], r["Opt_Monthly_Expiry"]), axis=1)
    ]
    if len(bad_monthly) > 0:
        raise AssertionError(f"Opt_Monthly_Expiry not in monthly-expiry set on {len(bad_monthly)} rows.")

    # Ensure ordering (already sorted) and dtypes are datetime-like
    out = out.sort_values(["Symbol", "TradeDate"], kind="mergesort").reset_index(drop=True)

    LOGGER.info("Writing parquet: %s", paths.output_parquet)
    _ensure_parent_dir(paths.output_parquet)
    out.to_parquet(paths.output_parquet, index=False)

    LOGGER.info("Writing csv: %s", paths.output_csv)
    _ensure_parent_dir(paths.output_csv)
    out.to_csv(paths.output_csv, index=False)

    LOGGER.info("Done. Output rows=%d", len(out))
    return out


if __name__ == "__main__":
    build_trade_calendar()
