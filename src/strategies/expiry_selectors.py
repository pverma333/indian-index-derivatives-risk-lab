from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ALLOWED_TENORS = {"WEEKLY", "MONTHLY", "BOTH"}

# We now require opt_*_expiry to map the roll entry_date -> tradable expiry.
_REQUIRED_COLUMNS = {
    "date",
    "symbol",
    "expiry_dt",
    "is_trading_day",
    "is_opt_weekly_expiry",
    "is_opt_monthly_expiry",
    "opt_weekly_expiry",
    "opt_monthly_expiry",
}


def build_expiry_cycles(market_df: pd.DataFrame, symbol: str, tenor: str) -> pd.DataFrame:
    """
    Build deterministic trade cycles using dataset flags (Phase 2).

    Key semantics (fix vs old behavior):
      - Use expiry flags to identify "anchor" expiries (the just-finished series).
      - entry_date is the next trading day strictly after the anchor expiry_dt.
      - The *tradable* expiry_dt for entry_date is taken from:
          WEEKLY  -> opt_weekly_expiry on entry_date
          MONTHLY -> opt_monthly_expiry on entry_date
      - exit_date == expiry_dt (Phase 2 exit at expiry).
      - Drop cycles where entry_date is missing or opt_*_expiry is missing; log reason with context.

    This resolves the real issue you observed: previously, entry_date was after expiry_dt while still
    trying to trade the expired contract, producing empty chains.

    Returns
    -------
    cycles_df columns:
        symbol, tenor, expiry_dt, entry_date, exit_date
    """
    _validate_inputs(market_df=market_df, symbol=symbol, tenor=tenor)

    if market_df.empty:
        return _empty_cycles_df()

    df = market_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["expiry_dt"] = pd.to_datetime(df["expiry_dt"], errors="coerce").dt.normalize()
    df["opt_weekly_expiry"] = pd.to_datetime(df["opt_weekly_expiry"], errors="coerce").dt.normalize()
    df["opt_monthly_expiry"] = pd.to_datetime(df["opt_monthly_expiry"], errors="coerce").dt.normalize()

    df_sym = df.loc[df["symbol"] == symbol].copy()
    if df_sym.empty:
        return _empty_cycles_df()

    trading_dates = _get_sorted_trading_dates(df_sym)
    tenors_to_build = _expand_tenor(tenor)

    rows: List[dict] = []

    for tnr in tenors_to_build:
        anchor_expiries = _get_unique_anchor_expiry_dates(df_sym, tnr)
        if len(anchor_expiries) == 0:
            continue

        expiry_ref_col = "opt_weekly_expiry" if tnr == "WEEKLY" else "opt_monthly_expiry"

        for anchor_expiry_dt in anchor_expiries:
            entry_dt = _next_trading_day_after(trading_dates, anchor_expiry_dt)
            if pd.isna(entry_dt):
                logger.warning(
                    "MISSING_ENTRY_DATE drop cycle | symbol=%s tenor=%s anchor_expiry_dt=%s",
                    symbol,
                    tnr,
                    _fmt_date(anchor_expiry_dt),
                )
                continue

            trade_expiry_dt = _trade_expiry_for_entry_date(df_sym, entry_dt, expiry_ref_col)
            if pd.isna(trade_expiry_dt):
                logger.warning(
                    "MISSING_TRADE_EXPIRY drop cycle | symbol=%s tenor=%s entry_date=%s ref_col=%s",
                    symbol,
                    tnr,
                    _fmt_date(entry_dt),
                    expiry_ref_col,
                )
                continue

            if entry_dt > trade_expiry_dt:
                logger.warning(
                    "ENTRY_AFTER_TRADE_EXPIRY drop cycle | symbol=%s tenor=%s entry_date=%s trade_expiry_dt=%s",
                    symbol,
                    tnr,
                    _fmt_date(entry_dt),
                    _fmt_date(trade_expiry_dt),
                )
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "tenor": tnr,
                    "expiry_dt": trade_expiry_dt,
                    "entry_date": entry_dt,
                    "exit_date": trade_expiry_dt,
                }
            )

    if not rows:
        return _empty_cycles_df()

    cycles_df = pd.DataFrame(rows, columns=["symbol", "tenor", "expiry_dt", "entry_date", "exit_date"])
    cycles_df = cycles_df.drop_duplicates()
    cycles_df = cycles_df.sort_values(["expiry_dt", "tenor"], kind="mergesort").reset_index(drop=True)

    if cycles_df["entry_date"].isna().any():
        raise ValueError("Invariant violated: cycles_df contains null entry_date after drop logic.")
    if cycles_df["expiry_dt"].isna().any():
        raise ValueError("Invariant violated: cycles_df contains null expiry_dt after drop logic.")
    if not (cycles_df["exit_date"] == cycles_df["expiry_dt"]).all():
        raise ValueError("Invariant violated: exit_date must equal expiry_dt for Phase 2.")

    return cycles_df


def _validate_inputs(market_df: pd.DataFrame, symbol: str, tenor: str) -> None:
    if not isinstance(market_df, pd.DataFrame):
        raise TypeError("market_df must be a pandas DataFrame.")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string.")
    if tenor not in _ALLOWED_TENORS:
        raise ValueError(f"tenor must be one of {_ALLOWED_TENORS}, got: {tenor!r}")

    missing = sorted(_REQUIRED_COLUMNS - set(market_df.columns))
    if missing:
        raise ValueError(f"market_df missing required columns: {missing}")


def _expand_tenor(tenor: str) -> Sequence[str]:
    if tenor == "BOTH":
        return ("WEEKLY", "MONTHLY")
    return (tenor,)


def _get_unique_anchor_expiry_dates(df_sym: pd.DataFrame, tenor: str) -> np.ndarray:
    if tenor == "WEEKLY":
        flag_col = "is_opt_weekly_expiry"
    elif tenor == "MONTHLY":
        flag_col = "is_opt_monthly_expiry"
    else:
        raise ValueError(f"Unexpected tenor: {tenor}")

    expiry_series = df_sym.loc[df_sym[flag_col] == True, "expiry_dt"]  # noqa: E712
    expiry_series = expiry_series.dropna().drop_duplicates()
    expiry_dates = pd.to_datetime(expiry_series, errors="coerce").dropna().dt.normalize().unique()
    return np.sort(expiry_dates)


def _get_sorted_trading_dates(df_sym: pd.DataFrame) -> np.ndarray:
    dates = df_sym.loc[df_sym["is_trading_day"] == True, "date"]  # noqa: E712
    dates = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().unique()
    return np.sort(dates)


def _next_trading_day_after(sorted_trading_dates: np.ndarray, expiry_dt: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(expiry_dt) or sorted_trading_dates.size == 0:
        return pd.NaT

    exp64 = np.datetime64(pd.Timestamp(expiry_dt).normalize())
    idx = np.searchsorted(sorted_trading_dates, exp64, side="right")
    if idx >= sorted_trading_dates.size:
        return pd.NaT
    return pd.Timestamp(sorted_trading_dates[idx]).normalize()


def _trade_expiry_for_entry_date(df_sym: pd.DataFrame, entry_dt: pd.Timestamp, col: str) -> pd.Timestamp:
    s = df_sym.loc[df_sym["date"] == entry_dt, col]
    if s.empty:
        return pd.NaT

    vals = pd.to_datetime(s.dropna().unique(), errors="coerce")
    vals = [pd.Timestamp(v).normalize() for v in vals if not pd.isna(v)]
    if not vals:
        return pd.NaT

    return sorted(vals)[0]


def _empty_cycles_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "tenor", "expiry_dt", "entry_date", "exit_date"])


def _fmt_date(x: object) -> str:
    try:
        ts = pd.Timestamp(x)
        if pd.isna(ts):
            return "NaT"
        return ts.date().isoformat()
    except Exception:
        return str(x)
