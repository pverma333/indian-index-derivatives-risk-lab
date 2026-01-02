from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ALLOWED_TENORS = {"WEEKLY", "MONTHLY", "BOTH"}

_REQUIRED_COLUMNS = {
    "date",
    "symbol",
    "expiry_dt",
    "is_trading_day",
    "is_opt_weekly_expiry",
    "is_opt_monthly_expiry",
}


def build_expiry_cycles(market_df: pd.DataFrame, symbol: str, tenor: str) -> pd.DataFrame:
    """
    Build deterministic expiry cycles using dataset flags only.

    Rules (Phase 2):
      - MONTHLY expiry_dt: unique expiry_dt where is_opt_monthly_expiry == True for symbol
      - WEEKLY expiry_dt: unique expiry_dt where is_opt_weekly_expiry == True for symbol
      - entry_date: next trading day strictly after expiry_dt (min date > expiry_dt with is_trading_day == True)
      - exit_date: expiry_dt
      - drop cycles where entry_date missing; log reason MISSING_ENTRY_DATE with symbol/tenor/expiry_dt
      - sort by expiry_dt then tenor

    Returns
    -------
    cycles_df: pd.DataFrame with columns:
        symbol, tenor, expiry_dt, entry_date, exit_date
    """
    _validate_inputs(market_df=market_df, symbol=symbol, tenor=tenor)

    if market_df.empty:
        return _empty_cycles_df()

    df = market_df.copy()

    # Normalize date columns deterministically (safe even if already datetime64)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["expiry_dt"] = pd.to_datetime(df["expiry_dt"], errors="coerce").dt.normalize()

    df_sym = df.loc[df["symbol"] == symbol].copy()
    if df_sym.empty:
        return _empty_cycles_df()

    trading_dates = _get_sorted_trading_dates(df_sym)

    tenors_to_build = _expand_tenor(tenor)
    rows: List[dict] = []

    for tnr in tenors_to_build:
        expiry_dates = _get_unique_expiry_dates_by_tenor(df_sym, tnr)
        if len(expiry_dates) == 0:
            continue

        for exp_dt in expiry_dates:
            entry_dt = _next_trading_day_after(trading_dates, exp_dt)
            if pd.isna(entry_dt):
                logger.warning(
                    "MISSING_ENTRY_DATE drop cycle | symbol=%s tenor=%s expiry_dt=%s",
                    symbol,
                    tnr,
                    _fmt_date(exp_dt),
                )
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "tenor": tnr,
                    "expiry_dt": exp_dt,
                    "entry_date": entry_dt,
                    "exit_date": exp_dt,
                }
            )

    if not rows:
        return _empty_cycles_df()

    cycles_df = pd.DataFrame(rows, columns=["symbol", "tenor", "expiry_dt", "entry_date", "exit_date"])
    cycles_df = cycles_df.sort_values(["expiry_dt", "tenor"], kind="mergesort").reset_index(drop=True)

    # Safety: ensure no null entry_date
    if cycles_df["entry_date"].isna().any():
        raise ValueError("Invariant violated: cycles_df contains null entry_date after drop logic.")

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


def _get_unique_expiry_dates_by_tenor(df_sym: pd.DataFrame, tenor: str) -> np.ndarray:
    if tenor == "WEEKLY":
        flag_col = "is_opt_weekly_expiry"
    elif tenor == "MONTHLY":
        flag_col = "is_opt_monthly_expiry"
    else:
        raise ValueError(f"Unexpected tenor: {tenor}")

    expiry_series = df_sym.loc[df_sym[flag_col] == True, "expiry_dt"]  # noqa: E712
    expiry_series = expiry_series.dropna().drop_duplicates()
    expiry_dates = pd.to_datetime(expiry_series, errors="coerce").dropna().dt.normalize().unique()
    expiry_dates = np.sort(expiry_dates)
    return expiry_dates


def _get_sorted_trading_dates(df_sym: pd.DataFrame) -> np.ndarray:
    dates = df_sym.loc[df_sym["is_trading_day"] == True, "date"]  # noqa: E712
    dates = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().unique()
    return np.sort(dates)


def _next_trading_day_after(sorted_trading_dates: np.ndarray, expiry_dt: pd.Timestamp) -> pd.Timestamp:
    """
    Given sorted unique trading dates (datetime64[D]/Timestamp normalized),
    return the smallest trading date strictly > expiry_dt; NaT if none.
    """
    if pd.isna(expiry_dt) or sorted_trading_dates.size == 0:
        return pd.NaT

    exp64 = np.datetime64(pd.Timestamp(expiry_dt).normalize())
    idx = np.searchsorted(sorted_trading_dates, exp64, side="right")
    if idx >= sorted_trading_dates.size:
        return pd.NaT
    return pd.Timestamp(sorted_trading_dates[idx]).normalize()


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
