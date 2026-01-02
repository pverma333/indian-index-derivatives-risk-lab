from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_LIQUIDITY_MODES = {"OFF", "ABSOLUTE", "PERCENTILE"}


def get_chain(
    market_df: pd.DataFrame,
    symbol: str,
    expiry_dt: pd.Timestamp,
    entry_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Extract the option chain slice needed by strategies for a single entry date and expiry.

    Filters:
      - symbol
      - instrument == "OPTIDX"
      - expiry_dt
      - date == entry_date

    Edge behavior:
      - If empty: returns empty DataFrame (skip-safe)
    """
    required = {"date", "symbol", "instrument", "expiry_dt"}
    _require_columns(market_df, required, ctx="get_chain")

    # Normalize to Timestamp for robust comparisons
    expiry_dt = pd.Timestamp(expiry_dt)
    entry_date = pd.Timestamp(entry_date)

    mask = (
        (market_df["symbol"] == symbol)
        & (market_df["instrument"] == "OPTIDX")
        & (pd.to_datetime(market_df["expiry_dt"]) == expiry_dt)
        & (pd.to_datetime(market_df["date"]) == entry_date)
    )
    chain_df = market_df.loc[mask].copy()
    return chain_df


def infer_strike_interval(chain_df: pd.DataFrame) -> Optional[int]:
    """
    Infer strike spacing as the mode of diffs between sorted unique strikes.

    Returns None if insufficient strikes.
    """
    if chain_df is None or chain_df.empty:
        return None

    _require_columns(chain_df, {"strike_pr"}, ctx="infer_strike_interval")

    strikes = (
        pd.to_numeric(chain_df["strike_pr"], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
    strikes = sorted(strikes)
    if len(strikes) < 2:
        return None

    diffs = np.diff(np.array(strikes, dtype=float))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None

    # mode with deterministic tie-break: choose smallest mode value
    s = pd.Series(diffs)
    modes = s.mode(dropna=True)
    if modes.empty:
        return None

    mode_val = float(modes.min())
    # Return int (ticket signature). Most strikes are integer-ish; round safely.
    rounded = int(round(mode_val))
    if not np.isfinite(mode_val):
        return None
    if abs(mode_val - rounded) > 1e-6:
        logger.warning(
            "infer_strike_interval: non-integer mode spacing %.6f; rounding to %d",
            mode_val,
            rounded,
        )
    return rounded


def apply_strike_band(
    chain_df: pd.DataFrame,
    spot_close: float,
    strike_band_n: int,
) -> pd.DataFrame:
    """
    Select strikes within ± strike_band_n around ATM by *strike count* (not points).
    Preserves both CE and PE rows for retained strikes.

    Determinism:
      - ATM tie-break: smaller strike wins when abs distance ties
      - Output sorted by (strike_pr asc, option_typ asc) for stable downstream behavior
    """
    if chain_df is None or chain_df.empty:
        return chain_df.copy() if chain_df is not None else pd.DataFrame()

    _require_columns(chain_df, {"strike_pr", "option_typ"}, ctx="apply_strike_band")

    if strike_band_n < 0:
        raise ValueError("strike_band_n must be >= 0")

    strikes = (
        pd.to_numeric(chain_df["strike_pr"], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
    strikes = sorted(strikes)
    if len(strikes) == 0:
        return chain_df.iloc[0:0].copy()

    spot_close_f = float(spot_close)

    # Pick ATM with deterministic tie-breaks
    strike_dists = [(abs(float(k) - spot_close_f), float(k)) for k in strikes]
    strike_dists.sort(key=lambda x: (x[0], x[1]))
    atm_strike = strike_dists[0][1]

    # Band by strike count
    idx = strikes.index(atm_strike)
    lo = max(0, idx - strike_band_n)
    hi = min(len(strikes) - 1, idx + strike_band_n)
    band_strikes = set(float(s) for s in strikes[lo : hi + 1])

    band_df = chain_df.loc[
        pd.to_numeric(chain_df["strike_pr"], errors="coerce").astype(float).isin(band_strikes)
    ].copy()

    # Stable ordering
    band_df["strike_pr"] = pd.to_numeric(band_df["strike_pr"], errors="coerce")
    band_df = band_df.sort_values(["strike_pr", "option_typ"], kind="mergesort").reset_index(drop=True)
    return band_df


def apply_liquidity_filters(
    band_df: pd.DataFrame,
    liquidity_mode: str,
    min_contracts: float,
    min_open_int: float,
    liquidity_percentile: float,
) -> pd.DataFrame:
    """
    Apply liquidity filters in 3 modes: OFF, ABSOLUTE, PERCENTILE.

    Strategy-safety note:
      - This function may remove CE/PE rows independently. Downstream must verify both legs exist
        for selected strikes (select_atm_strike enforces this).
    """
    if band_df is None or band_df.empty:
        return band_df.copy() if band_df is not None else pd.DataFrame()

    mode = str(liquidity_mode).upper().strip()
    if mode not in _LIQUIDITY_MODES:
        raise ValueError(f"liquidity_mode must be one of {sorted(_LIQUIDITY_MODES)}; got {liquidity_mode!r}")

    if mode == "OFF":
        return band_df.copy()

    required = {"contracts", "open_int", "settle_pr"}
    _require_columns(band_df, required, ctx="apply_liquidity_filters")

    df = band_df.copy()
    df["contracts"] = pd.to_numeric(df["contracts"], errors="coerce")
    df["open_int"] = pd.to_numeric(df["open_int"], errors="coerce")
    df["settle_pr"] = pd.to_numeric(df["settle_pr"], errors="coerce")

    # settle_pr > 0 always required in ABSOLUTE/PERCENTILE modes
    base_mask = df["settle_pr"].fillna(0.0) > 0

    if mode == "ABSOLUTE":
        mask = (
            base_mask
            & (df["contracts"].fillna(-np.inf) >= float(min_contracts))
            & (df["open_int"].fillna(-np.inf) >= float(min_open_int))
        )
        out = df.loc[mask].copy()
        return out.reset_index(drop=True)

    # PERCENTILE thresholds within band_df
    p = float(liquidity_percentile)
    if p < 0 or p > 100:
        raise ValueError("liquidity_percentile must be within [0, 100]")

    contracts_thr = _nanpercentile(df["contracts"].to_numpy(dtype=float), p)
    open_int_thr = _nanpercentile(df["open_int"].to_numpy(dtype=float), p)

    mask = (
        base_mask
        & (df["contracts"].fillna(-np.inf) >= contracts_thr)
        & (df["open_int"].fillna(-np.inf) >= open_int_thr)
    )
    out = df.loc[mask].copy()
    return out.reset_index(drop=True)


def select_atm_strike(
    filt_df: pd.DataFrame,
    spot_close: float,
    max_atm_search_steps: int,
) -> Optional[float]:
    """
    Select closest strike to spot where both CE and PE exist after filtering.

    Fallback:
      - Try the next-closest strikes, up to max_atm_search_steps additional attempts.
      - Interpreted as: examine strikes in distance order for i=0..max_atm_search_steps inclusive.
    """
    if filt_df is None or filt_df.empty:
        return None

    _require_columns(filt_df, {"strike_pr", "option_typ"}, ctx="select_atm_strike")

    if max_atm_search_steps < 0:
        raise ValueError("max_atm_search_steps must be >= 0")

    strikes = (
        pd.to_numeric(filt_df["strike_pr"], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
    strikes = sorted(float(s) for s in strikes)
    if len(strikes) == 0:
        return None

    spot = float(spot_close)
    ordered = sorted(strikes, key=lambda k: (abs(k - spot), k))

    # Precompute available option types per strike for fast checking
    tmp = filt_df.copy()
    tmp["strike_pr"] = pd.to_numeric(tmp["strike_pr"], errors="coerce")
    tmp["option_typ"] = tmp["option_typ"].astype(str)

    types_by_strike = (
        tmp.dropna(subset=["strike_pr"])
        .groupby("strike_pr")["option_typ"]
        .apply(lambda s: set(x.upper().strip() for x in s.tolist()))
        .to_dict()
    )

    for i, strike in enumerate(ordered):
        if i > max_atm_search_steps:
            break
        types = types_by_strike.get(strike, set())
        if "CE" in types and "PE" in types:
            return float(strike)

    return None


def select_otm_strike_above(filt_df: pd.DataFrame, atm: float, points: float) -> Optional[float]:
    """
    Preferred: atm + points. If unavailable: nearest higher strike available. None if none above.
    """
    return _select_otm_strike_directional(filt_df, atm=float(atm), target=float(atm) + float(points), direction="above")


def select_otm_strike_below(filt_df: pd.DataFrame, atm: float, points: float) -> Optional[float]:
    """
    Preferred: atm - points. If unavailable: nearest lower strike available. None if none below.
    """
    return _select_otm_strike_directional(filt_df, atm=float(atm), target=float(atm) - float(points), direction="below")


def _select_otm_strike_directional(
    filt_df: pd.DataFrame, atm: float, target: float, direction: str
) -> Optional[float]:
    if filt_df is None or filt_df.empty:
        return None
    _require_columns(filt_df, {"strike_pr"}, ctx=f"_select_otm_strike_directional({direction})")

    strikes = (
        pd.to_numeric(filt_df["strike_pr"], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
    strikes = sorted(float(s) for s in strikes)
    if not strikes:
        return None

    strike_set = set(strikes)
    if target in strike_set:
        return float(target)

    if direction == "above":
        above = [s for s in strikes if s > atm]
        if not above:
            return None
        # nearest above by absolute distance from target; tie-break smaller strike
        above.sort(key=lambda s: (abs(s - target), s))
        return float(above[0])

    if direction == "below":
        below = [s for s in strikes if s < atm]
        if not below:
            return None
        # nearest below by absolute distance from target; tie-break larger strike (closer to atm)
        below.sort(key=lambda s: (abs(s - target), -s))
        return float(below[0])

    raise ValueError(f"direction must be 'above' or 'below'; got {direction!r}")


def _require_columns(df: pd.DataFrame, required: Sequence[str], ctx: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{ctx}: missing required columns: {missing}")


def _nanpercentile(values: np.ndarray, percentile: float) -> float:
    """
    NumPy percentile API changed from 'interpolation' to 'method'.
    This helper keeps behavior deterministic across versions.
    """
    values = values.astype(float)
    if values.size == 0:
        return float("nan")
    try:
        # NumPy >= 1.22
        return float(np.nanpercentile(values, percentile, method="linear"))
    except TypeError:
        # Older NumPy
        return float(np.nanpercentile(values, percentile, interpolation="linear"))
