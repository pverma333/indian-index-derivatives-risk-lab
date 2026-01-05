from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Literal, Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CoverageMode = Literal["ASOF", "STRICT"]

# --- Skip reasons (enum-like strings) ---
SKIP_EMPTY_LEG_SLICE: Final[str] = "EMPTY_LEG_SLICE"
SKIP_MISSING_ROWS_USED_SLICE: Final[str] = "MISSING_MARKET_ROWS_IN_USED_SLICE"
SKIP_STRICT_MARKET_END_BEFORE_EXIT: Final[str] = "MARKET_WINDOW_END_BEFORE_EXIT_STRICT"


# --- Output columns (required by ticket) ---
LEGS_PNL_REQUIRED_COLS: Final[List[str]] = [
    # Identifiers
    "date",
    "strategy_name",
    "tenor",
    "trade_id",
    "leg_id",
    "symbol",
    "instrument",
    "expiry_dt",
    "strike_pr",
    "option_typ",
    "side",
    "qty_lots",
    "lot_size",
    "units",
    "entry_date",
    "exit_date",
    "entry_price",
    # Pricing + P&L
    "settle_used",
    "settle_prev_used",
    "mtm_pnl",
    "gap_risk_pnl_proxy",
    "gap_method",
    "price_method",
    # Pass-through (when present)
    "rate_91d",
    "rate_182d",
    "rate_364d",
    "vix_close",
    "contracts",
    "open_int",
    "chg_in_oi",
    # ASOF / lifecycle
    "market_max_date",
    "as_of_date_used",
    "end_date_used",
    "is_open",
    "status",
    "coverage_mode",
]

SKIPS_REQUIRED_COLS: Final[List[str]] = [
    "strategy_name",
    "tenor",
    "trade_id",
    "leg_id",
    "symbol",
    "instrument",
    "expiry_dt",
    "strike_pr",
    "option_typ",
    "entry_date",
    "exit_date",
    "market_max_date",
    "as_of_date_used",
    "end_date_used",
    "coverage_mode",
    "reason",
    "details",
]


PASS_THROUGH_COLS: Final[List[str]] = [
    "rate_91d",
    "rate_182d",
    "rate_364d",
    "vix_close",
    "contracts",
    "open_int",
    "chg_in_oi",
]


@dataclass(frozen=True)
class _SkipRow:
    data: Dict[str, Any]

    @staticmethod
    def from_leg_base(
        leg: pd.Series,
        *,
        market_max_date: pd.Timestamp,
        as_of_date_used: pd.Timestamp,
        end_date_used: pd.Timestamp,
        coverage_mode: CoverageMode,
        reason: str,
        details: str = "",
    ) -> "_SkipRow":
        # Deterministic, short details only
        details = (details or "").strip()
        if len(details) > 200:
            details = details[:200]

        base = {
            "strategy_name": leg.get("strategy_name"),
            "tenor": leg.get("tenor"),
            "trade_id": leg.get("trade_id"),
            "leg_id": leg.get("leg_id"),
            "symbol": leg.get("symbol"),
            "instrument": leg.get("instrument"),
            "expiry_dt": leg.get("expiry_dt"),
            "strike_pr": leg.get("strike_pr"),
            "option_typ": leg.get("option_typ"),
            "entry_date": leg.get("entry_date"),
            "exit_date": leg.get("exit_date"),
            "market_max_date": market_max_date,
            "as_of_date_used": as_of_date_used,
            "end_date_used": end_date_used,
            "coverage_mode": coverage_mode,
            "reason": reason,
            "details": details,
        }
        return _SkipRow(base)


def _normalize_date_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()


def _require_columns(df: pd.DataFrame, required: List[str], *, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _normalize_inputs(market_df: pd.DataFrame, trades_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    m = market_df.copy()
    t = trades_df.copy()

    for c in ("date", "expiry_dt"):
        _normalize_date_col(m, c)

    for c in ("entry_date", "exit_date", "expiry_dt"):
        _normalize_date_col(t, c)

    return m, t


def _compute_asof_dates(market_df: pd.DataFrame, as_of_date: Optional[pd.Timestamp]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if market_df.empty:
        raise ValueError("market_df is empty after validation; cannot compute market_max_date")

    market_max_date = pd.to_datetime(market_df["date"]).max()
    if pd.isna(market_max_date):
        raise ValueError("market_df.date has no valid values; cannot compute market_max_date")

    if as_of_date is None:
        as_of_date_used = market_max_date
    else:
        a = pd.to_datetime(as_of_date).normalize()
        as_of_date_used = min(a, market_max_date)

    return market_max_date, as_of_date_used


def _expected_dates_for_symbol(market_df: pd.DataFrame, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    # Critical nuance: expected dates derived ONLY from input market_df (already capped by runner).
    mask = (market_df["symbol"].astype(str) == str(symbol)) & (market_df["date"] >= start) & (market_df["date"] <= end)
    d = market_df.loc[mask, "date"].dropna().drop_duplicates().sort_values()
    return d.to_numpy()


def _filter_leg_market(market_df: pd.DataFrame, leg: pd.Series) -> pd.DataFrame:
    symbol = str(leg["symbol"])
    instrument = str(leg["instrument"])
    expiry_dt = leg["expiry_dt"]

    mask = (
        (market_df["symbol"].astype(str) == symbol)
        & (market_df["instrument"].astype(str) == instrument)
        & (market_df["expiry_dt"] == expiry_dt)
    )

    if instrument == "OPTIDX":
        mask = mask & (market_df["strike_pr"] == leg["strike_pr"]) & (market_df["option_typ"].astype(str) == str(leg["option_typ"]))

    out = market_df.loc[mask].copy()
    out = out.sort_values("date", kind="mergesort")  # stable sort
    return out


def compute_legs_pnl(
    market_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    coverage_mode: CoverageMode = "ASOF",
    as_of_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-leg per-day MTM P&L with deterministic ASOF coverage and OPEN/CLOSED labeling.

    Returns
    -------
    legs_pnl_df : pd.DataFrame
        Per-leg per-day timeseries.
    skips_df : pd.DataFrame
        One row per skipped leg (deterministic, manifest-friendly).
    """
    if coverage_mode not in ("ASOF", "STRICT"):
        raise ValueError(f"coverage_mode must be ASOF or STRICT, got {coverage_mode}")

    if market_df is None or not isinstance(market_df, pd.DataFrame):
        raise ValueError(f"market_df must be a pandas DataFrame, got {type(market_df)}")
    if trades_df is None or not isinstance(trades_df, pd.DataFrame):
        raise ValueError(f"trades_df must be a pandas DataFrame, got {type(trades_df)}")

    # Base required cols
    _require_columns(
        market_df,
        ["date", "symbol", "instrument", "expiry_dt", "settle_used", "lot_size", "index_open_price"],
        name="market_df",
    )
    _require_columns(
        trades_df,
        ["strategy_name", "tenor", "trade_id", "leg_id", "symbol", "instrument", "expiry_dt",
         "entry_date", "exit_date", "side", "qty_lots", "entry_price"],
        name="trades_df",
    )

    # Conditional requirements for OPTIDX legs
    has_opt = (trades_df["instrument"].astype(str) == "OPTIDX").any()
    if has_opt:
        _require_columns(market_df, ["strike_pr", "option_typ"], name="market_df")
        _require_columns(trades_df, ["strike_pr", "option_typ"], name="trades_df")

    mkt, trd = _normalize_inputs(market_df, trades_df)

    # Fail-fast null entry_price
    if trd["entry_price"].isna().any():
        bad = trd.loc[trd["entry_price"].isna(), ["strategy_name", "tenor", "trade_id", "leg_id"]].head(20)
        raise ValueError(f"trades_df contains null entry_price (first 20 keys):\n{bad.to_string(index=False)}")

    # Fail-fast: OPTIDX legs require strike_pr and option_typ not-null
    if has_opt:
        opt_legs = trd["instrument"].astype(str) == "OPTIDX"
        miss_strike = opt_legs & trd["strike_pr"].isna()
        miss_typ = opt_legs & trd["option_typ"].isna()
        if miss_strike.any() or miss_typ.any():
            bad = trd.loc[miss_strike | miss_typ, ["strategy_name", "tenor", "trade_id", "leg_id", "strike_pr", "option_typ"]].head(20)
            raise ValueError(f"OPTIDX legs missing strike_pr/option_typ (first 20):\n{bad.to_string(index=False)}")

    market_max_date, as_of_date_used = _compute_asof_dates(mkt, as_of_date)

    legs_out: List[pd.DataFrame] = []
    skips: List[_SkipRow] = []

    # Deterministic iteration order
    trd_iter = trd.sort_values(["strategy_name", "tenor", "trade_id", "leg_id"], kind="mergesort")

    for _, leg in trd_iter.iterrows():
        entry_date = leg["entry_date"]
        exit_date = leg["exit_date"]

        end_date_used = min(exit_date, as_of_date_used)
        is_open = bool(end_date_used < exit_date)
        status = "OPEN" if is_open else "CLOSED"

        # STRICT coverage rule
        if coverage_mode == "STRICT" and market_max_date < exit_date:
            skips.append(
                _SkipRow.from_leg_base(
                    leg,
                    market_max_date=market_max_date,
                    as_of_date_used=as_of_date_used,
                    end_date_used=end_date_used,
                    coverage_mode=coverage_mode,
                    reason=SKIP_STRICT_MARKET_END_BEFORE_EXIT,
                    details=f"market_max_date={market_max_date.date().isoformat()} < exit_date={exit_date.date().isoformat()}",
                )
            )
            continue

        # Contract slice
        contract_df = _filter_leg_market(mkt, leg)
        used_mask = (contract_df["date"] >= entry_date) & (contract_df["date"] <= end_date_used)
        used_df = contract_df.loc[used_mask].copy()

        if used_df.empty:
            skips.append(
                _SkipRow.from_leg_base(
                    leg,
                    market_max_date=market_max_date,
                    as_of_date_used=as_of_date_used,
                    end_date_used=end_date_used,
                    coverage_mode=coverage_mode,
                    reason=SKIP_EMPTY_LEG_SLICE,
                    details="no market rows after filtering contract and [entry_date, end_date_used]",
                )
            )
            continue

        # Expected trading dates nuance: derived from input market_df for this symbol, capped to end_date_used.
        expected_dates = _expected_dates_for_symbol(mkt, str(leg["symbol"]), entry_date, end_date_used)
        present_dates = used_df["date"].dropna().drop_duplicates().sort_values().to_numpy()
        missing_dates = np.setdiff1d(expected_dates, present_dates, assume_unique=False)

        if missing_dates.size > 0:
            # deterministic details: first up to 5 dates
            first5 = [pd.Timestamp(d).date().isoformat() for d in missing_dates[:5]]
            skips.append(
                _SkipRow.from_leg_base(
                    leg,
                    market_max_date=market_max_date,
                    as_of_date_used=as_of_date_used,
                    end_date_used=end_date_used,
                    coverage_mode=coverage_mode,
                    reason=SKIP_MISSING_ROWS_USED_SLICE,
                    details=f"missing_count={int(missing_dates.size)} first_missing={first5}",
                )
            )
            continue

        # Compute units(t) = side * qty_lots * lot_size(t)
        side = float(leg["side"])
        qty_lots = float(leg["qty_lots"])
        used_df["units"] = side * qty_lots * used_df["lot_size"].astype(float)

        # Settle prev (day-0 anchored to entry_price)
        used_df = used_df.sort_values("date", kind="mergesort")
        used_df["settle_prev_used"] = used_df["settle_used"].shift(1)
        used_df.iloc[0, used_df.columns.get_loc("settle_prev_used")] = float(leg["entry_price"])

        # MTM P&L
        used_df["mtm_pnl"] = used_df["units"] * (used_df["settle_used"].astype(float) - used_df["settle_prev_used"].astype(float))

        # Gap proxy
        instrument = str(leg["instrument"])
        if instrument == "OPTIDX":
            opt_typ = str(leg["option_typ"])
            strike = float(leg["strike_pr"])
            idx_open = used_df["index_open_price"].astype(float).to_numpy()

            if opt_typ == "CE":
                intr_open = np.maximum(idx_open - strike, 0.0)
            elif opt_typ == "PE":
                intr_open = np.maximum(strike - idx_open, 0.0)
            else:
                raise ValueError(f"OPTIDX option_typ must be CE or PE, got '{opt_typ}' for leg_id={leg.get('leg_id')}")

            used_df["gap_risk_pnl_proxy"] = used_df["units"] * (intr_open - used_df["settle_prev_used"].astype(float).to_numpy())
            used_df["gap_method"] = "INTRINSIC_OPEN_PROXY"

        elif instrument == "FUTIDX":
            used_df["gap_risk_pnl_proxy"] = used_df["units"] * (
                used_df["index_open_price"].astype(float) - used_df["settle_prev_used"].astype(float)
            )
            used_df["gap_method"] = "INDEX_OPEN_PROXY"
        else:
            raise ValueError(f"Unsupported instrument '{instrument}' in engine_pnl")

        # price_method pass-through
        if "price_method" in used_df.columns:
            used_df["price_method"] = used_df["price_method"].astype(str)
        else:
            used_df["price_method"] = "SETTLE_USED"

        # Pass-through market fields (NA if missing)
        for c in PASS_THROUGH_COLS:
            if c not in used_df.columns:
                used_df[c] = pd.NA

        # Attach trade identifiers + lifecycle fields
        used_df["strategy_name"] = leg["strategy_name"]
        used_df["tenor"] = leg["tenor"]
        used_df["trade_id"] = leg["trade_id"]
        used_df["leg_id"] = leg["leg_id"]

        used_df["side"] = leg["side"]
        used_df["qty_lots"] = leg["qty_lots"]
        used_df["entry_date"] = entry_date
        used_df["exit_date"] = exit_date
        used_df["entry_price"] = leg["entry_price"]

        used_df["market_max_date"] = market_max_date
        used_df["as_of_date_used"] = as_of_date_used
        used_df["end_date_used"] = end_date_used
        used_df["is_open"] = is_open
        used_df["status"] = status
        used_df["coverage_mode"] = coverage_mode

        # Ensure identifier columns exist even for FUTIDX (stable schema)
        if "strike_pr" not in used_df.columns:
            used_df["strike_pr"] = leg.get("strike_pr", 0.0)
        if "option_typ" not in used_df.columns:
            used_df["option_typ"] = leg.get("option_typ", pd.NA)

        # Select/Order required columns
        legs_out.append(used_df.reindex(columns=LEGS_PNL_REQUIRED_COLS))

    legs_pnl_df = pd.concat(legs_out, axis=0, ignore_index=True) if legs_out else pd.DataFrame(columns=LEGS_PNL_REQUIRED_COLS)
    skips_df = pd.DataFrame([s.data for s in skips], columns=SKIPS_REQUIRED_COLS)

    # Deterministic sorting
    if not legs_pnl_df.empty:
        legs_pnl_df = legs_pnl_df.sort_values(["strategy_name", "tenor", "trade_id", "leg_id", "date"], kind="mergesort").reset_index(drop=True)
    if not skips_df.empty:
        skips_df = skips_df.sort_values(["strategy_name", "tenor", "trade_id", "leg_id"], kind="mergesort").reset_index(drop=True)

    # No-NaN requirement for settle_prev_used after day-0 for valid slices
    if not legs_pnl_df.empty and legs_pnl_df["settle_prev_used"].isna().any():
        # This should not happen; fail fast to avoid silent P&L corruption.
        bad = legs_pnl_df.loc[legs_pnl_df["settle_prev_used"].isna(), ["strategy_name", "tenor", "trade_id", "leg_id", "date"]].head(20)
        raise ValueError(f"settle_prev_used contains NaN for valid slices (first 20):\n{bad.to_string(index=False)}")

    logger.info(
        "compute_legs_pnl done: legs_rows=%d skips=%d market_max_date=%s as_of_date_used=%s coverage_mode=%s",
        int(len(legs_pnl_df)),
        int(len(skips_df)),
        str(market_max_date.date()),
        str(as_of_date_used.date()),
        coverage_mode,
    )

    return legs_pnl_df, skips_df
