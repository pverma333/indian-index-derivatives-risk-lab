from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: Tuple[str, ...] = (
    # Keys
    "strategy_name",
    "tenor",
    "trade_id",
    "leg_id",
    # Contract
    "symbol",
    "instrument",
    "expiry_dt",
    "strike_pr",
    "option_typ",
    # Position
    "side",
    "qty_lots",
    # Lifecycle
    "entry_date",
    "exit_date",
    # Selection params
    "strike_band_n",
    "width_points",
    "otm_distance_points",
    "max_atm_search_steps",
    "liquidity_mode",
    "min_contracts",
    "min_open_int",
    "liquidity_percentile",
    "exit_rule",
    "exit_k_days",
    "fees_bps",
    "fixed_fee_per_lot",
)

KEY_COLUMNS: Tuple[str, ...] = ("strategy_name", "tenor", "trade_id", "leg_id")

ALLOWED_TENORS: Set[str] = {"WEEKLY", "MONTHLY"}
ALLOWED_LIQUIDITY_MODES: Set[str] = {"OFF", "ABSOLUTE", "PERCENTILE"}
ALLOWED_SIDES: Set[int] = {1, -1}

# If your repo uses a different string, update this set.
K_DAYS_EXIT_RULES: Set[str] = {"K_DAYS_BEFORE_EXPIRY"}


@dataclass(frozen=True)
class TradeSchemaError(ValueError):
    """
    Raised when trades_df violates required schema/invariants.

    Attributes
    ----------
    missing_columns:
        Required columns that are absent.
    violations_keys:
        First 20 violating rows (keys only) to help the caller debug joins/P&L mixing.
    rule_errors:
        Human-readable rule error strings (optional; useful for logs).
    """

    message: str
    missing_columns: List[str]
    violations_keys: pd.DataFrame
    rule_errors: List[str]

    def __str__(self) -> str:
        base = self.message
        parts: List[str] = [base]

        if self.missing_columns:
            parts.append(f"Missing columns ({len(self.missing_columns)}): {self.missing_columns}")

        if self.rule_errors:
            parts.append("Rule errors:")
            parts.extend([f"- {e}" for e in self.rule_errors])

        if not self.violations_keys.empty:
            parts.append("First 20 violating rows (keys only):")
            # Render deterministically
            parts.append(self.violations_keys.to_string(index=False))

        return "\n".join(parts)


def validate_trades_df(trades_df: pd.DataFrame) -> None:
    """
    Validate trades_df before engine computations.

    Raises
    ------
    TradeSchemaError
        If required columns are missing or invariants are violated.
    """
    if trades_df is None:
        raise TradeSchemaError(
            message="trades_df is None",
            missing_columns=list(REQUIRED_COLUMNS),
            violations_keys=pd.DataFrame(columns=list(KEY_COLUMNS)),
            rule_errors=["trades_df is None"],
        )

    if not isinstance(trades_df, pd.DataFrame):
        raise TradeSchemaError(
            message=f"trades_df must be a pandas DataFrame, got {type(trades_df)}",
            missing_columns=list(REQUIRED_COLUMNS),
            violations_keys=pd.DataFrame(columns=list(KEY_COLUMNS)),
            rule_errors=[f"bad_type={type(trades_df)}"],
        )

    logger.info("Validating trades_df schema: rows=%d cols=%d", len(trades_df), len(trades_df.columns))

    missing = _missing_required_columns(trades_df, REQUIRED_COLUMNS)
    if missing:
        raise TradeSchemaError(
            message="trades_df missing required columns",
            missing_columns=missing,
            violations_keys=pd.DataFrame(columns=list(KEY_COLUMNS)),
            rule_errors=[],
        )

    # Fail-fast: leg_id uniqueness across entire trades_df
    dup_keys = _duplicate_leg_id_keys(trades_df)
    if not dup_keys.empty:
        raise TradeSchemaError(
            message="Duplicate leg_id detected (must be unique across entire trades_df)",
            missing_columns=[],
            violations_keys=dup_keys.head(20).copy(),
            rule_errors=["DUPLICATE_LEG_ID"],
        )

    rule_errors: List[str] = []
    violating_keys = []

    # side ∈ {+1, -1}
    bad_side = _keys_where_side_invalid(trades_df)
    if not bad_side.empty:
        rule_errors.append("INVALID_SIDE: side must be in {+1, -1}")
        violating_keys.append(bad_side)

    # qty_lots integer >= 1
    bad_qty = _keys_where_qty_lots_invalid(trades_df)
    if not bad_qty.empty:
        rule_errors.append("INVALID_QTY_LOTS: qty_lots must be integer and >= 1")
        violating_keys.append(bad_qty)

    # entry_date <= exit_date (and parseable)
    bad_dates = _keys_where_entry_after_exit(trades_df)
    if not bad_dates.empty:
        rule_errors.append("INVALID_DATES: entry_date must be <= exit_date (and both parseable)")
        violating_keys.append(bad_dates)

    # tenor enum
    bad_tenor = _keys_where_tenor_invalid(trades_df)
    if not bad_tenor.empty:
        rule_errors.append(f"INVALID_TENOR: tenor must be in {sorted(ALLOWED_TENORS)}")
        violating_keys.append(bad_tenor)

    # liquidity_mode enum
    bad_liq = _keys_where_liquidity_mode_invalid(trades_df)
    if not bad_liq.empty:
        rule_errors.append(f"INVALID_LIQUIDITY_MODE: liquidity_mode must be in {sorted(ALLOWED_LIQUIDITY_MODES)}")
        violating_keys.append(bad_liq)

    # exit_k_days nullable unless K-days rule; if K-days rule then required int >= 1
    bad_exit_k = _keys_where_exit_k_days_invalid(trades_df)
    if not bad_exit_k.empty:
        rule_errors.append(
            "INVALID_EXIT_K_DAYS: exit_k_days must be int >= 1 when exit_rule is a K-days rule; otherwise nullable"
        )
        violating_keys.append(bad_exit_k)

    if rule_errors:
        keys_union = _union_keys(violating_keys).head(20).copy()
        raise TradeSchemaError(
            message="trades_df failed validation",
            missing_columns=[],
            violations_keys=keys_union,
            rule_errors=rule_errors,
        )

    logger.info("trades_df validation passed: rows=%d", len(trades_df))


def _missing_required_columns(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    present = set(df.columns)
    missing = [c for c in required if c not in present]
    return missing


def _duplicate_leg_id_keys(df: pd.DataFrame) -> pd.DataFrame:
    dup_mask = df.duplicated(subset=["leg_id"], keep=False)
    if not dup_mask.any():
        return pd.DataFrame(columns=list(KEY_COLUMNS))
    return df.loc[dup_mask, list(KEY_COLUMNS)].sort_values(list(KEY_COLUMNS))


def _keys_where_side_invalid(df: pd.DataFrame) -> pd.DataFrame:
    s = df["side"]
    bad = s.isna() | ~s.astype(object).isin(ALLOWED_SIDES)
    if not bad.any():
        return pd.DataFrame(columns=list(KEY_COLUMNS))
    return df.loc[bad, list(KEY_COLUMNS)]


def _keys_where_qty_lots_invalid(df: pd.DataFrame) -> pd.DataFrame:
    q = df["qty_lots"]
    # numeric coercion
    q_num = pd.to_numeric(q, errors="coerce")
    is_int_like = q_num.notna() & np.isfinite(q_num.to_numpy()) & (np.floor(q_num) == q_num)
    bad = ~is_int_like | (q_num < 1)
    if not bad.any():
        return pd.DataFrame(columns=list(KEY_COLUMNS))
    return df.loc[bad, list(KEY_COLUMNS)]


def _keys_where_entry_after_exit(df: pd.DataFrame) -> pd.DataFrame:
    entry = pd.to_datetime(df["entry_date"], errors="coerce")
    exit_ = pd.to_datetime(df["exit_date"], errors="coerce")
    bad = entry.isna() | exit_.isna() | (entry > exit_)
    if not bad.any():
        return pd.DataFrame(columns=list(KEY_COLUMNS))
    return df.loc[bad, list(KEY_COLUMNS)]


def _keys_where_tenor_invalid(df: pd.DataFrame) -> pd.DataFrame:
    t = df["tenor"].astype(str)
    bad = ~t.isin(ALLOWED_TENORS)
    if not bad.any():
        return pd.DataFrame(columns=list(KEY_COLUMNS))
    return df.loc[bad, list(KEY_COLUMNS)]


def _keys_where_liquidity_mode_invalid(df: pd.DataFrame) -> pd.DataFrame:
    m = df["liquidity_mode"].astype(str)
    bad = ~m.isin(ALLOWED_LIQUIDITY_MODES)
    if not bad.any():
        return pd.DataFrame(columns=list(KEY_COLUMNS))
    return df.loc[bad, list(KEY_COLUMNS)]


def _keys_where_exit_k_days_invalid(df: pd.DataFrame) -> pd.DataFrame:
    rule = df["exit_rule"].astype(str)
    k = df["exit_k_days"]

    needs_k = rule.isin(K_DAYS_EXIT_RULES)
    if not needs_k.any():
        return pd.DataFrame(columns=list(KEY_COLUMNS))

    k_num = pd.to_numeric(k, errors="coerce")
    is_int_like = k_num.notna() & np.isfinite(k_num.to_numpy()) & (np.floor(k_num) == k_num)
    bad_when_needed = needs_k & (~is_int_like | (k_num < 1))

    if not bad_when_needed.any():
        return pd.DataFrame(columns=list(KEY_COLUMNS))
    return df.loc[bad_when_needed, list(KEY_COLUMNS)]


def _union_keys(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=list(KEY_COLUMNS))
    out = pd.concat(frames, axis=0, ignore_index=True)
    # Deduplicate rows for readability
    out = out.drop_duplicates(subset=list(KEY_COLUMNS))
    return out.sort_values(list(KEY_COLUMNS))
