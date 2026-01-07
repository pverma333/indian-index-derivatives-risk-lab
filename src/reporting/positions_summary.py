from __future__ import annotations

from typing import List

import pandas as pd


LEG_KEY_COLS: List[str] = [
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
    "entry_date",
    "exit_date",
]


META_COLS: List[str] = [
    "market_max_date",
    "as_of_date_used",
    "end_date_used",
    "coverage_mode",
    "status",
    "is_open",
]


def build_positions_df(legs_pnl_df: pd.DataFrame) -> pd.DataFrame:
    """
    positions_df (dashboard-safe):

    Group at leg level:
      cum_pnl_asof = sum(mtm_pnl)
      last_valuation_date = max(date)
      status constant per leg
      realized_pnl = cum_pnl_asof if status == "CLOSED" else 0
      unrealized_pnl = cum_pnl_asof if status == "OPEN" else 0
      total_pnl = realized_pnl + unrealized_pnl
    """
    out_cols = (
        LEG_KEY_COLS
        + META_COLS
        + ["last_valuation_date", "cum_pnl_asof", "realized_pnl", "unrealized_pnl", "total_pnl"]
    )

    if legs_pnl_df is None or legs_pnl_df.empty:
        return pd.DataFrame(columns=out_cols)

    df = legs_pnl_df.copy()

    required = set(LEG_KEY_COLS + META_COLS + ["date", "mtm_pnl"])
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"legs_pnl_df missing required columns for positions summary: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    for c in ("expiry_dt", "entry_date", "exit_date", "market_max_date", "as_of_date_used", "end_date_used"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="raise").dt.normalize()

    df["mtm_pnl"] = pd.to_numeric(df["mtm_pnl"], errors="coerce").fillna(0.0)

    group_cols = LEG_KEY_COLS

    # IMPORTANT:
    # Some key columns can be NA (e.g., option_typ for FUTIDX legs).
    # pandas groupby defaults to dropna=True, which would drop those legs entirely.
    # We must preserve them -> dropna=False.
    status_nunique = df.groupby(group_cols, dropna=False)["status"].nunique(dropna=False)
    bad = status_nunique[status_nunique > 1]
    if not bad.empty:
        sample = bad.head(20).reset_index()
        raise ValueError(
            "status must be constant per leg; found multiple statuses for some legs. "
            f"Sample (first 20):\n{sample.to_string(index=False)}"
        )

    agg = (
        df.sort_values(group_cols + ["date"], kind="mergesort")
        .groupby(group_cols, as_index=False, dropna=False)
        .agg(
            cum_pnl_asof=("mtm_pnl", "sum"),
            last_valuation_date=("date", "max"),
            status=("status", "first"),
            is_open=("is_open", "first"),
            market_max_date=("market_max_date", "first"),
            as_of_date_used=("as_of_date_used", "first"),
            end_date_used=("end_date_used", "first"),
            coverage_mode=("coverage_mode", "first"),
        )
    )

    agg["realized_pnl"] = agg["cum_pnl_asof"].where(agg["status"].astype(str) == "CLOSED", 0.0)
    agg["unrealized_pnl"] = agg["cum_pnl_asof"].where(agg["status"].astype(str) == "OPEN", 0.0)
    agg["total_pnl"] = agg["realized_pnl"] + agg["unrealized_pnl"]

    agg = agg.reindex(columns=out_cols)
    agg = agg.sort_values(["strategy_name", "tenor", "trade_id", "leg_id"], kind="mergesort").reset_index(drop=True)
    return agg
