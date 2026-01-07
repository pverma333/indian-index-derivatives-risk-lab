from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd


LEG_KEY_COLS: List[str] = ["strategy_name", "tenor", "trade_id", "leg_id"]
TRADE_KEY_COLS: List[str] = ["strategy_name", "tenor", "trade_id"]
STRAT_KEY_COLS: List[str] = ["strategy_name", "tenor"]


def _distinct_legs(legs_pnl_df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "strategy_name",
        "tenor",
        "trade_id",
        "leg_id",
        "status",
        "market_max_date",
        "as_of_date_used",
        "end_date_used",
        "coverage_mode",
    ]
    missing = [c for c in required if c not in legs_pnl_df.columns]
    if missing:
        raise ValueError(f"legs_pnl_df missing required columns for aggregation: {missing}")

    leg_level = (
        legs_pnl_df.sort_values(LEG_KEY_COLS + ["date"], kind="mergesort")
        .groupby(LEG_KEY_COLS, as_index=False)
        .agg(
            status=("status", "first"),
            market_max_date=("market_max_date", "first"),
            as_of_date_used=("as_of_date_used", "first"),
            end_date_used=("end_date_used", "first"),
            coverage_mode=("coverage_mode", "first"),
        )
    )
    return leg_level


def aggregate_legs_to_trade_pnl(
    legs_pnl_df: pd.DataFrame,
    skips_df: pd.DataFrame,
    *,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """
    trade_pnl_df must include:
      - market_max_date, as_of_date_used, coverage_mode
      - counts: n_open_legs, n_closed_legs, n_skipped_legs
    Reconciliation:
      sum(legs.mtm_pnl) == sum(trade.total_mtm_pnl) within tolerance.
    """
    out_cols = (
        TRADE_KEY_COLS
        + [
            "total_mtm_pnl",
            "total_gap_risk_pnl_proxy",
            "market_max_date",
            "as_of_date_used",
            "end_date_used",
            "coverage_mode",
            "n_legs_emitted",
            "n_open_legs",
            "n_closed_legs",
            "n_skipped_legs",
        ]
    )

    if legs_pnl_df is None or legs_pnl_df.empty:
        return pd.DataFrame(columns=out_cols)

    df = legs_pnl_df.copy()

    if "mtm_pnl" not in df.columns:
        raise ValueError("legs_pnl_df missing mtm_pnl")
    df["mtm_pnl"] = pd.to_numeric(df["mtm_pnl"], errors="coerce").fillna(0.0)

    if "gap_risk_pnl_proxy" in df.columns:
        df["gap_risk_pnl_proxy"] = pd.to_numeric(df["gap_risk_pnl_proxy"], errors="coerce").fillna(0.0)
    else:
        df["gap_risk_pnl_proxy"] = 0.0

    totals = (
        df.groupby(TRADE_KEY_COLS, as_index=False)
        .agg(
            total_mtm_pnl=("mtm_pnl", "sum"),
            total_gap_risk_pnl_proxy=("gap_risk_pnl_proxy", "sum"),
            market_max_date=("market_max_date", "first"),
            as_of_date_used=("as_of_date_used", "first"),
            end_date_used=("end_date_used", "first"),
            coverage_mode=("coverage_mode", "first"),
        )
    )

    leg_level = _distinct_legs(df)
    counts = (
        leg_level.groupby(TRADE_KEY_COLS, as_index=False)
        .agg(
            n_legs_emitted=("leg_id", "count"),
            n_open_legs=("status", lambda s: int((s.astype(str) == "OPEN").sum())),
            n_closed_legs=("status", lambda s: int((s.astype(str) == "CLOSED").sum())),
        )
    )

    # Skips counts from skips_df (artifact-driven)
    if skips_df is not None and not skips_df.empty:
        need = ["strategy_name", "tenor", "trade_id", "leg_id"]
        miss = [c for c in need if c not in skips_df.columns]
        if miss:
            raise ValueError(f"skips_df missing required columns for aggregation: {miss}")
        skipped_legs = skips_df[need].drop_duplicates()
        skipped_counts = skipped_legs.groupby(TRADE_KEY_COLS, as_index=False).agg(n_skipped_legs=("leg_id", "count"))
    else:
        skipped_counts = pd.DataFrame(columns=TRADE_KEY_COLS + ["n_skipped_legs"])

    out = totals.merge(counts, how="left", on=TRADE_KEY_COLS).merge(skipped_counts, how="left", on=TRADE_KEY_COLS)
    out["n_skipped_legs"] = pd.to_numeric(out["n_skipped_legs"], errors="coerce").fillna(0).astype(int)

    legs_sum = float(df["mtm_pnl"].sum())
    trades_sum = float(out["total_mtm_pnl"].sum())
    if abs(legs_sum - trades_sum) > tolerance:
        raise AssertionError(
            f"Reconciliation failed: sum(legs.mtm_pnl)={legs_sum} vs sum(trade.total_mtm_pnl)={trades_sum}"
        )

    out = out.reindex(columns=out_cols)
    out = out.sort_values(TRADE_KEY_COLS, kind="mergesort").reset_index(drop=True)
    return out


def aggregate_trades_to_strategy_pnl(
    trade_pnl_df: pd.DataFrame,
    *,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """
    strategy_pnl_df must include:
      - market_max_date, as_of_date_used, coverage_mode
      - counts: n_open_legs, n_closed_legs, n_skipped_legs
    Reconciliation:
      sum(trade.total_mtm_pnl) == sum(strategy.total_mtm_pnl) within tolerance.
    """
    out_cols = (
        STRAT_KEY_COLS
        + [
            "total_mtm_pnl",
            "total_gap_risk_pnl_proxy",
            "market_max_date",
            "as_of_date_used",
            "end_date_used",
            "coverage_mode",
            "n_trades",
            "n_legs_emitted",
            "n_open_legs",
            "n_closed_legs",
            "n_skipped_legs",
        ]
    )

    if trade_pnl_df is None or trade_pnl_df.empty:
        return pd.DataFrame(columns=out_cols)

    df = trade_pnl_df.copy()
    df["total_mtm_pnl"] = pd.to_numeric(df["total_mtm_pnl"], errors="coerce").fillna(0.0)
    df["total_gap_risk_pnl_proxy"] = pd.to_numeric(df["total_gap_risk_pnl_proxy"], errors="coerce").fillna(0.0)

    out = (
        df.groupby(STRAT_KEY_COLS, as_index=False)
        .agg(
            total_mtm_pnl=("total_mtm_pnl", "sum"),
            total_gap_risk_pnl_proxy=("total_gap_risk_pnl_proxy", "sum"),
            market_max_date=("market_max_date", "first"),
            as_of_date_used=("as_of_date_used", "first"),
            end_date_used=("end_date_used", "first"),
            coverage_mode=("coverage_mode", "first"),
            n_trades=("trade_id", "nunique"),
            n_legs_emitted=("n_legs_emitted", "sum"),
            n_open_legs=("n_open_legs", "sum"),
            n_closed_legs=("n_closed_legs", "sum"),
            n_skipped_legs=("n_skipped_legs", "sum"),
        )
    )

    trades_sum = float(df["total_mtm_pnl"].sum())
    strat_sum = float(out["total_mtm_pnl"].sum())
    if abs(trades_sum - strat_sum) > tolerance:
        raise AssertionError(
            f"Reconciliation failed: sum(trade.total_mtm_pnl)={trades_sum} vs sum(strategy.total_mtm_pnl)={strat_sum}"
        )

    out = out.reindex(columns=out_cols)
    out = out.sort_values(STRAT_KEY_COLS, kind="mergesort").reset_index(drop=True)
    return out
