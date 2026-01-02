# tests/smoke_test_contract_selectors.py
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure repo root is on sys.path so `import src...` works when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.contract_selectors import (  # noqa: E402
    apply_liquidity_filters,
    apply_strike_band,
    get_chain,
    infer_strike_interval,
    select_atm_strike,
    select_otm_strike_above,
    select_otm_strike_below,
)

logger = logging.getLogger("smoke_test_contract_selectors")


REQUIRED_COLS = [
    "date",
    "symbol",
    "instrument",
    "expiry_dt",
    "strike_pr",
    "option_typ",
    "contracts",
    "open_int",
    "settle_pr",
]


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["expiry_dt"] = pd.to_datetime(out["expiry_dt"], errors="coerce")

    for c in ["strike_pr", "contracts", "open_int", "settle_pr"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["symbol"] = out["symbol"].astype(str)
    out["instrument"] = out["instrument"].astype(str)
    out["option_typ"] = out["option_typ"].astype(str)

    return out


def _pick_deterministic_sample(market_df: pd.DataFrame, symbol: str | None):
    """
    Deterministic sample:
      - prefer provided symbol; else smallest symbol lexicographically among OPTIDX rows
      - choose earliest date, then earliest expiry within that date+symbol
    """
    opt = market_df.loc[market_df["instrument"] == "OPTIDX"].copy()
    if opt.empty:
        raise SystemExit("No OPTIDX rows found in the CSV; cannot smoke-test option selectors.")

    chosen_symbol = symbol
    if not chosen_symbol:
        chosen_symbol = sorted(opt["symbol"].dropna().unique().tolist())[0]

    opt = opt.loc[opt["symbol"] == chosen_symbol]
    if opt.empty:
        raise SystemExit(f"No OPTIDX rows found for symbol={chosen_symbol!r}.")

    entry_date = opt["date"].dropna().min()
    if pd.isna(entry_date):
        raise SystemExit("All 'date' values are NaT after parsing; check the CSV date format.")

    opt_d = opt.loc[opt["date"] == entry_date]
    expiry_dt = opt_d["expiry_dt"].dropna().min()
    if pd.isna(expiry_dt):
        raise SystemExit("All 'expiry_dt' values are NaT after parsing; check the CSV expiry_dt format.")

    return chosen_symbol, pd.Timestamp(entry_date), pd.Timestamp(expiry_dt)


def _infer_spot_close_from_chain(chain_df: pd.DataFrame) -> float:
    """
    We often don't have an explicit spot close in the options chain CSV.
    Deterministic proxy:
      1) strike with max total open_int across CE+PE
      2) else median strike
    """
    strikes = chain_df["strike_pr"].dropna()
    if strikes.empty:
        raise SystemExit("Chain has no numeric strikes; cannot infer a spot proxy.")

    if chain_df["open_int"].notna().any():
        by_strike = (
            chain_df.dropna(subset=["strike_pr"])
            .groupby("strike_pr")["open_int"]
            .sum(min_count=1)
            .sort_values(ascending=False, kind="mergesort")
        )
        if not by_strike.empty and pd.notna(by_strike.index[0]):
            return float(by_strike.index[0])

    return float(strikes.median())


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test for contract selectors on a derivatives_clean CSV.")
    ap.add_argument(
        "--csv",
        type=str,
        default="data/curated/derivatives_clean_Q1_2025.csv",
        help="Path to derivatives_clean CSV",
    )
    ap.add_argument("--symbol", type=str, default=None, help="Optional symbol (e.g., NIFTY)")
    ap.add_argument("--strike-band-n", type=int, default=3, help="±N strikes around ATM (by count)")
    ap.add_argument("--max-atm-steps", type=int, default=6, help="ATM fallback search steps")
    ap.add_argument("--liquidity-percentile", type=float, default=50.0, help="Percentile for PERCENTILE mode")
    ap.add_argument("--min-contracts", type=float, default=10.0, help="Min contracts for ABSOLUTE mode")
    ap.add_argument("--min-open-int", type=float, default=100.0, help="Min open_int for ABSOLUTE mode")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    _configure_logging(args.verbose)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    logger.info("Reading CSV: %s", csv_path)
    market_df = pd.read_csv(csv_path)

    logger.info("Rows=%d Cols=%d", len(market_df), len(market_df.columns))
    _require_columns(market_df, REQUIRED_COLS)

    market_df = _coerce_types(market_df)

    # Quick sanity stats
    null_date = int(market_df["date"].isna().sum())
    null_exp = int(market_df["expiry_dt"].isna().sum())
    logger.info("Parsed dates: null date=%d, null expiry_dt=%d", null_date, null_exp)

    symbol, entry_date, expiry_dt = _pick_deterministic_sample(market_df, args.symbol)
    logger.info("Sample chosen: symbol=%s entry_date=%s expiry_dt=%s", symbol, entry_date.date(), expiry_dt.date())

    chain_df = get_chain(market_df, symbol=symbol, expiry_dt=expiry_dt, entry_date=entry_date)
    logger.info("get_chain -> rows=%d", len(chain_df))
    if chain_df.empty:
        logger.warning("Chain is empty for chosen sample. This is skip-safe, but smoke test can't proceed.")
        return 0

    spot_close = _infer_spot_close_from_chain(chain_df)
    logger.info("spot_close proxy used=%.2f", spot_close)

    interval = infer_strike_interval(chain_df)
    logger.info("infer_strike_interval -> %s", interval)

    band_df = apply_strike_band(chain_df, spot_close=spot_close, strike_band_n=args.strike_band_n)
    logger.info("apply_strike_band -> rows=%d strikes=%d", len(band_df), band_df["strike_pr"].nunique())

    # Liquidity modes
    off_df = apply_liquidity_filters(
        band_df,
        liquidity_mode="OFF",
        min_contracts=args.min_contracts,
        min_open_int=args.min_open_int,
        liquidity_percentile=args.liquidity_percentile,
    )
    logger.info("liquidity OFF -> rows=%d", len(off_df))

    abs_df = apply_liquidity_filters(
        band_df,
        liquidity_mode="ABSOLUTE",
        min_contracts=args.min_contracts,
        min_open_int=args.min_open_int,
        liquidity_percentile=args.liquidity_percentile,
    )
    logger.info("liquidity ABSOLUTE -> rows=%d (dropped=%d)", len(abs_df), len(band_df) - len(abs_df))

    pct_df = apply_liquidity_filters(
        band_df,
        liquidity_mode="PERCENTILE",
        min_contracts=args.min_contracts,
        min_open_int=args.min_open_int,
        liquidity_percentile=args.liquidity_percentile,
    )
    logger.info("liquidity PERCENTILE(p=%.1f) -> rows=%d (dropped=%d)", args.liquidity_percentile, len(pct_df), len(band_df) - len(pct_df))

    # Strike selection on the most restrictive df (percentile) to exercise fallback logic
    filt_df = pct_df
    atm = select_atm_strike(filt_df, spot_close=spot_close, max_atm_search_steps=args.max_atm_steps)
    logger.info("select_atm_strike -> %s", atm)

    if atm is None:
        logger.warning("ATM strike not found after filtering (this is allowed; strategy should skip).")
        return 0

    otm_up = select_otm_strike_above(filt_df, atm=atm, points=interval or 0)
    otm_dn = select_otm_strike_below(filt_df, atm=atm, points=interval or 0)
    logger.info("select_otm_strike_above(points=%s) -> %s", interval, otm_up)
    logger.info("select_otm_strike_below(points=%s) -> %s", interval, otm_dn)

    # Minimal invariants
    assert isinstance(atm, float)
    logger.info("Smoke test PASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
