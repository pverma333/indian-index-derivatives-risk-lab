import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.expiry_selectors import build_expiry_cycles  # noqa: E402


REQUIRED_COLS = {
    "date",
    "symbol",
    "expiry_dt",
    "is_trading_day",
    "is_opt_weekly_expiry",
    "is_opt_monthly_expiry",
}


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    """
    Robust coercion for common encodings: True/False, 1/0, 'true'/'false', 'Y'/'N'.
    Unknowns become False (safe for flags).
    """
    if s.dtype == bool:
        return s

    # Handle numeric
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)

    # Handle strings / mixed
    truthy = {"true", "t", "1", "y", "yes"}
    falsy = {"false", "f", "0", "n", "no", "", "nan", "none"}

    def to_bool(x):
        if pd.isna(x):
            return False
        if isinstance(x, bool):
            return x
        v = str(x).strip().lower()
        if v in truthy:
            return True
        if v in falsy:
            return False
        # safest default for flags
        return False

    return s.map(to_bool).astype(bool)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(
            "CSV missing required columns.\n"
            f"Missing: {missing}\n"
            f"Found: {sorted(df.columns.tolist())}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test: build expiry cycles from Q1 2025 dataset")
    ap.add_argument(
        "--path",
        default="data/curated/derivatives_clean_Q1_2025.csv",
        help="Path to derivatives_clean_Q1_2025.csv",
    )
    ap.add_argument("--symbol", required=True, help="Symbol to test, e.g., NIFTY")
    ap.add_argument("--tenor", default="BOTH", choices=["WEEKLY", "MONTHLY", "BOTH"])
    ap.add_argument("--max_print", type=int, default=5, help="How many rows to print head/tail")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    _validate_columns(df)

    # Normalize types (matches build_expiry_cycles expectations)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["expiry_dt"] = pd.to_datetime(df["expiry_dt"], errors="coerce")

    for col in ["is_trading_day", "is_opt_weekly_expiry", "is_opt_monthly_expiry"]:
        df[col] = _coerce_bool_series(df[col])

    # Quick dataset sanity
    df_sym = df[df["symbol"] == args.symbol].copy()
    if df_sym.empty:
        raise ValueError(f"No rows found for symbol={args.symbol!r} in {path}")

    weekly_exp_cnt = int(df_sym["is_opt_weekly_expiry"].sum())
    monthly_exp_cnt = int(df_sym["is_opt_monthly_expiry"].sum())
    trading_cnt = int(df_sym["is_trading_day"].sum())
    print(f"[DATA] rows={len(df_sym):,} trading_days={trading_cnt:,} "
          f"weekly_flag_rows={weekly_exp_cnt:,} monthly_flag_rows={monthly_exp_cnt:,}")

    # Run
    cycles = build_expiry_cycles(df, symbol=args.symbol, tenor=args.tenor)

    # Summary
    if cycles.empty:
        print("[RESULT] cycles_df is EMPTY (this can happen if entry_date is missing for all expiries).")
        return 0

    print(f"[RESULT] cycles rows={len(cycles):,} | tenors={sorted(cycles['tenor'].unique().tolist())}")
    print("[RESULT] counts by tenor:")
    print(cycles["tenor"].value_counts().sort_index().to_string())

    # Assertions / invariants
    assert cycles["entry_date"].isna().sum() == 0, "Invariant failed: null entry_date present"
    assert (cycles["exit_date"] == cycles["expiry_dt"]).all(), "Invariant failed: exit_date != expiry_dt"
    assert (cycles["entry_date"] > cycles["expiry_dt"]).all(), "Invariant failed: entry_date not strictly after expiry_dt"

    # Sorted check
    sorted_copy = cycles.sort_values(["expiry_dt", "tenor"], kind="mergesort").reset_index(drop=True)
    assert sorted_copy.equals(cycles.reset_index(drop=True)), "Invariant failed: cycles_df not sorted by expiry_dt, tenor"

    # Display
    n = max(0, int(args.max_print))
    print("\n[HEAD]")
    print(cycles.head(n).to_string(index=False))
    print("\n[TAIL]")
    print(cycles.tail(n).to_string(index=False))

    print("\n[OK] Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
