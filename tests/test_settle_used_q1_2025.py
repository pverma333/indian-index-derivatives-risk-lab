import sys
from pathlib import Path

# Ensures imports work regardless of where pytest is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../<repo_root>
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.engine.settlement_marking import (
    compute_settle_used,
    PRICE_METHOD_INTRINSIC,
    PRICE_METHOD_SETTLE,
)


@pytest.mark.integration
def test_settle_used_on_real_data_jan_2025():
    """
    Integration test:
    Uses real curated file:
      data/curated/derivatives_clean_Q1_2025.csv
    Filters:
      2025-01-01 to 2025-01-31 (inclusive)
    Validates:
      - settle_used exists, non-negative
      - price_method values are valid
      - SETTLE_PR rows: settle_used == settle_pr
      - INTRINSIC_ON_EXPIRY rows: OPTIDX expiry and settle_used matches intrinsic CE/PE
    Prints a summary when run with -s.
    """
    data_path = _PROJECT_ROOT / "data" / "curated" / "derivatives_clean_Q1_2025.csv"
    if not data_path.exists():
        pytest.skip(f"Real data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Detect date column robustly (common variants).
    date_candidates = ["date", "trade_date", "as_of_date", "dt", "timestamp"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise AssertionError(
            "Could not find a date column in the real data. "
            f"Tried {date_candidates}. Available cols sample: {list(df.columns)[:30]}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise AssertionError(f"Failed to parse any dates in column '{date_col}'")

    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2025-01-31")
    df_jan = df.loc[(df[date_col] >= start) & (df[date_col] <= end)].copy()

    if df_jan.empty:
        pytest.skip(f"No rows found between {start.date()} and {end.date()} in {data_path.name}")

    required = ["instrument", "option_typ", "strike_pr", "settle_pr", "spot_close", "cal_days_to_expiry"]
    missing = [c for c in required if c not in df_jan.columns]
    if missing:
        raise AssertionError(
            f"Real data missing required columns needed by compute_settle_used: {missing}. "
            "This test expects the curated file to match the market_df schema for the engine."
        )

    out = compute_settle_used(df_jan)

    assert "settle_used" in out.columns
    assert "price_method" in out.columns

    allowed_methods = {PRICE_METHOD_SETTLE, PRICE_METHOD_INTRINSIC}
    bad_methods = set(out["price_method"].dropna().unique()) - allowed_methods
    assert not bad_methods, f"Unexpected price_method values: {bad_methods}"

    assert (out["settle_used"].astype(float) >= 0).all(), "Found negative settle_used in real data output"

    # SETTLE_PR rows must match settle_pr
    settle_rows = out["price_method"] == PRICE_METHOD_SETTLE
    if settle_rows.any():
        a = out.loc[settle_rows, "settle_used"].astype(float).to_numpy()
        b = out.loc[settle_rows, "settle_pr"].astype(float).to_numpy()
        assert np.allclose(a, b, equal_nan=False), "SETTLE_PR rows: settle_used != settle_pr"

    # Intrinsic rows must match intrinsic value and be OPTIDX expiry
    intrinsic_rows = out["price_method"] == PRICE_METHOD_INTRINSIC
    if intrinsic_rows.any():
        subset = out.loc[intrinsic_rows].copy()

        assert (subset["instrument"].astype(str) == "OPTIDX").all(), "Intrinsic rows contain non-OPTIDX instruments"
        assert (subset["cal_days_to_expiry"] == 0).all(), "Intrinsic rows contain non-expiry rows"

        ce = subset["option_typ"].astype(str) == "CE"
        pe = subset["option_typ"].astype(str) == "PE"
        assert (ce | pe).all(), "Intrinsic rows contain option_typ not in {CE, PE}"

        spot = subset["spot_close"].astype(float).to_numpy()
        strike = subset["strike_pr"].astype(float).to_numpy()
        expected = np.where(ce.to_numpy(), spot - strike, strike - spot)
        expected = np.maximum(expected, 0.0)

        actual = subset["settle_used"].astype(float).to_numpy()
        assert np.allclose(actual, expected, equal_nan=False), "Intrinsic rows: settle_used != expected intrinsic"

    # Actual result - run with: pytest -s
    print("\n--- settle_used real-data summary (Jan 2025) ---")
    print(f"file: {data_path}")
    print(f"date_col: {date_col}, rows_in_range: {len(out)}")
    print(out["price_method"].value_counts(dropna=False).to_string())

    if intrinsic_rows.any():
        cols = ["instrument", "option_typ", "strike_pr", "spot_close", "settle_pr", "settle_used", "cal_days_to_expiry"]
        sample = out.loc[intrinsic_rows, cols].head(10)
        print("\nSample intrinsic rows (head 10):")
        print(sample.to_string(index=False))
