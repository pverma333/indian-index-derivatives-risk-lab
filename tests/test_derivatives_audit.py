"""
Single-file test runner for src/data_validation/derivatives_audit.py

Usage:
  python tests/test_derivatives_audit_onefile.py

This script:
- Adds repo_root/src to sys.path so `data_validation` is importable
- Runs deterministic assertion-based tests
- Exits with non-zero code on failure
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pandas as pd


def _add_src_to_syspath() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_add_src_to_syspath()

# Now imports should work
from data_validation.derivatives_audit import (  # noqa: E402
    AuditConfig,
    assert_no_duplicate_contract_keys,
    compute_basis_near_futures,
    validate_expiry_rank_near,
    validate_futures_strike_zero,
    validate_moneyness_polarity,
)


def _base_df() -> pd.DataFrame:
    """
    Build a tiny deterministic dataframe that matches the required columns for the tested functions.

    Notes:
    - Includes FUTIDX strike_pr==0.0
    - Includes OPTIDX CE/PE with moneyness defined exactly as the audit expects
    - Uses expiry_rank==1 and only one expiry_dt for rank check
    """
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-02"]),
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-02"]),
            "symbol": ["NIFTY", "NIFTY", "NIFTY", "NIFTY"],
            "instrument": ["FUTIDX", "OPTIDX", "OPTIDX", "FUTIDX"],
            "expiry_dt": pd.to_datetime(["2025-01-30", "2025-01-30", "2025-01-30", "2025-01-30"]),
            "strike_pr": [0.0, 22000.0, 22000.0, 0.0],
            "option_typ": [None, "CE", "PE", None],
            "settle_pr": [22110.0, 100.0, 120.0, 22120.0],
            "close": [22105.0, 100.0, 120.0, 22115.0],
            "open": [0.0, 0.0, 0.0, 0.0],
            "high": [0.0, 0.0, 0.0, 0.0],
            "low": [0.0, 0.0, 0.0, 0.0],
            "contracts": [1.0, 1.0, 1.0, 1.0],
            "open_int": [1.0, 1.0, 1.0, 1.0],
            "chg_in_oi": [0.0, 0.0, 0.0, 0.0],
            "spot_close": [22100.0, 22100.0, 22100.0, 22110.0],
            "vix_close": [14.0, 14.0, 14.0, 14.1],
            "lot_size": [25, 25, 25, 25],
            "rate_91d": [0.07, 0.07, 0.07, 0.07],
            "rate_182d": [0.071, 0.071, 0.071, 0.071],
            "rate_364d": [0.072, 0.072, 0.072, 0.072],
            "cal_days_to_expiry": [29, 29, 29, 28],
            "trading_days_to_expiry": [20, 20, 20, 19],
            "expiry_rank": [1, 1, 1, 1],
            # moneyness definition enforced by audit:
            # CE: spot - strike; PE: strike - spot
            "moneyness": [
                np.nan,
                22100.0 - 22000.0,
                22000.0 - 22100.0,
                np.nan,
            ],
            "opt_weekly_expiry": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
            "opt_monthly_expiry": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
            "is_trade_calendar_date": [True, True, True, True],
            "is_trading_day": [True, True, True, True],
            "is_opt_weekly_expiry": [False, False, False, False],
            "is_opt_monthly_expiry": [False, False, False, False],
        }
    )


def test_no_duplicate_contract_key_passes() -> None:
    df = _base_df()
    cfg = AuditConfig()
    # Should not raise
    assert_no_duplicate_contract_keys(df, cfg.unique_contract_key)


def test_futures_strike_zero_passes() -> None:
    df = _base_df()
    validate_futures_strike_zero(df)


def test_moneyness_polarity_passes() -> None:
    df = _base_df()
    validate_moneyness_polarity(df)


def test_expiry_rank_near_passes() -> None:
    df = _base_df()
    validate_expiry_rank_near(df)


def test_basis_computation() -> None:
    df = _base_df()
    cfg = AuditConfig()
    fut_near = compute_basis_near_futures(df, cfg)

    assert "basis" in fut_near.columns
    fut = fut_near[fut_near["instrument"] == "FUTIDX"]
    assert len(fut) == 2

    # Row 0: 22110 - 22100 = 10
    assert np.isclose(float(fut.iloc[0]["basis"]), 22110.0 - 22100.0)
    # Row 3: 22120 - 22110 = 10
    assert np.isclose(float(fut.iloc[1]["basis"]), 22120.0 - 22110.0)


def _run_all() -> None:
    tests = [
        test_no_duplicate_contract_key_passes,
        test_futures_strike_zero_passes,
        test_moneyness_polarity_passes,
        test_expiry_rank_near_passes,
        test_basis_computation,
    ]

    failures = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()

    if failures:
        raise SystemExit(f"\n{failures} test(s) failed.")
    print("\nAll tests passed.")


if __name__ == "__main__":
    _run_all()
