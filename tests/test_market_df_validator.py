from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

# Add repo root so `import src....` works deterministically under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.exceptions import DataIntegrityError
from src.validation.market_df_validator import validate_market_df


def _base_market_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-02 15:30:00"),
                "symbol": "NIFTY",
                "instrument": "FUTIDX",
                "expiry_dt": pd.Timestamp("2025-01-30"),
                "strike_pr": 0.0,
                "option_typ": None,  # allowed in schema; validator coerces FUTIDX blank/null -> XX
                "is_opt_weekly_expiry": False,
                "is_opt_monthly_expiry": False,
                "cal_days_to_expiry": 28,
                "is_trading_day": True,
                "settle_pr": 100.0,
                "spot_close": 20000.0,
                "index_open_price": 19950.0,
                "lot_size": 50,
                "rate_182d": 0.07,
                "rate_364d": 0.072,
                "chg_in_oi": 0.0,
                "rate_91d": 0.069,
                "vix_close": 14.0,
                "contracts": 10.0,
                "open_int": 100.0,
            },
            {
                "date": pd.Timestamp("2025-01-02"),
                "symbol": "NIFTY",
                "instrument": "OPTIDX",
                "expiry_dt": pd.Timestamp("2025-01-30"),
                "strike_pr": 20000.0,
                "option_typ": "CE",
                "is_opt_weekly_expiry": False,
                "is_opt_monthly_expiry": False,
                "cal_days_to_expiry": 28,
                "is_trading_day": True,
                "settle_pr": 120.0,
                "spot_close": 20000.0,
                "index_open_price": 19950.0,
                "lot_size": 50,
                "rate_182d": 0.07,
                "rate_364d": 0.072,
                "chg_in_oi": 0.0,
                "rate_91d": 0.069,
                "vix_close": 14.0,
                "contracts": 10.0,
                "open_int": 100.0,
            },
        ]
    )


def test_validate_market_df_missing_required_columns_raises():
    df = _base_market_df().drop(columns=["settle_pr"])
    with pytest.raises(DataIntegrityError) as e:
        validate_market_df(df)
    assert "REQUIRED_COLUMNS_MISSING" in str(e.value)
    assert "settle_pr" in str(e.value)


def test_validate_market_df_option_typ_domain_raises_on_bad_values():
    df = _base_market_df()
    df.loc[df["instrument"] == "OPTIDX", "option_typ"] = "BAD"

    with pytest.raises(DataIntegrityError) as e:
        validate_market_df(df)

    msg = str(e.value)
    assert "OPTION_TYP_DOMAIN" in msg
    assert "violations=" in msg
    assert "First 20 violations" in msg
    for key in ["date", "symbol", "instrument", "expiry_dt", "option_typ", "strike_pr"]:
        assert key in msg


def test_validate_market_df_futidx_invariants():
    df = _base_market_df()
    df.loc[df["instrument"] == "FUTIDX", "strike_pr"] = 1.0

    with pytest.raises(DataIntegrityError) as e:
        validate_market_df(df)

    msg = str(e.value)
    assert "FUTIDX_INVARIANTS" in msg
    assert "violations=" in msg
    assert "First 20 violations" in msg


def test_validate_market_df_index_open_price_non_null():
    df = _base_market_df()
    df.loc[df["date"].astype(str).str.startswith("2025-01-02"), "index_open_price"] = np.nan

    # In-scope should fail
    with pytest.raises(DataIntegrityError) as e:
        validate_market_df(df.copy(deep=True), start_date="2025-01-02", end_date="2025-01-02")
    assert "INDEX_OPEN_PRICE_NULL_IN_SCOPE" in str(e.value)

    # Out-of-scope window should pass
    validate_market_df(df.copy(deep=True), start_date="2025-01-03", end_date="2025-01-03")
