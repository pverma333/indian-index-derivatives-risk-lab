import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.expiry_selectors import build_expiry_cycles  # noqa: E402


def _df(rows):
    df = pd.DataFrame(rows)
    # Keep input realistic: dates as strings are allowed; function normalizes.
    return df


def test_build_expiry_cycles_monthly():
    market_df = _df(
        [
            {
                "date": "2025-01-10",
                "symbol": "NIFTY",
                "expiry_dt": "2025-01-30",
                "is_trading_day": True,
                "is_opt_weekly_expiry": False,
                "is_opt_monthly_expiry": True,
            },
            {
                "date": "2025-01-31",
                "symbol": "NIFTY",
                "expiry_dt": "2025-02-27",
                "is_trading_day": True,
                "is_opt_weekly_expiry": False,
                "is_opt_monthly_expiry": False,
            },
            {
                "date": "2025-02-28",
                "symbol": "NIFTY",
                "expiry_dt": "2025-03-27",
                "is_trading_day": True,
                "is_opt_weekly_expiry": False,
                "is_opt_monthly_expiry": False,
            },
            {
                "date": "2025-02-10",
                "symbol": "NIFTY",
                "expiry_dt": "2025-02-27",
                "is_trading_day": True,
                "is_opt_weekly_expiry": False,
                "is_opt_monthly_expiry": True,
            },
        ]
    )

    cycles = build_expiry_cycles(market_df, symbol="NIFTY", tenor="MONTHLY")
    assert list(cycles.columns) == ["symbol", "tenor", "expiry_dt", "entry_date", "exit_date"]
    assert len(cycles) == 2

    # expiry 2025-01-30 -> next trading day is 2025-01-31
    row0 = cycles.iloc[0]
    assert row0["tenor"] == "MONTHLY"
    assert pd.Timestamp(row0["expiry_dt"]).date().isoformat() == "2025-01-30"
    assert pd.Timestamp(row0["entry_date"]).date().isoformat() == "2025-01-31"
    assert pd.Timestamp(row0["exit_date"]).date().isoformat() == "2025-01-30"

    # expiry 2025-02-27 -> next trading day is 2025-02-28
    row1 = cycles.iloc[1]
    assert pd.Timestamp(row1["expiry_dt"]).date().isoformat() == "2025-02-27"
    assert pd.Timestamp(row1["entry_date"]).date().isoformat() == "2025-02-28"


def test_build_expiry_cycles_weekly():
    market_df = _df(
        [
            {
                "date": "2025-01-06",
                "symbol": "NIFTY",
                "expiry_dt": "2025-01-09",
                "is_trading_day": True,
                "is_opt_weekly_expiry": True,
                "is_opt_monthly_expiry": False,
            },
            {
                "date": "2025-01-10",
                "symbol": "NIFTY",
                "expiry_dt": "2025-01-16",
                "is_trading_day": True,
                "is_opt_weekly_expiry": False,
                "is_opt_monthly_expiry": False,
            },
        ]
    )

    cycles = build_expiry_cycles(market_df, symbol="NIFTY", tenor="WEEKLY")
    assert len(cycles) == 1
    row = cycles.iloc[0]
    assert row["tenor"] == "WEEKLY"
    assert pd.Timestamp(row["expiry_dt"]).date().isoformat() == "2025-01-09"
    assert pd.Timestamp(row["entry_date"]).date().isoformat() == "2025-01-10"
    assert pd.isna(cycles["entry_date"]).sum() == 0


def test_build_expiry_cycles_missing_entry_date_drops(caplog):
    market_df = _df(
        [
            {
                "date": "2025-03-20",
                "symbol": "NIFTY",
                "expiry_dt": "2025-03-27",
                "is_trading_day": True,
                "is_opt_weekly_expiry": True,
                "is_opt_monthly_expiry": False,
            },
            # Note: no trading dates after 2025-03-27 in this dataset
        ]
    )

    with caplog.at_level("WARNING"):
        cycles = build_expiry_cycles(market_df, symbol="NIFTY", tenor="WEEKLY")

    assert len(cycles) == 0

    # Must log drop reason and include symbol/tenor/expiry_dt
    msgs = "\n".join(r.message for r in caplog.records)
    assert "MISSING_ENTRY_DATE" in msgs
    assert "symbol=NIFTY" in msgs
    assert "tenor=WEEKLY" in msgs
    assert "expiry_dt=2025-03-27" in msgs
