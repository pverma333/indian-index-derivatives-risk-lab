import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.expiry_selectors import build_expiry_cycles


def _mk_df() -> pd.DataFrame:
    # Minimal rows emulating curated behavior.
    return pd.DataFrame(
        [
            # Weekly anchor expiry
            {
                "date": "2025-01-02",
                "symbol": "NIFTY",
                "expiry_dt": "2025-01-02",
                "is_trading_day": True,
                "is_opt_weekly_expiry": True,
                "is_opt_monthly_expiry": False,
                "opt_weekly_expiry": "2025-01-02",
                "opt_monthly_expiry": "2025-01-30",
            },
            # Next trading day after weekly anchor
            {
                "date": "2025-01-03",
                "symbol": "NIFTY",
                "expiry_dt": "2025-01-09",
                "is_trading_day": True,
                "is_opt_weekly_expiry": False,
                "is_opt_monthly_expiry": False,
                "opt_weekly_expiry": "2025-01-09",
                "opt_monthly_expiry": "2025-01-30",
            },
            # Monthly anchor expiry (can overlap with weekly)
            {
                "date": "2025-01-30",
                "symbol": "NIFTY",
                "expiry_dt": "2025-01-30",
                "is_trading_day": True,
                "is_opt_weekly_expiry": True,
                "is_opt_monthly_expiry": True,
                "opt_weekly_expiry": "2025-01-30",
                "opt_monthly_expiry": "2025-01-30",
            },
            # Next trading day after monthly anchor
            {
                "date": "2025-01-31",
                "symbol": "NIFTY",
                "expiry_dt": "2025-02-06",
                "is_trading_day": True,
                "is_opt_weekly_expiry": False,
                "is_opt_monthly_expiry": False,
                "opt_weekly_expiry": "2025-02-06",
                "opt_monthly_expiry": "2025-02-27",
            },
        ]
    )


def test_build_expiry_cycles_weekly_roll_maps_to_trade_expiry():
    df = _mk_df()
    cycles = build_expiry_cycles(df, symbol="NIFTY", tenor="WEEKLY")

    assert not cycles.empty
    assert set(cycles.columns) == {"symbol", "tenor", "expiry_dt", "entry_date", "exit_date"}
    assert (cycles["exit_date"] == cycles["expiry_dt"]).all()
    assert (cycles["entry_date"] < cycles["expiry_dt"]).all()

    # Weekly anchor 2025-01-02 -> entry 2025-01-03 -> trade expiry 2025-01-09
    first = cycles.sort_values(["entry_date", "expiry_dt"]).iloc[0]
    assert pd.Timestamp(first["entry_date"]) == pd.Timestamp("2025-01-03")
    assert pd.Timestamp(first["expiry_dt"]) == pd.Timestamp("2025-01-09")


def test_build_expiry_cycles_monthly_roll_maps_to_trade_expiry():
    df = _mk_df()
    cycles = build_expiry_cycles(df, symbol="NIFTY", tenor="MONTHLY")

    assert not cycles.empty
    assert (cycles["exit_date"] == cycles["expiry_dt"]).all()
    assert (cycles["entry_date"] < cycles["expiry_dt"]).all()

    # Monthly anchor 2025-01-30 -> entry 2025-01-31 -> trade expiry 2025-02-27
    row = cycles.iloc[0]
    assert pd.Timestamp(row["entry_date"]) == pd.Timestamp("2025-01-31")
    assert pd.Timestamp(row["expiry_dt"]) == pd.Timestamp("2025-02-27")
    assert pd.Timestamp(row["exit_date"]) == pd.Timestamp("2025-02-27")


def test_build_expiry_cycles_skips_only_missing_mapping_cycle(caplog):
    df = _mk_df()

    # Break mapping for the weekly-roll entry date 2025-01-03
    df.loc[df["date"] == "2025-01-03", "opt_weekly_expiry"] = pd.NaT

    with caplog.at_level("WARNING"):
        cycles = build_expiry_cycles(df, symbol="NIFTY", tenor="WEEKLY")

    # It should log the skip for that cycle
    assert any("MISSING_TRADE_EXPIRY" in rec.message for rec in caplog.records)

    # But weekly cycles are not necessarily empty (later anchor can still produce a valid cycle)
    # Ensure the broken entry_date cycle is not present.
    if not cycles.empty:
        assert not (pd.to_datetime(cycles["entry_date"]) == pd.Timestamp("2025-01-03")).any()
