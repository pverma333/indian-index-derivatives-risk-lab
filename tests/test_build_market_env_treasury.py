from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
import pytest
import numpy as np

from src.data.build_market_env_treasury import (
    BuildConfig,
    build_market_data,
    build_treasury_curve,
    load_dividend_yields,
    percent_to_decimal,
    write_parquet_stable,
)

# ... (Existing tests for percent_to_decimal, market_ffill, and parquet_stable remain the same) ...

def test_treasury_curve_enforces_july_2019_start(tmp_path: Path):
    """
    Verify that the treasury curve strictly adheres to the July 1, 2019 start
    date logic even if spot data starts earlier.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create mock data starting exactly on 2019-07-01
    for fname in ["3-MonBondYield.csv", "6-MonBondYield.csv", "1-YearBondYield.csv"]:
        pd.DataFrame(
            {
                "Date": ["01-07-2019", "02-07-2019"],
                "Price": [6.5, 6.6],  # Yield range (0-15)
            }
        ).to_csv(raw_dir / fname, index=False)

    cfg = BuildConfig(raw_dir=raw_dir, processed_dir=tmp_path)

    # Provide spot dates starting in Jan 2019
    out = build_treasury_curve(raw_dir, spot_min_date="2019-01-01", spot_max_date="2019-07-05", cfg=cfg)

    # The curve MUST start at 2019-07-01 because target_start is 2019-07-01
    assert out["date"].min() == "2019-07-01"
    assert out["date"].max() == "2019-07-05"
    # Check that 6.5 was converted to 0.065
    assert abs(out.loc[out["date"] == "2019-07-01", "rate"].iloc[0] - 0.065) < 1e-12


def test_treasury_curve_handles_price_vs_yield_ranges(tmp_path: Path):
    """
    Test the logic that converts high values (Prices 60-105)
    and low values (Yields 0-15) to decimal yields.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 91D has Price data (98.5), 364D has Yield data (6.0)
    pd.DataFrame({
        "Date": ["01-07-2019"],
        "Price": ["98.5"]
    }).to_csv(raw_dir / "3-MonBondYield.csv", index=False)

    pd.DataFrame({
        "Date": ["01-07-2019"],
        "Price": ["6.0"]
    }).to_csv(raw_dir / "1-YearBondYield.csv", index=False)

    # Fill the missing required file for the loop
    pd.DataFrame({"Date": ["01-07-2019"], "Price": ["6.0"]}).to_csv(raw_dir / "6-MonBondYield.csv", index=False)

    cfg = BuildConfig(raw_dir=raw_dir, processed_dir=tmp_path)
    out = build_treasury_curve(raw_dir, spot_min_date="2019-07-01", spot_max_date="2019-07-01", cfg=cfg)

    # Check 91D: (100 - 98.5) / 100 = 0.015
    rate_91d = out.loc[out["tenor"] == "91D", "rate"].iloc[0]
    assert abs(rate_91d - 0.015) < 1e-12

    # Check 364D: 6.0 / 100 = 0.060
    rate_364d = out.loc[out["tenor"] == "364D", "rate"].iloc[0]
    assert abs(rate_364d - 0.060) < 1e-12


def test_treasury_curve_rejects_out_of_range_values(tmp_path: Path):
    """
    Values outside 0-15 or 60-105 should result in NaNs and be forward-filled.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create a sequence: Good -> Out-of-range (40.0) -> Good
    for fname in ["3-MonBondYield.csv", "6-MonBondYield.csv", "1-YearBondYield.csv"]:
        pd.DataFrame({
            "Date": ["01-07-2019", "02-07-2019", "03-07-2019"],
            "Price": [7.0, 40.0, 7.2]
        }).to_csv(raw_dir / fname, index=False)

    cfg = BuildConfig(raw_dir=raw_dir, processed_dir=tmp_path)
    out = build_treasury_curve(raw_dir, spot_min_date="2019-07-01", spot_max_date="2019-07-03", cfg=cfg)

    # Date 02-07-2019 should have been ffilled from 01-07-2019 (0.07)
    mid_rate = out.loc[out["date"] == "2019-07-02", "rate"].unique()
    assert len(mid_rate) == 1
    assert abs(mid_rate[0] - 0.07) < 1e-12
