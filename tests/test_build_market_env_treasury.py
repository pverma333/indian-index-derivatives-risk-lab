from __future__ import annotations

# Allow running this file directly via:
#   python tests/test_build_market_env_treasury.py
# without requiring `pip install -e .` or PYTHONPATH hacks.
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
import pytest

from src.data.build_market_env_treasury import (
    BuildConfig,
    build_market_data,
    build_treasury_curve,
    load_dividend_yields,
    percent_to_decimal,
    write_parquet_stable,
)


def test_percent_to_decimal_handles_numeric_and_percent_sign():
    s = pd.Series(["7.123", "7.123%", None, ""])
    out = percent_to_decimal(s, ctx="test")
    assert out.dtype == "float64"
    assert abs(out.iloc[0] - 0.07123) < 1e-12
    assert abs(out.iloc[1] - 0.07123) < 1e-12
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])


def test_market_ffill_limit_enforced_for_vix():
    dates = pd.date_range("2020-01-01", periods=7, freq="D").strftime("%Y-%m-%d")
    spot = pd.DataFrame({"date": dates, "index_name": ["NIFTY"] * 7, "spot_close": range(100, 107)})

    vix = pd.DataFrame({"date": [dates[0]], "vix_close": [12.0]})
    div = pd.DataFrame({"date": [dates[0]], "index_name": ["NIFTY"], "div_yield": [0.01]})

    class Cfg:
        vix_ffill_limit_days = 5
        div_ffill_limit_days = 5

    with pytest.raises(ValueError, match="vix_close still has nulls"):
        build_market_data(spot, vix, div, Cfg())


def test_write_parquet_stable_bit_identical_same_env(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-01"],
            "index_name": ["NIFTY", "NIFTY"],
            "spot_close": [101.0, 100.0],
            "vix_close": [12.5, 12.0],
            "div_yield": [0.01, 0.01],
        }
    )

    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"

    write_parquet_stable(
        df,
        p1,
        sort_cols=["date", "index_name"],
        column_types={
            "date": "string",
            "index_name": "string",
            "spot_close": "float64",
            "vix_close": "float64",
            "div_yield": "float64",
        },
    )
    write_parquet_stable(
        df,
        p2,
        sort_cols=["date", "index_name"],
        column_types={
            "date": "string",
            "index_name": "string",
            "spot_close": "float64",
            "vix_close": "float64",
            "div_yield": "float64",
        },
    )

    assert p1.read_bytes() == p2.read_bytes()


def test_load_dividend_yields_handles_trailing_space_headers(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fp = raw_dir / "NIFTY 50-yield-01-01-2020-to-31-12-2020.csv"
    df = pd.DataFrame(
        {
            "Index Name ": ["NIFTY 50", "NIFTY 50"],
            "Date ": ["01-Jan-2020", "02-Jan-2020"],
            "P/E ": [20.0, 20.1],
            "P/B ": [3.0, 3.1],
            "Div Yield% ": ["1.23%", "1.25%"],
        }
    )
    df.to_csv(fp, index=False)

    out = load_dividend_yields(raw_dir)
    assert set(out.columns) == {"date", "index_name", "div_yield"}
    assert out["date"].tolist() == ["2020-01-01", "2020-01-02"]
    assert out["index_name"].unique().tolist() == ["NIFTY"]
    assert abs(out["div_yield"].iloc[0] - 0.0123) < 1e-12
    assert abs(out["div_yield"].iloc[1] - 0.0125) < 1e-12


def test_treasury_curve_starts_at_first_treasury_date_no_backfill(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["3-MonBondYield.csv", "6-MonBondYield.csv", "1-YearBondYield.csv"]:
        pd.DataFrame(
            {
                "Date": ["01-07-2019", "02-07-2019"],
                "Price": [7.0, 7.1],  # percent -> decimal
            }
        ).to_csv(raw_dir / fname, index=False)

    cfg = BuildConfig(raw_dir=raw_dir, processed_dir=tmp_path)

    out = build_treasury_curve(raw_dir, spot_min_date="2019-01-01", spot_max_date="2019-07-05", cfg=cfg)

    assert out["date"].min() == "2019-07-01"
    assert out["date"].max() == "2019-07-05"
    assert set(out["tenor"].unique()) == {"91D", "182D", "364D"}
    assert out["rate"].isna().sum() == 0
    assert abs(out["rate"].min() - 0.07) < 1e-12


if __name__ == "__main__":
    # Convenience: allow direct execution for a quick local check.
    # Proper usage is still `pytest -q`.
    raise SystemExit(pytest.main([__file__]))
