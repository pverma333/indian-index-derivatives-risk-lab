from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make `from src...` work when running from repo root or as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_curated_derivatives import ETLConfig, build_curated_derivatives  # noqa: E402


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_curated_derivatives_core_invariants(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    curated = tmp_path / "data" / "curated"
    processed.mkdir(parents=True, exist_ok=True)
    curated.mkdir(parents=True, exist_ok=True)

    # Trade calendar (no 2025-01-04 to simulate weekend/holiday)
    cal_days = pd.to_datetime(
        ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]
    )
    cal = pd.DataFrame(
        {
            "TradeDate": cal_days,
            "Symbol": ["NIFTY"] * len(cal_days),
            # extra cols are fine; your loader ignores them
            "Fut_Near_Expiry": pd.to_datetime(["2025-01-30"] * len(cal_days)),
            "Fut_Next_Expiry": pd.to_datetime(["2025-02-27"] * len(cal_days)),
            "Fut_Far_Expiry": pd.to_datetime(["2025-03-27"] * len(cal_days)),
            "Opt_Weekly_Expiry": pd.to_datetime(["2025-01-09"] * len(cal_days)),
            "Opt_Monthly_Expiry": pd.to_datetime(["2025-01-30"] * len(cal_days)),
        }
    )
    _write_parquet(cal, processed / "trade_calendar.parquet")

    # Market data includes weekend date too (2025-01-04)
    market = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-04"],
            "index_name": ["NIFTY", "NIFTY"],
            "spot_close": [20100.0, 20300.0],
            "vix_close": [13.5, 14.2],
            "div_yield": [0.012, 0.012],
        }
    )
    _write_parquet(market, processed / "market_data.parquet")

    # Lot size (current mapping)
    lot = pd.DataFrame(
        {
            "symbol": ["NIFTY"],
            "start_date": pd.to_datetime(["2020-01-01"]),
            "end_date": pd.to_datetime([None]),
            "lot_size": [50],
        }
    )
    _write_parquet(lot, processed / "lot_size_map.parquet")

    # Treasury curve (percent-like, script should convert to decimals)
    curve = pd.DataFrame(
        {
            "date": ["2025-01-02"] * 3 + ["2025-01-04"] * 3,
            "tenor": ["91D", "182D", "364D"] * 2,
            "rate": [7.0, 7.2, 7.5, 7.0, 7.2, 7.5],
        }
    )
    _write_parquet(curve, processed / "treasury_curve.parquet")

    # Derivatives CSV (ISO dates only for determinism)
    deriv = pd.DataFrame(
        {
            "INSTRUMENT": ["FUTIDX", "FUTIDX", "OPTIDX", "OPTIDX", "OPTIDX"],
            "SYMBOL": ["NIFTY", "NIFTY", "NIFTY", "NIFTY", "NIFTY"],
            "EXPIRY_DT": ["2025-01-30", "2025-02-27", "2025-01-09", "2025-01-30", "2025-01-09"],
            "STRIKE_PR": ["99999", "88888", "20000", "21000", "20200"],
            "OPTION_TYP": ["", "", "CE", "PE", "CE"],
            "OPEN": ["1", "1", "100", "120", "50"],
            "HIGH": ["1", "1", "110", "130", "60"],
            "LOW": ["1", "1", "90", "115", "40"],
            "CLOSE": ["1", "1", "105", "125", "55"],
            "SETTLE_PR": ["1", "1", "106", "126", "56"],
            "CONTRACTS": ["1", "1", "2", "3", "4"],
            "OPEN_INT": ["10", "10", "20", "30", "40"],
            "CHG_IN_OI": ["0", "0", "1", "-1", "2"],
            "TIMESTAMP": ["2025-01-02", "2025-01-02", "2025-01-02", "2025-01-02", "2025-01-04"],
        }
    )
    deriv_path = processed / "Nifty_Historical_Derivatives.csv"
    deriv.to_csv(deriv_path, index=False)

    cfg = ETLConfig(
        derivatives_csv=deriv_path,
        market_data_parquet=processed / "market_data.parquet",
        lot_size_map_parquet=processed / "lot_size_map.parquet",
        treasury_curve_parquet=processed / "treasury_curve.parquet",
        trade_calendar_parquet=processed / "trade_calendar.parquet",
        output_parquet=curated / "derivatives_clean.parquet",
        chunksize=50,
    )

    out = build_curated_derivatives(cfg)
    assert len(out) == 5

    # 1) Futures strike forced to 0.0
    fut = out[out["instrument"] == "FUTIDX"]
    assert (fut["strike_pr"] == 0.0).all()

    # 2) Required joins not null
    for col in ["spot_close", "vix_close", "lot_size", "rate_91d", "rate_182d", "rate_364d"]:
        assert int(out[col].isna().sum()) == 0

    # 3) Treasury rates are decimals
    assert (out["rate_91d"] < 1.0).all()
    assert pytest.approx(out.loc[out["date"] == pd.Timestamp("2025-01-02"), "rate_91d"].iloc[0], 1e-9) == 0.07

    # 4) expiry_rank dense ascending per (date, symbol, instrument)
    fut_0102 = out[(out["date"] == pd.Timestamp("2025-01-02")) & (out["instrument"] == "FUTIDX")].sort_values("expiry_dt")
    assert fut_0102["expiry_rank"].tolist() == [1, 2]

    opt_0102 = out[(out["date"] == pd.Timestamp("2025-01-02")) & (out["instrument"] == "OPTIDX")].sort_values("expiry_dt")
    assert opt_0102["expiry_rank"].tolist() == [1, 2]

    # 5) moneyness directional
    call = out[(out["date"] == pd.Timestamp("2025-01-02")) & (out["instrument"] == "OPTIDX") & (out["option_typ"] == "CE")].iloc[0]
    put = out[(out["date"] == pd.Timestamp("2025-01-02")) & (out["instrument"] == "OPTIDX") & (out["option_typ"] == "PE")].iloc[0]
    assert call["moneyness"] == pytest.approx(20100.0 - 20000.0)  # +100
    assert put["moneyness"] == pytest.approx(21000.0 - 20100.0)   # +900

    # 6) weekend/holiday flag + trading_days_to_expiry
    wk = out[(out["date"] == pd.Timestamp("2025-01-04")) & (out["instrument"] == "OPTIDX")].iloc[0]
    # IMPORTANT: cast to python bool, don't use "is False" on numpy.bool_
    assert bool(wk["is_trading_day"]) is False
    assert bool(wk["is_trade_calendar_date"]) is False
    assert int(wk["trading_days_to_expiry"]) == 4  # Jan06,07,08,09

    # 7) calendar TTE non-negative
    assert (out["cal_days_to_expiry"] >= 0).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
