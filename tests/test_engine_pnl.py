import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import src...` works even when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.strategies.engine_pnl import BaseStrategy, BaseStrategyConfig, DataIntegrityError


class DummyStrategy(BaseStrategy):
    def build_trades(self, market_df: pd.DataFrame, entry_days: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Not needed for unit tests.")


def _make_market_df(
    dates,
    settle,
    close,
    lot_size=50,
    symbol="NIFTY",
    expiry_dt="2023-02-23",
    instrument="OPTIDX",
    strike_pr=18000.0,
    option_typ="CE",
    expiry_rank=1,
    is_trading_day=True,
    is_opt_monthly_expiry=False,
):
    n = len(dates)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates).normalize(),
            "symbol": [symbol] * n,
            "instrument": [instrument] * n,
            "expiry_dt": pd.to_datetime([expiry_dt] * n).normalize(),
            "strike_pr": [strike_pr] * n,
            "option_typ": [option_typ] * n,
            "close": close,
            "settle_pr": settle,
            "lot_size": [lot_size] * n,
            "expiry_rank": [expiry_rank] * n,
            "is_trading_day": [is_trading_day] * n,
            "is_opt_monthly_expiry": [is_opt_monthly_expiry] * n,
        }
    )


def test_long_call_one_month_vectorized_pnl_runs_and_matches_expected():
    dates = ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-02-23"]
    settle = [100.0, 110.0, 105.0, 120.0, 130.0]
    close =  [100.0, 110.0, 105.0, 120.0, 130.0]

    market = _make_market_df(dates, settle=settle, close=close, expiry_dt="2023-02-23", lot_size=50)

    trades = pd.DataFrame(
        {
            "trade_id": ["T1"],
            "symbol": ["NIFTY"],
            "instrument": ["OPTIDX"],
            "expiry_dt": [pd.to_datetime("2023-02-23")],
            "strike_pr": [18000.0],
            "option_typ": ["CE"],
            "entry_date": [pd.to_datetime("2023-01-02")],
            "position_sign": [1],
        }
    )

    engine = DummyStrategy(BaseStrategyConfig(input_parquet_path="unused"))
    out = engine.compute_mtm_pnl_rupee(market, trades)

    # Expected:
    # Day0: (100-100)*50 = 0
    # Day1: (110-100)*50 = 500
    # Day2: (105-110)*50 = -250
    # Day3: (120-105)*50 = 750
    # Expiry: (130-120)*50 = 500
    expected_daily = np.array([0.0, 500.0, -250.0, 750.0, 500.0])
    expected_cum = np.cumsum(expected_daily)

    assert out["daily_pnl_rupee"].to_numpy().tolist() == expected_daily.tolist()
    assert out["cum_pnl_rupee"].to_numpy().tolist() == expected_cum.tolist()
    assert out["date"].max() == pd.to_datetime("2023-02-23")


def test_short_position_positive_when_settle_decreases():
    dates = ["2023-01-02", "2023-01-03", "2023-01-04", "2023-02-23"]
    settle = [200.0, 190.0, 180.0, 175.0]
    close =  [200.0, 190.0, 180.0, 175.0]

    market = _make_market_df(dates, settle=settle, close=close, expiry_dt="2023-02-23", lot_size=50)

    trades = pd.DataFrame(
        {
            "trade_id": ["T2"],
            "symbol": ["NIFTY"],
            "instrument": ["OPTIDX"],
            "expiry_dt": [pd.to_datetime("2023-02-23")],
            "strike_pr": [18000.0],
            "option_typ": ["CE"],
            "entry_date": [pd.to_datetime("2023-01-02")],
            "position_sign": [-1],
        }
    )

    engine = DummyStrategy(BaseStrategyConfig(input_parquet_path="unused"))
    out = engine.compute_mtm_pnl_rupee(market, trades)

    # For short: -1 * (settle_t - settle_{t-1}) * lot_size > 0 when settle decreases
    assert out.loc[out["date"] == pd.to_datetime("2023-01-03"), "daily_pnl_rupee"].iloc[0] > 0
    assert out.loc[out["date"] == pd.to_datetime("2023-01-04"), "daily_pnl_rupee"].iloc[0] > 0
    assert out.loc[out["date"] == pd.to_datetime("2023-02-23"), "daily_pnl_rupee"].iloc[0] > 0


def test_lot_size_change_raises_data_integrity_error():
    dates = ["2023-01-02", "2023-01-03", "2023-02-23"]
    settle = [100.0, 101.0, 102.0]
    close =  [100.0, 101.0, 102.0]

    market = _make_market_df(dates, settle=settle, close=close, expiry_dt="2023-02-23", lot_size=50)
    market.loc[market["date"] == pd.to_datetime("2023-01-03"), "lot_size"] = 25

    trades = pd.DataFrame(
        {
            "trade_id": ["T3"],
            "symbol": ["NIFTY"],
            "instrument": ["OPTIDX"],
            "expiry_dt": [pd.to_datetime("2023-02-23")],
            "strike_pr": [18000.0],
            "option_typ": ["CE"],
            "entry_date": [pd.to_datetime("2023-01-02")],
            "position_sign": [1],
        }
    )

    engine = DummyStrategy(BaseStrategyConfig(input_parquet_path="unused"))
    with pytest.raises(DataIntegrityError):
        engine.compute_mtm_pnl_rupee(market, trades)


if __name__ == "__main__":
    # Allows: python tests/test_engine_pnl.py
    raise SystemExit(pytest.main([__file__]))
