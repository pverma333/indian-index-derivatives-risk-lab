from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure repo root is importable when running as: python tests/test_bull_call_spread.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strategies.bull_call_spread import (  # noqa: E402
    BullCallSpreadStrategy,
    BullCallSpreadStrategyConfig,
)


def _mk_row(
    *,
    date: str,
    expiry_dt: str,
    strike: int,
    settle: float,
    close: float,
    spot_close: float,
    is_monthly_expiry_day: bool,
    open_int: float = 100.0,
    volume: float = 100.0,
    lot_size: int = 50,
    expiry_rank: int = 2,
) -> dict:
    """
    Minimal-but-sufficient row for:
      - BaseStrategy REQUIRED_MARKET_COLS
      - BullCallSpreadStrategy selection (spot_close, open_int, volume)
      - Engine compute_mtm_pnl_rupee (close, settle_pr, lot_size, expiry_dt window + expiry row)
    """
    d = pd.Timestamp(date).normalize()
    e = pd.Timestamp(expiry_dt).normalize()

    return {
        "date": d,
        "symbol": "NIFTY",
        "instrument": "OPTIDX",
        "expiry_dt": e,
        "strike_pr": float(strike),
        "option_typ": "CE",
        "close": float(close),
        "settle_pr": float(settle),
        "lot_size": int(lot_size),
        "expiry_rank": int(expiry_rank),
        "is_trading_day": True,
        "is_opt_monthly_expiry": bool(is_monthly_expiry_day),
        # Strategy needs these:
        "spot_close": float(spot_close),
        "open_int": float(open_int),
        "volume": float(volume),
    }


def _make_strategy() -> BullCallSpreadStrategy:
    cfg = BullCallSpreadStrategyConfig(
        input_parquet_path="data>curated>derivatives_clean.parquet",
        symbol="NIFTY",
        require_expiry_rank_1=False,  # important for monthly chain behavior
        strike_interval=50,
        otm_points=200,
        max_abort_ratio=0.90,  # tests intentionally create small datasets
        option_instrument="OPTIDX",
    )
    return BullCallSpreadStrategy(cfg)


def test_build_trades_unique_leg_ids_and_signs():
    strat = _make_strategy()

    # Monthly expiry day exists at 2024-01-03; we enter 2024-01-02.
    market_df = pd.DataFrame(
        [
            _mk_row(
                date="2024-01-02",
                expiry_dt="2024-01-03",
                strike=20000,
                settle=100,
                close=100,
                spot_close=20010,
                is_monthly_expiry_day=False,
            ),
            _mk_row(
                date="2024-01-02",
                expiry_dt="2024-01-03",
                strike=20200,
                settle=40,
                close=40,
                spot_close=20010,
                is_monthly_expiry_day=False,
            ),
            # Mark monthly expiry date on any row (engine/strategy reads from market_df["date"])
            _mk_row(
                date="2024-01-03",
                expiry_dt="2024-01-03",
                strike=20000,
                settle=110,
                close=110,
                spot_close=20050,
                is_monthly_expiry_day=True,
            ),
        ]
    )

    entry_days = pd.DataFrame([{"symbol": "NIFTY", "entry_date": pd.Timestamp("2024-01-02")}])
    trades = strat.build_trades(market_df, entry_days)

    assert len(trades) == 2
    assert trades["trade_id"].nunique() == 2

    parent = "NIFTY_BCS_20240102"
    assert set(trades["trade_id"]) == {f"{parent}_LONG_CE", f"{parent}_SHORT_CE"}

    long_leg = trades.loc[trades["trade_id"] == f"{parent}_LONG_CE"].iloc[0]
    short_leg = trades.loc[trades["trade_id"] == f"{parent}_SHORT_CE"].iloc[0]

    assert int(long_leg["position_sign"]) == 1
    assert int(short_leg["position_sign"]) == -1
    assert int(long_leg["strike_pr"]) == 20000
    assert int(short_leg["strike_pr"]) == 20200
    assert pd.Timestamp(long_leg["expiry_dt"]).normalize() == pd.Timestamp("2024-01-03")
    assert pd.Timestamp(short_leg["expiry_dt"]).normalize() == pd.Timestamp("2024-01-03")


def test_liquidity_guard_aborts_trade_when_any_leg_illiquid():
    strat = _make_strategy()

    market_df = pd.DataFrame(
        [
            # Make OTM leg illiquid at entry (volume=0) -> should abort
            _mk_row(
                date="2024-01-02",
                expiry_dt="2024-01-03",
                strike=20000,
                settle=100,
                close=100,
                spot_close=20010,
                is_monthly_expiry_day=False,
                open_int=100,
                volume=100,
            ),
            _mk_row(
                date="2024-01-02",
                expiry_dt="2024-01-03",
                strike=20200,
                settle=40,
                close=40,
                spot_close=20010,
                is_monthly_expiry_day=False,
                open_int=100,
                volume=0,  # illiquid
            ),
            _mk_row(
                date="2024-01-03",
                expiry_dt="2024-01-03",
                strike=20000,
                settle=110,
                close=110,
                spot_close=20050,
                is_monthly_expiry_day=True,
            ),
        ]
    )

    entry_days = pd.DataFrame([{"symbol": "NIFTY", "entry_date": pd.Timestamp("2024-01-02")}])
    trades = strat.build_trades(market_df, entry_days)

    assert trades.empty


def test_mtm_and_parent_aggregation_is_sum_of_legs():
    strat = _make_strategy()

    market_df = pd.DataFrame(
        [
            # Entry date (2024-01-02), expiry date is 2024-01-03
            _mk_row(
                date="2024-01-02",
                expiry_dt="2024-01-03",
                strike=20000,
                settle=100,
                close=100,
                spot_close=20010,
                is_monthly_expiry_day=False,
            ),
            _mk_row(
                date="2024-01-02",
                expiry_dt="2024-01-03",
                strike=20200,
                settle=40,
                close=40,
                spot_close=20010,
                is_monthly_expiry_day=False,
            ),
            # Expiry day rows for both strikes (engine requires last date == expiry_dt per trade)
            _mk_row(
                date="2024-01-03",
                expiry_dt="2024-01-03",
                strike=20000,
                settle=110,
                close=110,
                spot_close=20050,
                is_monthly_expiry_day=True,
            ),
            _mk_row(
                date="2024-01-03",
                expiry_dt="2024-01-03",
                strike=20200,
                settle=45,
                close=45,
                spot_close=20050,
                is_monthly_expiry_day=True,
            ),
        ]
    )

    entry_days = pd.DataFrame([{"symbol": "NIFTY", "entry_date": pd.Timestamp("2024-01-02")}])
    trades = strat.build_trades(market_df, entry_days)
    assert len(trades) == 2

    legs_mtm = strat.compute_mtm_pnl_rupee(market_df, trades)

    # Expected on expiry date:
    # long: +1*(110-100)*50 = +500
    # short: -1*(45-40)*50 = -250
    # spread = +250
    out = strat._aggregate_legs_to_parent(legs_mtm, trades_df=trades)

    parent_id = "NIFTY_BCS_20240102"

    # entry date pnl is deterministic; engine defines it as entry_pnl using settle-entry_price -> 0
    r_entry = out.loc[(out["trade_id"] == parent_id) & (out["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert float(r_entry["strategy_pnl_rupee"]) == 0.0

    r_exp = out.loc[(out["trade_id"] == parent_id) & (out["date"] == pd.Timestamp("2024-01-03"))].iloc[0]
    assert float(r_exp["strategy_pnl_rupee"]) == 250.0
    assert float(r_exp["cum_pnl_rupee"]) == 250.0
    assert int(r_exp["atm_strike"]) == 20000
    assert int(r_exp["otm_strike"]) == 20200


if __name__ == "__main__":
    # Allows running directly:
    #   python tests/test_bull_call_spread.py
    raise SystemExit(pytest.main([__file__]))
