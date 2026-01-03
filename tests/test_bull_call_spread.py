import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.bull_call_spread import BullCallSpreadStrategy
import src.strategies.bull_call_spread as bcs


def _base_cfg(width_points: float = 200.0) -> dict:
    return {
        "symbol": "NIFTY",
        "tenor": "WEEKLY",
        "qty_lots": 1,
        "width_points": width_points,
        "strike_band_n": 10,
        "max_atm_search_steps": 3,
        "liquidity_mode": "OFF",
        "min_contracts": 1,
        "min_open_int": 1,
        "liquidity_percentile": 50,
        "exit_rule": "EXPIRY",
        "exit_k_days": None,
        "fees_bps": 0.0,
        "fixed_fee_per_lot": 0.0,
        "otm_distance_points": None,
    }


def test_bull_call_spread_emits_correct_legs_and_sides(monkeypatch):
    cycles_df = pd.DataFrame(
        [
            {"symbol": "NIFTY", "tenor": "WEEKLY", "expiry_dt": "2025-01-09", "entry_date": "2025-01-03", "exit_date": "2025-01-09"},
        ]
    )
    monkeypatch.setattr(bcs.es, "build_expiry_cycles", lambda market_df, symbol, tenor: cycles_df)

    chain_df = pd.DataFrame(
        [
            {"strike_pr": 20000.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 10.0},
            {"strike_pr": 20200.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 5.0},
        ]
    )
    monkeypatch.setattr(bcs.cs, "get_chain", lambda market_df, symbol, expiry_dt, entry_date: chain_df)
    monkeypatch.setattr(bcs.cs, "infer_strike_interval", lambda chain_df: 50)
    monkeypatch.setattr(bcs.cs, "apply_strike_band", lambda chain_df, spot_close, strike_band_n: chain_df)
    monkeypatch.setattr(
        bcs.cs,
        "apply_liquidity_filters",
        lambda band_df, liquidity_mode, min_contracts, min_open_int, liquidity_percentile: band_df,
    )
    monkeypatch.setattr(bcs.cs, "select_atm_strike", lambda filt_df, spot_close, max_atm_search_steps: 20000.0)

    strategy = BullCallSpreadStrategy()
    trades_df = strategy.build_trades(market_df=pd.DataFrame(), cfg=_base_cfg(width_points=200.0))

    assert len(trades_df) == 2
    assert (trades_df["option_typ"] == "CE").all()

    # Long ATM, short OTM
    atm_row = trades_df.loc[trades_df["strike_pr"] == 20000.0].iloc[0]
    otm_row = trades_df.loc[trades_df["strike_pr"] == 20200.0].iloc[0]

    assert int(atm_row["side"]) == +1
    assert int(otm_row["side"]) == -1

    assert float(atm_row["width_points"]) == 200.0
    assert float(otm_row["width_points"]) == 200.0
    assert int(atm_row["strike_interval_used"]) == 50
    assert int(otm_row["strike_interval_used"]) == 50


def test_bull_call_spread_otm_fallback_above(monkeypatch):
    cycles_df = pd.DataFrame(
        [
            {"symbol": "NIFTY", "tenor": "WEEKLY", "expiry_dt": "2025-01-09", "entry_date": "2025-01-03", "exit_date": "2025-01-09"},
        ]
    )
    monkeypatch.setattr(bcs.es, "build_expiry_cycles", lambda market_df, symbol, tenor: cycles_df)

    # ATM=20000, preferred=20200 (width=200)
    # Exact 20200 is missing; strikes above ATM: 20150 and 20250
    # Our fallback chooses nearest to preferred among strikes > ATM, tie -> smaller.
    chain_df = pd.DataFrame(
        [
            {"strike_pr": 20000.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 10.0},
            {"strike_pr": 20150.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 7.0},
            {"strike_pr": 20250.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 4.0},
        ]
    )

    monkeypatch.setattr(bcs.cs, "get_chain", lambda market_df, symbol, expiry_dt, entry_date: chain_df)
    monkeypatch.setattr(bcs.cs, "infer_strike_interval", lambda chain_df: 50)
    monkeypatch.setattr(bcs.cs, "apply_strike_band", lambda chain_df, spot_close, strike_band_n: chain_df)
    monkeypatch.setattr(
        bcs.cs,
        "apply_liquidity_filters",
        lambda band_df, liquidity_mode, min_contracts, min_open_int, liquidity_percentile: band_df,
    )
    monkeypatch.setattr(bcs.cs, "select_atm_strike", lambda filt_df, spot_close, max_atm_search_steps: 20000.0)

    strategy = BullCallSpreadStrategy()
    trades_df = strategy.build_trades(market_df=pd.DataFrame(), cfg=_base_cfg(width_points=200.0))

    assert len(trades_df) == 2
    assert (trades_df["side"].isin([+1, -1])).all()

    # Preferred is 20200 missing; nearest above ATM to 20200 is 20150 (distance 50) and 20250 (distance 50), tie -> 20150
    assert (trades_df["strike_pr"] == 20150.0).any()
    assert (trades_df["strike_pr"] == 20000.0).any()
