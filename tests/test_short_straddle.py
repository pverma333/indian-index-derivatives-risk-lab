import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.short_straddle import ShortStraddleStrategy
import src.strategies.short_straddle as ss


def _base_cfg() -> dict:
    return {
        "symbol": "NIFTY",
        "tenor": "WEEKLY",
        "qty_lots": 1,
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
        # required-but-nullable selection params
        "width_points": None,
        "otm_distance_points": None,
    }


def test_short_straddle_emits_two_legs_per_trade(monkeypatch):
    cycles_df = pd.DataFrame(
        [
            {"symbol": "NIFTY", "tenor": "WEEKLY", "expiry_dt": "2025-01-09", "entry_date": "2025-01-03", "exit_date": "2025-01-09"},
            {"symbol": "NIFTY", "tenor": "WEEKLY", "expiry_dt": "2025-01-16", "entry_date": "2025-01-10", "exit_date": "2025-01-16"},
        ]
    )
    monkeypatch.setattr(ss.es, "build_expiry_cycles", lambda market_df, symbol, tenor: cycles_df)

    chain_df = pd.DataFrame(
        [
            {"strike_pr": 20000.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 10.0},
            {"strike_pr": 20000.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 12.0},
        ]
    )
    monkeypatch.setattr(ss.cs, "get_chain", lambda market_df, symbol, expiry_dt, entry_date: chain_df)
    monkeypatch.setattr(ss.cs, "infer_strike_interval", lambda chain_df: 50)
    monkeypatch.setattr(ss.cs, "apply_strike_band", lambda chain_df, spot_close, strike_band_n: chain_df)
    monkeypatch.setattr(ss.cs, "apply_liquidity_filters", lambda band_df, liquidity_mode, min_contracts, min_open_int, liquidity_percentile: band_df)
    monkeypatch.setattr(ss.cs, "select_atm_strike", lambda filt_df, spot_close, max_atm_search_steps: 20000.0)

    strategy = ShortStraddleStrategy()
    trades_df = strategy.build_trades(market_df=pd.DataFrame(), cfg=_base_cfg())

    # 2 cycles => 2 trades => 4 legs
    assert len(trades_df) == 4
    assert set(trades_df["option_typ"].unique()) == {"CE", "PE"}
    assert (trades_df["side"] == -1).all()
    assert (trades_df.groupby("trade_id").size() == 2).all()


def test_short_straddle_leg_ids_unique(monkeypatch):
    cycles_df = pd.DataFrame(
        [
            {"symbol": "NIFTY", "tenor": "WEEKLY", "expiry_dt": "2025-01-09", "entry_date": "2025-01-03", "exit_date": "2025-01-09"},
            {"symbol": "NIFTY", "tenor": "WEEKLY", "expiry_dt": "2025-01-16", "entry_date": "2025-01-10", "exit_date": "2025-01-16"},
        ]
    )
    monkeypatch.setattr(ss.es, "build_expiry_cycles", lambda market_df, symbol, tenor: cycles_df)

    chain_df = pd.DataFrame(
        [
            {"strike_pr": 20000.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 10.0},
            {"strike_pr": 20000.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 12.0},
        ]
    )
    monkeypatch.setattr(ss.cs, "get_chain", lambda market_df, symbol, expiry_dt, entry_date: chain_df)
    monkeypatch.setattr(ss.cs, "infer_strike_interval", lambda chain_df: 50)
    monkeypatch.setattr(ss.cs, "apply_strike_band", lambda chain_df, spot_close, strike_band_n: chain_df)
    monkeypatch.setattr(ss.cs, "apply_liquidity_filters", lambda band_df, liquidity_mode, min_contracts, min_open_int, liquidity_percentile: band_df)
    monkeypatch.setattr(ss.cs, "select_atm_strike", lambda filt_df, spot_close, max_atm_search_steps: 20000.0)

    strategy = ShortStraddleStrategy()
    trades_df = strategy.build_trades(market_df=pd.DataFrame(), cfg=_base_cfg())

    assert trades_df["leg_id"].is_unique


def test_short_straddle_skips_when_no_chain(monkeypatch):
    cycles_df = pd.DataFrame(
        [{"symbol": "NIFTY", "tenor": "WEEKLY", "expiry_dt": "2025-01-09", "entry_date": "2025-01-03", "exit_date": "2025-01-09"}]
    )
    monkeypatch.setattr(ss.es, "build_expiry_cycles", lambda market_df, symbol, tenor: cycles_df)

    monkeypatch.setattr(ss.cs, "get_chain", lambda market_df, symbol, expiry_dt, entry_date: pd.DataFrame())

    strategy = ShortStraddleStrategy()
    trades_df = strategy.build_trades(market_df=pd.DataFrame(), cfg=_base_cfg())

    assert trades_df.empty
