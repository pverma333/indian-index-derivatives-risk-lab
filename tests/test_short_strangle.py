import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.short_strangle import ShortStrangleStrategy
import src.strategies.short_strangle as ss


def _base_cfg(otm_distance_points: float = 300.0) -> dict:
    return {
        "symbol": "NIFTY",
        "tenor": "WEEKLY",
        "qty_lots": 1,
        "otm_distance_points": otm_distance_points,
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
        # required-but-nullable selection param not used here
        "width_points": None,
    }


def test_short_strangle_emits_two_short_legs(monkeypatch):
    cycles_df = pd.DataFrame(
        [
            {"symbol": "NIFTY", "tenor": "WEEKLY", "expiry_dt": "2025-01-09", "entry_date": "2025-01-03", "exit_date": "2025-01-09"},
        ]
    )
    monkeypatch.setattr(ss.es, "build_expiry_cycles", lambda market_df, symbol, tenor: cycles_df)

    # ATM=20000; exact targets exist at +/-300
    chain_df = pd.DataFrame(
        [
            {"strike_pr": 20300.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 4.0},
            {"strike_pr": 19700.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 5.0},
            {"strike_pr": 20000.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 10.0},
            {"strike_pr": 20000.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 11.0},
        ]
    )
    monkeypatch.setattr(ss.cs, "get_chain", lambda market_df, symbol, expiry_dt, entry_date: chain_df)
    monkeypatch.setattr(ss.cs, "infer_strike_interval", lambda chain_df: 50)
    monkeypatch.setattr(ss.cs, "apply_strike_band", lambda chain_df, spot_close, strike_band_n: chain_df)
    monkeypatch.setattr(
        ss.cs,
        "apply_liquidity_filters",
        lambda band_df, liquidity_mode, min_contracts, min_open_int, liquidity_percentile: band_df,
    )
    monkeypatch.setattr(ss.cs, "select_atm_strike", lambda filt_df, spot_close, max_atm_search_steps: 20000.0)

    trades_df = ShortStrangleStrategy().build_trades(market_df=pd.DataFrame(), cfg=_base_cfg(300.0))
    assert len(trades_df) == 2
    assert set(trades_df["option_typ"].tolist()) == {"CE", "PE"}
    assert set(trades_df["side"].tolist()) == {-1}
    assert float(trades_df["otm_distance_points"].iloc[0]) == 300.0


def test_short_strangle_otm_fallback_both_sides(monkeypatch):
    cycles_df = pd.DataFrame(
        [
            {"symbol": "NIFTY", "tenor": "WEEKLY", "expiry_dt": "2025-01-09", "entry_date": "2025-01-03", "exit_date": "2025-01-09"},
        ]
    )
    monkeypatch.setattr(ss.es, "build_expiry_cycles", lambda market_df, symbol, tenor: cycles_df)

    # ATM=20000; preferred CE=20325 missing, preferred PE=19675 missing
    # Available strikes: CE above: 20300 and 20350 (tie -> smaller => 20300)
    # PE below: 19700 and 19650 (tie -> larger => 19700)
    chain_df = pd.DataFrame(
        [
            {"strike_pr": 20000.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 10.0},
            {"strike_pr": 20000.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 11.0},
            {"strike_pr": 20300.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 4.0},
            {"strike_pr": 20350.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 3.5},
            {"strike_pr": 19700.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 5.0},
            {"strike_pr": 19650.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 4.8},
        ]
    )
    monkeypatch.setattr(ss.cs, "get_chain", lambda market_df, symbol, expiry_dt, entry_date: chain_df)
    monkeypatch.setattr(ss.cs, "infer_strike_interval", lambda chain_df: 50)
    monkeypatch.setattr(ss.cs, "apply_strike_band", lambda chain_df, spot_close, strike_band_n: chain_df)
    monkeypatch.setattr(
        ss.cs,
        "apply_liquidity_filters",
        lambda band_df, liquidity_mode, min_contracts, min_open_int, liquidity_percentile: band_df,
    )
    monkeypatch.setattr(ss.cs, "select_atm_strike", lambda filt_df, spot_close, max_atm_search_steps: 20000.0)

    trades_df = ShortStrangleStrategy().build_trades(market_df=pd.DataFrame(), cfg=_base_cfg(325.0))
    assert len(trades_df) == 2

    ce_strike = float(trades_df.loc[trades_df["option_typ"] == "CE", "strike_pr"].iloc[0])
    pe_strike = float(trades_df.loc[trades_df["option_typ"] == "PE", "strike_pr"].iloc[0])

    assert ce_strike == 20300.0
    assert pe_strike == 19700.0
    assert (trades_df["side"] == -1).all()
