import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.bear_put_spread import BearPutSpreadStrategy
import src.strategies.bear_put_spread as bps


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
        # required-but-nullable selection param not used here
        "otm_distance_points": None,
    }


def test_bear_put_spread_emits_correct_put_legs(monkeypatch):
    cycles_df = pd.DataFrame(
        [
            {"symbol": "NIFTY", "tenor": "WEEKLY", "expiry_dt": "2025-01-09", "entry_date": "2025-01-03", "exit_date": "2025-01-09"},
        ]
    )
    monkeypatch.setattr(bps.es, "build_expiry_cycles", lambda market_df, symbol, tenor: cycles_df)

    chain_df = pd.DataFrame(
        [
            {"strike_pr": 20000.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 11.0},
            {"strike_pr": 19800.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 6.0},
            # CE/PE at ATM exists in real filtered chain; select_atm_strike relies on chain post-filter.
            {"strike_pr": 20000.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 10.0},
        ]
    )

    monkeypatch.setattr(bps.cs, "get_chain", lambda market_df, symbol, expiry_dt, entry_date: chain_df)
    monkeypatch.setattr(bps.cs, "infer_strike_interval", lambda chain_df: 50)
    monkeypatch.setattr(bps.cs, "apply_strike_band", lambda chain_df, spot_close, strike_band_n: chain_df)
    monkeypatch.setattr(
        bps.cs,
        "apply_liquidity_filters",
        lambda band_df, liquidity_mode, min_contracts, min_open_int, liquidity_percentile: band_df,
    )
    monkeypatch.setattr(bps.cs, "select_atm_strike", lambda filt_df, spot_close, max_atm_search_steps: 20000.0)

    trades_df = BearPutSpreadStrategy().build_trades(market_df=pd.DataFrame(), cfg=_base_cfg(200.0))
    assert len(trades_df) == 2
    assert (trades_df["option_typ"] == "PE").all()

    atm = trades_df.loc[trades_df["strike_pr"] == 20000.0].iloc[0]
    otm = trades_df.loc[trades_df["strike_pr"] == 19800.0].iloc[0]
    assert int(atm["side"]) == +1
    assert int(otm["side"]) == -1
    assert float(atm["width_points"]) == 200.0
    assert float(otm["width_points"]) == 200.0


def test_bear_put_spread_otm_fallback_below(monkeypatch):
    cycles_df = pd.DataFrame(
        [
            {"symbol": "NIFTY", "tenor": "WEEKLY", "expiry_dt": "2025-01-09", "entry_date": "2025-01-03", "exit_date": "2025-01-09"},
        ]
    )
    monkeypatch.setattr(bps.es, "build_expiry_cycles", lambda market_df, symbol, tenor: cycles_df)

    # ATM=20000, preferred OTM=19800 missing.
    # Below-ATM PE strikes available: 19850 and 19750 (tie distance 50), tie -> larger => 19850
    chain_df = pd.DataFrame(
        [
            {"strike_pr": 20000.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 11.0},
            {"strike_pr": 20000.0, "option_typ": "CE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 10.0},
            {"strike_pr": 19850.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 7.0},
            {"strike_pr": 19750.0, "option_typ": "PE", "spot_close": 20010.0, "contracts": 100.0, "open_int": 1000.0, "settle_pr": 6.5},
        ]
    )
    monkeypatch.setattr(bps.cs, "get_chain", lambda market_df, symbol, expiry_dt, entry_date: chain_df)
    monkeypatch.setattr(bps.cs, "infer_strike_interval", lambda chain_df: 50)
    monkeypatch.setattr(bps.cs, "apply_strike_band", lambda chain_df, spot_close, strike_band_n: chain_df)
    monkeypatch.setattr(
        bps.cs,
        "apply_liquidity_filters",
        lambda band_df, liquidity_mode, min_contracts, min_open_int, liquidity_percentile: band_df,
    )
    monkeypatch.setattr(bps.cs, "select_atm_strike", lambda filt_df, spot_close, max_atm_search_steps: 20000.0)

    trades_df = BearPutSpreadStrategy().build_trades(market_df=pd.DataFrame(), cfg=_base_cfg(200.0))
    assert len(trades_df) == 2
    short_strike = float(trades_df.loc[trades_df["side"] == -1, "strike_pr"].iloc[0])
    assert short_strike == 19850.0
