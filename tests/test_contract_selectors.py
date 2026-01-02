import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.contract_selectors import (
    apply_liquidity_filters,
    apply_strike_band,
    get_chain,
    select_atm_strike,
    select_otm_strike_above,
    select_otm_strike_below,
)


def _mk_row(
    date,
    symbol,
    instrument,
    expiry_dt,
    strike_pr,
    option_typ,
    contracts,
    open_int,
    settle_pr,
):
    return {
        "date": pd.Timestamp(date),
        "symbol": symbol,
        "instrument": instrument,
        "expiry_dt": pd.Timestamp(expiry_dt),
        "strike_pr": float(strike_pr),
        "option_typ": option_typ,
        "contracts": float(contracts),
        "open_int": float(open_int),
        "settle_pr": float(settle_pr),
    }


def test_get_chain_returns_expected_slice():
    expiry = pd.Timestamp("2025-01-30")
    entry = pd.Timestamp("2025-01-03")

    rows = [
        _mk_row(entry, "NIFTY", "OPTIDX", expiry, 100, "CE", 10, 100, 1.0),
        _mk_row(entry, "NIFTY", "OPTIDX", expiry, 100, "PE", 10, 100, 1.0),
        _mk_row(entry, "NIFTY", "FUTIDX", expiry, 0, "XX", 10, 100, 1.0),  # wrong instrument
        _mk_row(entry, "BANKNIFTY", "OPTIDX", expiry, 100, "CE", 10, 100, 1.0),  # wrong symbol
        _mk_row("2025-01-02", "NIFTY", "OPTIDX", expiry, 100, "CE", 10, 100, 1.0),  # wrong date
        _mk_row(entry, "NIFTY", "OPTIDX", "2025-02-06", 100, "CE", 10, 100, 1.0),  # wrong expiry
    ]
    market_df = pd.DataFrame(rows)

    chain = get_chain(market_df, "NIFTY", expiry, entry)

    assert len(chain) == 2
    assert set(chain["option_typ"].tolist()) == {"CE", "PE"}
    assert set(chain["instrument"].tolist()) == {"OPTIDX"}
    assert set(chain["symbol"].tolist()) == {"NIFTY"}
    assert set(chain["expiry_dt"].tolist()) == {expiry}
    assert set(chain["date"].tolist()) == {entry}


def test_apply_strike_band_contains_atm_and_neighbors():
    expiry = pd.Timestamp("2025-01-30")
    entry = pd.Timestamp("2025-01-03")

    strikes = [100, 110, 120, 130, 140]
    rows = []
    for k in strikes:
        rows.append(_mk_row(entry, "NIFTY", "OPTIDX", expiry, k, "CE", 10, 100, 1.0))
        rows.append(_mk_row(entry, "NIFTY", "OPTIDX", expiry, k, "PE", 10, 100, 1.0))
    chain_df = pd.DataFrame(rows)

    # spot close near 123 -> ATM should be 120 (tie-free here)
    band_df = apply_strike_band(chain_df, spot_close=123.0, strike_band_n=1)

    kept = sorted(band_df["strike_pr"].unique().tolist())
    assert kept == [110.0, 120.0, 130.0]
    assert len(band_df) == 6
    assert set(band_df["option_typ"].unique().tolist()) == {"CE", "PE"}


def test_liquidity_filters_off_no_change():
    df = pd.DataFrame(
        [
            _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 100, "CE", 1, 1, 1.0),
            _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 100, "PE", 2, 2, 2.0),
        ]
    )
    out = apply_liquidity_filters(df, "OFF", min_contracts=999, min_open_int=999, liquidity_percentile=99)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))


def test_liquidity_filters_absolute():
    df = pd.DataFrame(
        [
            _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 100, "CE", 10, 100, 1.0),  # keep
            _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 100, "PE", 0, 100, 1.0),   # drop (contracts)
            _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 110, "CE", 10, 0, 1.0),    # drop (open_int)
            _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 110, "PE", 10, 100, 0.0),  # drop (settle_pr)
        ]
    )
    out = apply_liquidity_filters(df, "ABSOLUTE", min_contracts=5, min_open_int=50, liquidity_percentile=50)

    assert len(out) == 1
    assert out.iloc[0]["strike_pr"] == 100.0
    assert out.iloc[0]["option_typ"] == "CE"


def test_liquidity_filters_percentile():
    # contracts: [10,20,30,40,50,60] -> p50 (linear) = 35
    # open_int:   [1, 2, 3, 4, 5, 6] -> p50 (linear) = 3.5
    rows = []
    base = [
        (100, "CE", 10, 1, 1.0),
        (100, "PE", 20, 2, 1.0),
        (110, "CE", 30, 3, 1.0),
        (110, "PE", 40, 4, 1.0),
        (120, "CE", 50, 5, 1.0),
        (120, "PE", 60, 6, 1.0),
    ]
    for strike, typ, c, oi, sp in base:
        rows.append(_mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", strike, typ, c, oi, sp))
    df = pd.DataFrame(rows)

    out = apply_liquidity_filters(df, "PERCENTILE", min_contracts=0, min_open_int=0, liquidity_percentile=50)

    # Expect keep rows with contracts >= 35 AND open_int >= 3.5
    # That means the last 3 rows only: (110, PE), (120, CE), (120, PE)
    kept_pairs = {(r["strike_pr"], r["option_typ"]) for _, r in out.iterrows()}
    assert kept_pairs == {(110.0, "PE"), (120.0, "CE"), (120.0, "PE")}


def test_select_atm_strike_fallback_steps():
    # spot=112 -> closest strike=110 but invalid (missing PE)
    # next=120 invalid (missing CE)
    # next=100 valid (has both)
    rows = [
        _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 110, "CE", 10, 10, 1.0),
        _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 120, "PE", 10, 10, 1.0),
        _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 100, "CE", 10, 10, 1.0),
        _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 100, "PE", 10, 10, 1.0),
    ]
    df = pd.DataFrame(rows)

    assert select_atm_strike(df, spot_close=112.0, max_atm_search_steps=0) is None
    assert select_atm_strike(df, spot_close=112.0, max_atm_search_steps=2) == 100.0


def test_select_otm_strike_fallback():
    rows = [
        _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 100, "CE", 10, 10, 1.0),
        _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 110, "CE", 10, 10, 1.0),
        _mk_row("2025-01-03", "NIFTY", "OPTIDX", "2025-01-30", 130, "CE", 10, 10, 1.0),
    ]
    df = pd.DataFrame(rows)

    # above: target missing -> nearest higher available is 130
    assert select_otm_strike_above(df, atm=110.0, points=10.0) == 130.0

    # below: target 90 missing -> nearest lower available is 100
    assert select_otm_strike_below(df, atm=110.0, points=20.0) == 100.0

    # none above case
    assert select_otm_strike_above(df, atm=130.0, points=10.0) is None
