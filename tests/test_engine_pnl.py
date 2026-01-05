from pathlib import Path
import sys

import pandas as pd
import pytest

# Import reliability (repo-root)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.engine.engine_pnl import compute_legs_pnl


def _mk_market_opt():
    # 2 trading dates in-market; exit_date intentionally beyond to test ASOF OPEN
    return pd.DataFrame(
        [
            {
                "date": "2025-01-02",
                "symbol": "NIFTY",
                "instrument": "OPTIDX",
                "expiry_dt": "2025-01-30",
                "strike_pr": 20000.0,
                "option_typ": "CE",
                "settle_used": 100.0,
                "lot_size": 50,
                "index_open_price": 20010.0,
                "rate_182d": 0.07,
                "chg_in_oi": 1.0,
            },
            {
                "date": "2025-01-03",
                "symbol": "NIFTY",
                "instrument": "OPTIDX",
                "expiry_dt": "2025-01-30",
                "strike_pr": 20000.0,
                "option_typ": "CE",
                "settle_used": 120.0,
                "lot_size": 50,
                "index_open_price": 20020.0,
                "rate_182d": 0.07,
                "chg_in_oi": 2.0,
            },
        ]
    )


def _mk_trades_opt():
    return pd.DataFrame(
        [
            {
                "strategy_name": "short_straddle",
                "tenor": "WEEKLY",
                "trade_id": "T1",
                "leg_id": "L1",
                "symbol": "NIFTY",
                "instrument": "OPTIDX",
                "expiry_dt": "2025-01-30",
                "strike_pr": 20000.0,
                "option_typ": "CE",
                "entry_date": "2025-01-02",
                "exit_date": "2025-02-10",  # beyond market coverage
                "side": 1,
                "qty_lots": 1,
                "entry_price": 90.0,
            }
        ]
    )


def test_engine_day0_settle_prev_used_equals_entry_price():
    mkt = _mk_market_opt()
    trd = _mk_trades_opt()
    legs, skips = compute_legs_pnl(mkt, trd, coverage_mode="ASOF")

    day0 = legs.loc[legs["date"] == pd.Timestamp("2025-01-02")]
    assert len(day0) == 1
    assert float(day0["settle_prev_used"].iloc[0]) == pytest.approx(90.0)


def test_engine_mtm_day0_anchor_no_nan():
    mkt = _mk_market_opt()
    trd = _mk_trades_opt()
    legs, _ = compute_legs_pnl(mkt, trd, coverage_mode="ASOF")

    # day0 mtm = units * (settle_used - entry_price)
    day0 = legs.loc[legs["date"] == pd.Timestamp("2025-01-02")].iloc[0]
    units0 = float(day0["units"])
    assert units0 == pytest.approx(1 * 1 * 50)

    expected = units0 * (100.0 - 90.0)
    assert float(day0["mtm_pnl"]) == pytest.approx(expected)
    assert legs["settle_prev_used"].isna().sum() == 0


def test_engine_gap_proxy_opt_intrinsic_open():
    mkt = _mk_market_opt()
    trd = _mk_trades_opt()
    legs, _ = compute_legs_pnl(mkt, trd, coverage_mode="ASOF")

    day0 = legs.loc[legs["date"] == pd.Timestamp("2025-01-02")].iloc[0]
    # CE intrinsic at open = max(index_open - strike, 0) = 10
    intr_open = 10.0
    expected = float(day0["units"]) * (intr_open - 90.0)
    assert day0["gap_method"] == "INTRINSIC_OPEN_PROXY"
    assert float(day0["gap_risk_pnl_proxy"]) == pytest.approx(expected)


def test_engine_asof_marks_open_when_end_date_used_lt_exit_date():
    mkt = _mk_market_opt()
    trd = _mk_trades_opt()
    legs, skips = compute_legs_pnl(mkt, trd, coverage_mode="ASOF")

    assert skips.empty
    assert (legs["status"] == "OPEN").all()
    assert legs["end_date_used"].max() == pd.Timestamp("2025-01-03")
    assert legs["date"].max() == pd.Timestamp("2025-01-03")


def test_engine_asof_does_not_create_missing_rows_beyond_asof():
    # Here, symbol has only two dates; exit_date is far.
    # Expected dates should be capped to [entry_date, end_date_used] and thus should not skip.
    mkt = _mk_market_opt()
    trd = _mk_trades_opt()
    legs, skips = compute_legs_pnl(mkt, trd, coverage_mode="ASOF")
    assert skips.empty
    assert len(legs) == 2


def test_engine_strict_skips_when_market_max_date_lt_exit_date():
    mkt = _mk_market_opt()
    trd = _mk_trades_opt()
    legs, skips = compute_legs_pnl(mkt, trd, coverage_mode="STRICT")

    assert legs.empty
    assert len(skips) == 1
    assert skips["reason"].iloc[0] == "MARKET_WINDOW_END_BEFORE_EXIT_STRICT"


def test_engine_skips_df_contains_required_fields_and_reason():
    # Force empty slice by changing strike so contract slice empties
    mkt = _mk_market_opt()
    trd = _mk_trades_opt()
    trd.loc[0, "strike_pr"] = 99999.0

    legs, skips = compute_legs_pnl(mkt, trd, coverage_mode="ASOF")
    assert legs.empty
    assert len(skips) == 1
    assert skips["reason"].iloc[0] == "EMPTY_LEG_SLICE"

    required = {
        "strategy_name", "tenor", "trade_id", "leg_id",
        "symbol", "instrument", "expiry_dt", "strike_pr", "option_typ",
        "entry_date", "exit_date",
        "market_max_date", "as_of_date_used", "end_date_used",
        "coverage_mode", "reason", "details",
    }
    assert required.issubset(set(skips.columns))


def test_engine_gap_proxy_fut_index_open():
    market = pd.DataFrame(
        [
            {
                "date": "2025-01-02",
                "symbol": "NIFTY",
                "instrument": "FUTIDX",
                "expiry_dt": "2025-01-30",
                "settle_used": 20100.0,
                "lot_size": 50,
                "index_open_price": 20050.0,
            },
            {
                "date": "2025-01-03",
                "symbol": "NIFTY",
                "instrument": "FUTIDX",
                "expiry_dt": "2025-01-30",
                "settle_used": 20200.0,
                "lot_size": 50,
                "index_open_price": 20100.0,
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "strategy_name": "fut_demo",
                "tenor": "WEEKLY",
                "trade_id": "T2",
                "leg_id": "L2",
                "symbol": "NIFTY",
                "instrument": "FUTIDX",
                "expiry_dt": "2025-01-30",
                "entry_date": "2025-01-02",
                "exit_date": "2025-01-03",
                "side": 1,
                "qty_lots": 1,
                "entry_price": 20000.0,
            }
        ]
    )

    legs, skips = compute_legs_pnl(market, trades, coverage_mode="ASOF")
    assert skips.empty
    day0 = legs.loc[legs["date"] == pd.Timestamp("2025-01-02")].iloc[0]
    # gap = units * (index_open - settle_prev) ; settle_prev day0 = entry_price
    expected = float(day0["units"]) * (20050.0 - 20000.0)
    assert day0["gap_method"] == "INDEX_OPEN_PROXY"
    assert float(day0["gap_risk_pnl_proxy"]) == pytest.approx(expected)
