import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.trade_schema import TradeSchemaError, validate_trades_df


def _base_valid_trades_df() -> pd.DataFrame:
    # Two legs, unique leg_id, all required fields present.
    return pd.DataFrame(
        [
            {
                "strategy_name": "short_straddle",
                "tenor": "WEEKLY",
                "trade_id": "T1",
                "leg_id": "L1",
                "symbol": "NIFTY",
                "instrument": "OPTIDX",
                "expiry_dt": pd.Timestamp("2025-01-30"),
                "strike_pr": 22000.0,
                "option_typ": "CE",
                "side": -1,
                "qty_lots": 1,
                "entry_date": pd.Timestamp("2025-01-24"),
                "exit_date": pd.Timestamp("2025-01-30"),
                "strike_band_n": 10,
                "width_points": None,
                "otm_distance_points": None,
                "max_atm_search_steps": 3,
                "liquidity_mode": "OFF",
                "min_contracts": 1,
                "min_open_int": 1,
                "liquidity_percentile": 50,
                "exit_rule": "EXPIRY",
                "exit_k_days": None,
                "fees_bps": 0.0,
                "fixed_fee_per_lot": 0.0,
            },
            {
                "strategy_name": "short_straddle",
                "tenor": "WEEKLY",
                "trade_id": "T1",
                "leg_id": "L2",
                "symbol": "NIFTY",
                "instrument": "OPTIDX",
                "expiry_dt": pd.Timestamp("2025-01-30"),
                "strike_pr": 22000.0,
                "option_typ": "PE",
                "side": -1,
                "qty_lots": 1,
                "entry_date": pd.Timestamp("2025-01-24"),
                "exit_date": pd.Timestamp("2025-01-30"),
                "strike_band_n": 10,
                "width_points": None,
                "otm_distance_points": None,
                "max_atm_search_steps": 3,
                "liquidity_mode": "OFF",
                "min_contracts": 1,
                "min_open_int": 1,
                "liquidity_percentile": 50,
                "exit_rule": "EXPIRY",
                "exit_k_days": None,
                "fees_bps": 0.0,
                "fixed_fee_per_lot": 0.0,
            },
        ]
    )


def test_validate_trades_df_missing_columns_raises():
    df = _base_valid_trades_df().drop(columns=["strategy_name", "qty_lots"])
    with pytest.raises(TradeSchemaError) as e:
        validate_trades_df(df)
    err = e.value
    assert "missing required columns" in str(err).lower()
    assert "strategy_name" in err.missing_columns
    assert "qty_lots" in err.missing_columns


def test_validate_trades_df_duplicate_leg_id_raises():
    df = _base_valid_trades_df()
    df.loc[1, "leg_id"] = "L1"  # duplicate
    with pytest.raises(TradeSchemaError) as e:
        validate_trades_df(df)
    err = e.value
    assert "duplicate leg_id" in str(err).lower()
    assert err.missing_columns == []
    assert not err.violations_keys.empty
    assert set(err.violations_keys["leg_id"].unique()) == {"L1"}


def test_validate_trades_df_invalid_side_raises():
    df = _base_valid_trades_df()
    df.loc[0, "side"] = 0
    with pytest.raises(TradeSchemaError) as e:
        validate_trades_df(df)
    err = e.value
    assert any("INVALID_SIDE" in s for s in err.rule_errors)
    assert "L1" in set(err.violations_keys["leg_id"])


def test_validate_trades_df_bad_tenor_raises():
    df = _base_valid_trades_df()
    df.loc[0, "tenor"] = "DAILY"
    with pytest.raises(TradeSchemaError) as e:
        validate_trades_df(df)
    err = e.value
    assert any("INVALID_TENOR" in s for s in err.rule_errors)
    assert "L1" in set(err.violations_keys["leg_id"])
