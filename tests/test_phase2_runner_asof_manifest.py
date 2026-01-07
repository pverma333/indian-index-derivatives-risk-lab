from pathlib import Path
import sys

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.reporting.positions_summary import build_positions_df
from src.strategies.aggregations import aggregate_legs_to_trade_pnl, aggregate_trades_to_strategy_pnl
from src.run_phase2_backtest import (
    compute_as_of_date_used,
    compute_market_max_date_symbol,
    filter_market_for_run_window,
    run_phase2,
)


def _demo_market_df(symbol: str = "NIFTY") -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": [symbol] * len(dates),
            "instrument": ["FUTIDX"] * len(dates),
            "expiry_dt": [pd.Timestamp("2024-01-25")] * len(dates),
            "settle_pr": [100.0] * len(dates),
            "spot_close": [100.0] * len(dates),
            "index_open_price": [100.0] * len(dates),
            "lot_size": [50] * len(dates),
            "is_opt_weekly_expiry": [False] * len(dates),
            "is_opt_monthly_expiry": [False] * len(dates),
            "cal_days_to_expiry": [0] * len(dates),
            "is_trading_day": [True] * len(dates),
            "rate_182d": [0.0] * len(dates),
            "rate_364d": [0.0] * len(dates),
            "chg_in_oi": [0.0] * len(dates),
        }
    )
    return df


def test_runner_market_max_date_computed_before_date_filter():
    m = _demo_market_df("NIFTY")

    # User chooses an end date earlier than dataset max.
    # market_max_date must still be dataset max (symbol-filtered), not the user end date.
    market_max = compute_market_max_date_symbol(m, "NIFTY")
    assert market_max == pd.Timestamp("2024-01-10")

    user_end = pd.Timestamp("2024-01-07")
    asof = compute_as_of_date_used(user_end_date=user_end, market_max_date=market_max, as_of_override=None)
    assert asof == pd.Timestamp("2024-01-07")


def test_runner_asof_date_bounded_by_market_max_date():
    m = _demo_market_df("NIFTY")
    market_max = compute_market_max_date_symbol(m, "NIFTY")
    assert market_max == pd.Timestamp("2024-01-10")

    user_end = pd.Timestamp("2024-02-01")  # beyond data
    asof = compute_as_of_date_used(user_end_date=user_end, market_max_date=market_max, as_of_override=None)
    assert asof == pd.Timestamp("2024-01-10")

    override = pd.Timestamp("2024-02-10")
    asof2 = compute_as_of_date_used(user_end_date=user_end, market_max_date=market_max, as_of_override=override)
    assert asof2 == pd.Timestamp("2024-01-10")

    override2 = pd.Timestamp("2024-01-05")
    asof3 = compute_as_of_date_used(user_end_date=user_end, market_max_date=market_max, as_of_override=override2)
    assert asof3 == pd.Timestamp("2024-01-05")


def test_runner_passes_asof_to_engine_and_persists_skips_df(monkeypatch, tmp_path: Path):
    # We patch:
    # - read parquet/csv via providing a real file (csv)
    # - validate_market_df, compute_settle_used no-op
    # - strategy build_trades to return deterministic trades
    # - engine compute_legs_pnl capture as_of_date and return dummy artifacts
    market_df = _demo_market_df("NIFTY")
    in_path = tmp_path / "market.csv"
    market_df.to_csv(in_path, index=False)

    # Patch validation and marking
    import src.validation.market_df_validator as vmod
    import src.engine.settlement_marking as smod
    import src.engine.engine_pnl as emod
    import src.config.phase2_params as pmod

    monkeypatch.setattr(vmod, "validate_market_df", lambda df: None)
    monkeypatch.setattr(smod, "compute_settle_used", lambda df: df.assign(settle_used=df["settle_pr"], price_method="SETTLE_PR"))

    # Patch strategy lookup: force one strategy class with build_trades()
    class _DemoStrategy:
        def build_trades(self, market_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
            # One leg that exits after asof -> OPEN in ASOF mode
            return pd.DataFrame(
                [
                    {
                        "strategy_name": "demo",
                        "tenor": cfg["tenor"],
                        "trade_id": "T1",
                        "leg_id": "L1",
                        "symbol": cfg["symbol"],
                        "instrument": "FUTIDX",
                        "expiry_dt": pd.Timestamp("2024-01-25"),
                        "strike_pr": 0.0,
                        "option_typ": pd.NA,
                        "side": 1,
                        "qty_lots": 1,
                        "entry_date": pd.Timestamp("2024-01-03"),
                        "exit_date": pd.Timestamp("2024-01-20"),
                        "entry_price": 100.0,
                        "strike_band_n": 10,
                        "width_points": 200,
                        "otm_distance_points": 0,
                        "max_atm_search_steps": 3,
                        "liquidity_mode": "OFF",
                        "min_contracts": 1,
                        "min_open_int": 1,
                        "liquidity_percentile": 0,
                        "exit_rule": "K_DAYS_BEFORE_EXPIRY",
                        "exit_k_days": 0,
                        "fees_bps": 0.0,
                        "fixed_fee_per_lot": 0.0,
                    }
                ]
            )

    import src.run_phase2_backtest as runner_mod
    monkeypatch.setattr(runner_mod, "_find_strategy_class", lambda name: _DemoStrategy)

    # Patch phase2_params minimal behavior
    monkeypatch.setattr(pmod, "get_phase2_default_run_config", lambda: None, raising=False)
    monkeypatch.setattr(pmod, "validate_run_config", lambda cfg: None, raising=False)
    monkeypatch.setattr(pmod, "validate_strategy_params", lambda strategy_name, params, tenor: None, raising=False)
    monkeypatch.setattr(pmod, "resolve_effective_strategy_params", lambda strategy_name, tenor, user_overrides=None: {}, raising=False)

    captured = {"as_of_date": None, "coverage_mode": None}

    def _fake_compute_legs_pnl(market_df: pd.DataFrame, trades_df: pd.DataFrame, *, coverage_mode="ASOF", as_of_date=None):
        captured["as_of_date"] = as_of_date
        captured["coverage_mode"] = coverage_mode

        legs = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "strategy_name": "demo",
                    "tenor": "WEEKLY",
                    "trade_id": "T1",
                    "leg_id": "L1",
                    "symbol": "NIFTY",
                    "instrument": "FUTIDX",
                    "expiry_dt": pd.Timestamp("2024-01-25"),
                    "strike_pr": 0.0,
                    "option_typ": pd.NA,
                    "side": 1,
                    "qty_lots": 1,
                    "lot_size": 50,
                    "units": 50,
                    "entry_date": pd.Timestamp("2024-01-03"),
                    "exit_date": pd.Timestamp("2024-01-20"),
                    "entry_price": 100.0,
                    "settle_used": 101.0,
                    "settle_prev_used": 100.0,
                    "mtm_pnl": 50.0,
                    "gap_risk_pnl_proxy": 0.0,
                    "gap_method": "INDEX_OPEN_PROXY",
                    "price_method": "SETTLE_USED",
                    "rate_91d": pd.NA,
                    "rate_182d": 0.0,
                    "rate_364d": 0.0,
                    "vix_close": pd.NA,
                    "contracts": pd.NA,
                    "open_int": pd.NA,
                    "chg_in_oi": 0.0,
                    "market_max_date": pd.Timestamp("2024-01-10"),
                    "as_of_date_used": pd.Timestamp("2024-01-06"),
                    "end_date_used": pd.Timestamp("2024-01-06"),
                    "is_open": True,
                    "status": "OPEN",
                    "coverage_mode": coverage_mode,
                }
            ]
        )

        skips = pd.DataFrame(
            [
                {
                    "strategy_name": "demo",
                    "tenor": "WEEKLY",
                    "trade_id": "T1",
                    "leg_id": "L999",
                    "symbol": "NIFTY",
                    "instrument": "FUTIDX",
                    "expiry_dt": pd.Timestamp("2024-01-25"),
                    "strike_pr": 0.0,
                    "option_typ": pd.NA,
                    "entry_date": pd.Timestamp("2024-01-03"),
                    "exit_date": pd.Timestamp("2024-01-20"),
                    "market_max_date": pd.Timestamp("2024-01-10"),
                    "as_of_date_used": pd.Timestamp("2024-01-06"),
                    "end_date_used": pd.Timestamp("2024-01-06"),
                    "coverage_mode": coverage_mode,
                    "reason": "EMPTY_LEG_SLICE",
                    "details": "demo",
                }
            ]
        )
        return legs, skips

    monkeypatch.setattr(emod, "compute_legs_pnl", _fake_compute_legs_pnl)

    outdir = tmp_path / "out"

    run_phase2(
        input_path=in_path,
        outdir=outdir,
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-01-06"),
        symbol="NIFTY",
        tenor="WEEKLY",
        strategies=["demo"],
        coverage_mode="ASOF",
        as_of_override=None,
        strategy_overrides={},
    )

    assert captured["coverage_mode"] == "ASOF"
    assert captured["as_of_date"] == pd.Timestamp("2024-01-06")

    # Required artifact persisted
    assert (outdir / "skips_df.parquet").exists()
    assert (outdir / "legs_pnl_df.parquet").exists()
    assert (outdir / "trade_pnl_df.parquet").exists()
    assert (outdir / "strategy_pnl_df.parquet").exists()
    assert (outdir / "positions_df.parquet").exists()
    assert (outdir / "run_manifest.json").exists()


def test_positions_summary_open_leg_unrealized_only():
    legs = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "strategy_name": "s",
                "tenor": "WEEKLY",
                "trade_id": "T1",
                "leg_id": "L1",
                "symbol": "NIFTY",
                "instrument": "FUTIDX",
                "expiry_dt": pd.Timestamp("2024-01-25"),
                "strike_pr": 0.0,
                "option_typ": pd.NA,
                "side": 1,
                "qty_lots": 1,
                "entry_date": pd.Timestamp("2024-01-01"),
                "exit_date": pd.Timestamp("2024-01-20"),
                "market_max_date": pd.Timestamp("2024-01-10"),
                "as_of_date_used": pd.Timestamp("2024-01-05"),
                "end_date_used": pd.Timestamp("2024-01-05"),
                "coverage_mode": "ASOF",
                "status": "OPEN",
                "is_open": True,
                "mtm_pnl": 10.0,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "strategy_name": "s",
                "tenor": "WEEKLY",
                "trade_id": "T1",
                "leg_id": "L1",
                "symbol": "NIFTY",
                "instrument": "FUTIDX",
                "expiry_dt": pd.Timestamp("2024-01-25"),
                "strike_pr": 0.0,
                "option_typ": pd.NA,
                "side": 1,
                "qty_lots": 1,
                "entry_date": pd.Timestamp("2024-01-01"),
                "exit_date": pd.Timestamp("2024-01-20"),
                "market_max_date": pd.Timestamp("2024-01-10"),
                "as_of_date_used": pd.Timestamp("2024-01-05"),
                "end_date_used": pd.Timestamp("2024-01-05"),
                "coverage_mode": "ASOF",
                "status": "OPEN",
                "is_open": True,
                "mtm_pnl": -2.0,
            },
        ]
    )

    pos = build_positions_df(legs)
    assert len(pos) == 1
    assert float(pos.loc[0, "realized_pnl"]) == 0.0
    assert float(pos.loc[0, "unrealized_pnl"]) == pytest.approx(8.0)
    assert float(pos.loc[0, "total_pnl"]) == pytest.approx(8.0)


def test_positions_summary_closed_leg_realized_only():
    legs = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "strategy_name": "s",
                "tenor": "WEEKLY",
                "trade_id": "T1",
                "leg_id": "L1",
                "symbol": "NIFTY",
                "instrument": "FUTIDX",
                "expiry_dt": pd.Timestamp("2024-01-25"),
                "strike_pr": 0.0,
                "option_typ": pd.NA,
                "side": 1,
                "qty_lots": 1,
                "entry_date": pd.Timestamp("2024-01-01"),
                "exit_date": pd.Timestamp("2024-01-02"),
                "market_max_date": pd.Timestamp("2024-01-10"),
                "as_of_date_used": pd.Timestamp("2024-01-10"),
                "end_date_used": pd.Timestamp("2024-01-02"),
                "coverage_mode": "ASOF",
                "status": "CLOSED",
                "is_open": False,
                "mtm_pnl": 5.0,
            }
        ]
    )

    pos = build_positions_df(legs)
    assert len(pos) == 1
    assert float(pos.loc[0, "unrealized_pnl"]) == 0.0
    assert float(pos.loc[0, "realized_pnl"]) == pytest.approx(5.0)
    assert float(pos.loc[0, "total_pnl"]) == pytest.approx(5.0)


def test_positions_summary_totals_reconcile():
    legs = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "strategy_name": "s",
                "tenor": "WEEKLY",
                "trade_id": "T1",
                "leg_id": "L1",
                "symbol": "NIFTY",
                "instrument": "FUTIDX",
                "expiry_dt": pd.Timestamp("2024-01-25"),
                "strike_pr": 0.0,
                "option_typ": pd.NA,
                "side": 1,
                "qty_lots": 1,
                "entry_date": pd.Timestamp("2024-01-01"),
                "exit_date": pd.Timestamp("2024-01-20"),
                "market_max_date": pd.Timestamp("2024-01-10"),
                "as_of_date_used": pd.Timestamp("2024-01-05"),
                "end_date_used": pd.Timestamp("2024-01-05"),
                "coverage_mode": "ASOF",
                "status": "OPEN",
                "is_open": True,
                "mtm_pnl": 3.0,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "strategy_name": "s",
                "tenor": "WEEKLY",
                "trade_id": "T2",
                "leg_id": "L2",
                "symbol": "NIFTY",
                "instrument": "FUTIDX",
                "expiry_dt": pd.Timestamp("2024-01-25"),
                "strike_pr": 0.0,
                "option_typ": pd.NA,
                "side": 1,
                "qty_lots": 1,
                "entry_date": pd.Timestamp("2024-01-02"),
                "exit_date": pd.Timestamp("2024-01-02"),
                "market_max_date": pd.Timestamp("2024-01-10"),
                "as_of_date_used": pd.Timestamp("2024-01-10"),
                "end_date_used": pd.Timestamp("2024-01-02"),
                "coverage_mode": "ASOF",
                "status": "CLOSED",
                "is_open": False,
                "mtm_pnl": 7.0,
            },
        ]
    )
    pos = build_positions_df(legs)
    total = float((pos["realized_pnl"] + pos["unrealized_pnl"]).sum())
    assert total == pytest.approx(float(pos["cum_pnl_asof"].sum()))


def test_aggregations_preserve_asof_metadata_and_counts():
    legs = pd.DataFrame(
        [
            # Trade T1 has 2 legs: one OPEN, one CLOSED
            {
                "date": pd.Timestamp("2024-01-01"),
                "strategy_name": "s",
                "tenor": "WEEKLY",
                "trade_id": "T1",
                "leg_id": "L1",
                "mtm_pnl": 10.0,
                "gap_risk_pnl_proxy": 1.0,
                "market_max_date": pd.Timestamp("2024-01-10"),
                "as_of_date_used": pd.Timestamp("2024-01-05"),
                "end_date_used": pd.Timestamp("2024-01-05"),
                "coverage_mode": "ASOF",
                "status": "OPEN",
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "strategy_name": "s",
                "tenor": "WEEKLY",
                "trade_id": "T1",
                "leg_id": "L2",
                "mtm_pnl": -4.0,
                "gap_risk_pnl_proxy": 0.0,
                "market_max_date": pd.Timestamp("2024-01-10"),
                "as_of_date_used": pd.Timestamp("2024-01-05"),
                "end_date_used": pd.Timestamp("2024-01-05"),
                "coverage_mode": "ASOF",
                "status": "CLOSED",
            },
        ]
    )
    skips = pd.DataFrame(
        [
            {
                "strategy_name": "s",
                "tenor": "WEEKLY",
                "trade_id": "T1",
                "leg_id": "L999",
                "reason": "EMPTY_LEG_SLICE",
            }
        ]
    )

    trade = aggregate_legs_to_trade_pnl(legs_pnl_df=legs, skips_df=skips)
    assert {"market_max_date", "as_of_date_used", "coverage_mode"} <= set(trade.columns)
    assert {"n_open_legs", "n_closed_legs", "n_skipped_legs"} <= set(trade.columns)
    assert int(trade.loc[0, "n_open_legs"]) == 1
    assert int(trade.loc[0, "n_closed_legs"]) == 1
    assert int(trade.loc[0, "n_skipped_legs"]) == 1

    # Reconciliation: totals match
    assert float(trade["total_mtm_pnl"].sum()) == pytest.approx(float(legs["mtm_pnl"].sum()))

    strat = aggregate_trades_to_strategy_pnl(trade_pnl_df=trade)
    assert float(strat["total_mtm_pnl"].sum()) == pytest.approx(float(trade["total_mtm_pnl"].sum()))
