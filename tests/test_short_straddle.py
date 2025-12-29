import sys
import pathlib
import importlib
import importlib.util
from types import ModuleType

import pandas as pd
import numpy as np
import pytest


def _bootstrap_paths() -> pathlib.Path:
    """
    Make imports work when running:
      - pytest -q
      - python tests/test_short_straddle.py

    Adds:
      - repo root
      - repo root / 'src' (common layout where packages live inside src/)
    """
    this_file = pathlib.Path(__file__).resolve()
    repo_root = this_file.parents[1]

    for p in (repo_root, repo_root / "src"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    return repo_root


def _load_module_from_file(module_name: str, file_path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module={module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _find_first(repo_root: pathlib.Path, filename: str) -> pathlib.Path:
    hits = list(repo_root.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} under {repo_root}")
    hits = sorted(hits, key=lambda p: (len(p.parts), str(p)))  # deterministic
    return hits[0]


def _ensure_engine_loaded(repo_root: pathlib.Path) -> None:
    """
    If engine_pnl.py is not importable, load it by file path so that
    short_straddle.py's imports can resolve.
    """
    for name in ("engine_pnl", "src.engine_pnl"):
        try:
            importlib.import_module(name)
            return
        except Exception:
            pass

    try:
        engine_file = _find_first(repo_root, "engine_pnl.py")
        _load_module_from_file("engine_pnl", engine_file)
    except FileNotFoundError:
        return


def _import_data_integrity_error(repo_root: pathlib.Path):
    for mod_name in ("engine_pnl", "src.engine_pnl"):
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, "DataIntegrityError")
        except Exception:
            continue
    return Exception


def _import_short_straddle(repo_root: pathlib.Path):
    """
    Import strategy robustly across layouts:
      1) try normal imports after sys.path bootstrap
      2) fallback: locate and load short_straddle.py by file path
    """
    tried = []
    last_exc = None

    candidates = [
        ("strategies.short_straddle", "ShortStraddleStrategy", "ShortStraddleStrategyConfig"),
        ("src.strategies.short_straddle", "ShortStraddleStrategy", "ShortStraddleStrategyConfig"),
    ]

    for mod_path, cls1, cls2 in candidates:
        try:
            mod = importlib.import_module(mod_path)
            return getattr(mod, cls1), getattr(mod, cls2), _import_data_integrity_error(repo_root)
        except Exception as e:  # noqa: BLE001
            tried.append(mod_path)
            last_exc = e

    try:
        _ensure_engine_loaded(repo_root)
        short_file = _find_first(repo_root, "short_straddle.py")
        mod = _load_module_from_file("short_straddle_under_test", short_file)
        return (
            getattr(mod, "ShortStraddleStrategy"),
            getattr(mod, "ShortStraddleStrategyConfig"),
            _import_data_integrity_error(repo_root),
        )
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "Could not import ShortStraddleStrategy.\n"
            f"Tried module imports: {tried}\n"
            f"Last module import error: {repr(last_exc)}\n"
            f"Fallback file-load error: {repr(e)}\n"
            f"Repo root used: {repo_root}\n"
            "Fix: confirm short_straddle.py exists (expected: src/strategies/short_straddle.py)."
        ) from e


def _parent_trade_id(trade_id: str) -> str:
    # "....__CE" -> "...."
    return str(trade_id).split("__", 1)[0]


REPO_ROOT = _bootstrap_paths()
ShortStraddleStrategy, ShortStraddleStrategyConfig, DataIntegrityError = _import_short_straddle(REPO_ROOT)


def _make_min_market_df(liquid: bool = True, bad_intrinsic: bool = False) -> pd.DataFrame:
    """
    Synthetic dataset that:
      - Provides spot rows (instrument='SPOT') and option rows (instrument='OPTIDX')
      - Has a +500 move by expiry to cause straddle loss
      - Ensures monthly expiry marker exists on expiry date
    """
    entry = pd.Timestamp("2020-01-02")
    expiry = pd.Timestamp("2020-01-30")
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-29", "2020-01-30"])

    symbol = "NIFTY"
    instrument_opt = "OPTIDX"
    instrument_spot = "SPOT"
    strike = 12000
    lot = 75

    # +500 move by expiry
    spot_close = {
        pd.Timestamp("2020-01-02"): 12010.0,
        pd.Timestamp("2020-01-03"): 12000.0,
        pd.Timestamp("2020-01-29"): 12480.0,
        pd.Timestamp("2020-01-30"): 12500.0,
    }

    # CE becomes deep ITM near expiry; PE goes to ~0
    ce_settle = {
        pd.Timestamp("2020-01-02"): 100.0,
        pd.Timestamp("2020-01-03"): 95.0,
        pd.Timestamp("2020-01-29"): 480.0,
        pd.Timestamp("2020-01-30"): 500.0,  # intrinsic for spot=12500 strike=12000
    }
    pe_settle = {
        pd.Timestamp("2020-01-02"): 110.0,
        pd.Timestamp("2020-01-03"): 105.0,
        pd.Timestamp("2020-01-29"): 5.0,
        pd.Timestamp("2020-01-30"): 0.0,  # intrinsic
    }

    if bad_intrinsic:
        ce_settle[pd.Timestamp("2020-01-30")] = 450.0  # wrong vs intrinsic

    rows = []

    # Spot rows (non-option rows)
    for d in dates:
        rows.append(
            dict(
                date=d,
                symbol=symbol,
                instrument=instrument_spot,
                expiry_dt=d,  # placeholder for spot
                strike_pr=np.nan,
                option_typ=np.nan,
                close=spot_close[d],
                settle_pr=spot_close[d],
                lot_size=lot,
                expiry_rank=1,
                is_trading_day=True,
                is_opt_monthly_expiry=(d == expiry),
                open_int=np.nan,
                volume=np.nan,
            )
        )

    # Option rows (CE/PE)
    for d in dates:
        for opt_typ, settle_map in [("CE", ce_settle), ("PE", pe_settle)]:
            vol = 100.0
            if (not liquid) and (opt_typ == "PE") and (d == entry):
                vol = 0.0  # make PE illiquid on entry
            rows.append(
                dict(
                    date=d,
                    symbol=symbol,
                    instrument=instrument_opt,
                    expiry_dt=expiry,
                    strike_pr=strike,
                    option_typ=opt_typ,
                    close=settle_map[d],
                    settle_pr=settle_map[d],
                    lot_size=lot,
                    expiry_rank=1,
                    is_trading_day=True,
                    is_opt_monthly_expiry=(d == expiry),
                    open_int=1000.0,
                    volume=vol,
                )
            )

    return pd.DataFrame(rows)


def test_liquidity_abort_month():
    market_df = _make_min_market_df(liquid=False)
    cfg = ShortStraddleStrategyConfig(
        input_parquet_path="NA",
        symbol="NIFTY",
        strike_interval=50,
        spot_instrument_candidates=("SPOT",),
        option_instrument="OPTIDX",
    )
    strat = ShortStraddleStrategy(cfg)

    entry_days = pd.DataFrame({"symbol": ["NIFTY"], "entry_date": [pd.Timestamp("2020-01-02")]})
    trades = strat.build_trades(market_df, entry_days)
    assert trades.empty, "Must abort entire straddle if either leg is illiquid on entry."


def test_straddle_large_move_is_loss():
    market_df = _make_min_market_df(liquid=True, bad_intrinsic=False)
    cfg = ShortStraddleStrategyConfig(
        input_parquet_path="NA",
        symbol="NIFTY",
        strike_interval=50,
        spot_instrument_candidates=("SPOT",),
        option_instrument="OPTIDX",
        max_abort_ratio=1.0,
    )
    strat = ShortStraddleStrategy(cfg)

    entry_days = pd.DataFrame({"symbol": ["NIFTY"], "entry_date": [pd.Timestamp("2020-01-02")]})
    trades = strat.build_trades(market_df, entry_days)

    # New invariant: each leg gets its own trade_id, so we still expect 2 rows total here,
    # but trade_id values differ (parent__CE, parent__PE).
    assert len(trades) == 2
    assert set(trades["option_typ"]) == {"CE", "PE"}
    assert trades["trade_id"].nunique() == 2
    assert all(trades["trade_id"].astype(str).str.contains("__"))

    legs = strat.compute_mtm_pnl_rupee(market_df, trades)

    # Validate expiry intrinsic (should pass)
    strat._validate_expiry_intrinsic(market_df, legs_mtm=legs, trades_df=trades)

    # If your implementation provides _drop_unsynced_parent_trades, use it;
    # else fall back to no-op sync here (tests focus on P&L sign).
    if hasattr(strat, "_drop_unsynced_parent_trades"):
        legs = strat._drop_unsynced_parent_trades(legs)

    # Aggregate by parent trade id (strip __CE/__PE)
    legs = legs.copy()
    legs["parent_trade_id"] = legs["trade_id"].map(_parent_trade_id)

    daily = (
        legs.groupby(["date", "parent_trade_id", "symbol", "strike_pr"], as_index=False)["daily_pnl_rupee"]
        .sum()
        .rename(columns={"daily_pnl_rupee": "strategy_pnl_rupee", "parent_trade_id": "trade_id"})
        .sort_values(["trade_id", "date"], kind="mergesort")
    )
    daily["cum_pnl_rupee"] = daily.groupby("trade_id")["strategy_pnl_rupee"].cumsum()

    final = float(daily.loc[daily["date"] == pd.Timestamp("2020-01-30"), "cum_pnl_rupee"].iloc[0])

    # With +500 move, short CE loses heavily; net should be significantly negative.
    assert final < -10000.0, f"Expected significant loss on +500 move; got {final}"


def test_intrinsic_validation_raises():
    market_df = _make_min_market_df(liquid=True, bad_intrinsic=True)
    cfg = ShortStraddleStrategyConfig(
        input_parquet_path="NA",
        symbol="NIFTY",
        strike_interval=50,
        spot_instrument_candidates=("SPOT",),
        option_instrument="OPTIDX",
    )
    strat = ShortStraddleStrategy(cfg)

    entry_days = pd.DataFrame({"symbol": ["NIFTY"], "entry_date": [pd.Timestamp("2020-01-02")]})
    trades = strat.build_trades(market_df, entry_days)
    legs = strat.compute_mtm_pnl_rupee(market_df, trades)

    with pytest.raises(DataIntegrityError):
        strat._validate_expiry_intrinsic(market_df, legs_mtm=legs, trades_df=trades)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
