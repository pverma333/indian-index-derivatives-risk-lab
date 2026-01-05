"""
tests/smoke_test_engine_pnl_q1_2025.py

End-to-end smoke test that runs the project pipeline on curated Q1 2025 market data:

    market_df (CSV/Parquet)
      -> validate_market_df()
      -> settlement_marking.compute_settle_used()
      -> Phase2 strategies build_trades() (exercises contract_selectors.py, expiry_selectors.py internally)
      -> engine_pnl.compute_legs_pnl()  (ASOF/STRICT + OPEN/CLOSED + skips_df)

Fixes vs earlier versions
-------------------------
1) phase2_params.resolve_effective_strategy_params(): signature-safe call (no accidental positional tenor).
2) Some strategy builders don't emit entry_price. Engine requires it.
   We derive entry_price by joining to market_df.settle_used on entry_date + contract keys.

Outputs (written to --outdir)
-----------------------------
- market_used.parquet
- trades_df.parquet
- legs_pnl_df.parquet
- skips_df.parquet
- summary.csv
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

import pandas as pd

logger = logging.getLogger("smoke_test_engine_pnl_q1_2025")


def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "tests":
        return here.parent.parent
    return Path.cwd().resolve()


def _ensure_repo_on_syspath(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _read_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension: {path} (use .csv or .parquet)")


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    elif suf == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output extension: {path}")


def _normalize_dates(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()


def _clip_market_window(market_df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    if "date" not in market_df.columns or (not start_date and not end_date):
        return market_df

    d = pd.to_datetime(market_df["date"], errors="coerce").dt.normalize()
    out = market_df.copy()

    if start_date:
        s = pd.Timestamp(start_date).normalize()
        out = out.loc[d >= s].copy()
        d = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    if end_date:
        e = pd.Timestamp(end_date).normalize()
        out = out.loc[d <= e].copy()

    return out


def _parse_json_dict(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object for --strategy-overrides-json")
    return obj


def _expand_run_tenor(run_tenor: str) -> Tuple[str, ...]:
    t = (run_tenor or "").strip().upper()
    if t == "BOTH":
        return ("WEEKLY", "MONTHLY")
    if t in ("WEEKLY", "MONTHLY"):
        return (t,)
    raise ValueError(f"--tenor must be WEEKLY|MONTHLY|BOTH, got: {run_tenor!r}")


def _find_strategy_class(strategy_name: str) -> Type[Any]:
    """
    Try to import src.strategies.<strategy_name> and locate a class with build_trades().
    Falls back to known mappings.
    """
    s = strategy_name.strip().lower()

    mod_names = [f"src.strategies.{s}", f"src.strategies.{s}_strategy"]
    last_err: Optional[Exception] = None

    for mn in mod_names:
        try:
            mod = importlib.import_module(mn)
            candidates: List[Tuple[str, Type[Any]]] = []
            for name, obj in vars(mod).items():
                if isinstance(obj, type) and hasattr(obj, "build_trades"):
                    candidates.append((name, obj))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                return candidates[0][1]
        except Exception as e:
            last_err = e

    mapping = {
        "short_straddle": ("src.strategies.short_straddle", "ShortStraddleStrategy"),
        "short_strangle": ("src.strategies.short_strangle", "ShortStrangleStrategy"),
        "bull_call_spread": ("src.strategies.bull_call_spread", "BullCallSpreadStrategy"),
        "bear_put_spread": ("src.strategies.bear_put_spread", "BearPutSpreadStrategy"),
    }
    if s in mapping:
        mn, cn = mapping[s]
        mod = importlib.import_module(mn)
        return getattr(mod, cn)

    raise ImportError(f"Could not locate strategy class for {strategy_name!r}. Last import error: {last_err!r}")


def _resolve_params_signature_safe(
    phase2_params_mod: Any,
    *,
    strategy_name: str,
    tenor: str,
    run_cfg: Optional[Any],
    user_overrides: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Calls resolve_effective_strategy_params() with ONLY supported keyword args.
    Avoids passing tenor as positional (which can be interpreted as overrides).
    """
    if not hasattr(phase2_params_mod, "resolve_effective_strategy_params"):
        return {}

    fn = getattr(phase2_params_mod, "resolve_effective_strategy_params")
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}

    if "strategy_name" in sig.parameters:
        kwargs["strategy_name"] = strategy_name
    elif "s" in sig.parameters:
        kwargs["s"] = strategy_name

    if "tenor" in sig.parameters:
        kwargs["tenor"] = tenor

    if run_cfg is not None:
        if "run_cfg" in sig.parameters:
            kwargs["run_cfg"] = run_cfg
        elif "run_config" in sig.parameters:
            kwargs["run_config"] = run_cfg

    if user_overrides is None:
        user_overrides = {}

    if "user_overrides" in sig.parameters:
        kwargs["user_overrides"] = dict(user_overrides)
    elif "overrides" in sig.parameters:
        kwargs["overrides"] = dict(user_overrides)
    elif "strategy_overrides" in sig.parameters:
        kwargs["strategy_overrides"] = dict(user_overrides)

    try:
        if kwargs:
            if ("strategy_name" not in sig.parameters) and ("s" not in sig.parameters):
                return fn(strategy_name, **kwargs)  # type: ignore
            return fn(**kwargs)  # type: ignore
        return fn(strategy_name)  # type: ignore
    except TypeError:
        return fn(strategy_name)  # type: ignore


def _filter_overrides_if_possible(phase2_params_mod: Any, strategy_name: str, overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Filter overrides to allowed keys if helper exists; else ignore overrides to avoid ConfigError.
    """
    if not overrides:
        return {}

    for helper_name in ("_allowed_strategy_param_keys", "allowed_strategy_param_keys", "get_allowed_strategy_param_keys"):
        if hasattr(phase2_params_mod, helper_name):
            helper = getattr(phase2_params_mod, helper_name)
            try:
                allowed = set(helper(strategy_name))
                filtered = {k: v for k, v in overrides.items() if k in allowed}
                dropped = sorted(set(overrides) - set(filtered))
                if dropped:
                    logger.warning("Dropping unknown strategy override keys for %s: %s", strategy_name, dropped)
                return filtered
            except Exception:
                break

    logger.warning("No allowed-keys helper found; ignoring overrides for %s to avoid ConfigError", strategy_name)
    return {}


def _build_trades_phase2(
    market_df: pd.DataFrame,
    *,
    symbol: str,
    run_tenor: str,
    strategies: Sequence[str],
    run_cfg: Optional[Any],
    strategy_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    max_trades: Optional[int] = None,
) -> pd.DataFrame:
    phase2_params_mod = importlib.import_module("src.config.phase2_params")
    strategy_overrides = strategy_overrides or {}

    all_trades: List[pd.DataFrame] = []
    tenors = _expand_run_tenor(run_tenor)

    for strat in strategies:
        strat_name = strat.strip().lower()
        StratCls = _find_strategy_class(strat_name)
        strat_obj = StratCls()

        for tenor in tenors:
            overrides_raw = dict(strategy_overrides.get(strat_name, {}) or {})
            overrides = _filter_overrides_if_possible(phase2_params_mod, strat_name, overrides_raw)

            params = _resolve_params_signature_safe(
                phase2_params_mod,
                strategy_name=strat_name,
                tenor=tenor,
                run_cfg=run_cfg,
                user_overrides=overrides,
            )

            cfg = {"symbol": symbol.strip().upper(), "tenor": tenor, **(params or {})}

            logger.info("Building trades: strategy=%s tenor=%s", strat_name, tenor)
            trades_df = strat_obj.build_trades(market_df=market_df, cfg=cfg)

            if trades_df is None or len(trades_df) == 0:
                logger.info("No trades returned: strategy=%s tenor=%s", strat_name, tenor)
                continue

            all_trades.append(trades_df)

    if not all_trades:
        return pd.DataFrame()

    out = pd.concat(all_trades, ignore_index=True)

    if max_trades is not None and max_trades > 0 and "trade_id" in out.columns:
        keep = out["trade_id"].astype(str).drop_duplicates().head(max_trades).tolist()
        out = out.loc[out["trade_id"].astype(str).isin(keep)].copy()

    sort_cols = [c for c in ["strategy_name", "tenor", "trade_id", "leg_id", "entry_date"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return out


def _ensure_entry_price(trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure trades_df has entry_price.
    If missing/null, derive entry_price from market_df.settle_used on entry_date and contract keys.
    """
    if trades_df.empty:
        return trades_df

    if "entry_price" in trades_df.columns and trades_df["entry_price"].notna().all():
        return trades_df

    if "settle_used" not in market_df.columns:
        raise AssertionError("market_df missing settle_used; run compute_settle_used() first.")

    t = trades_df.copy()
    m = market_df.copy()

    _normalize_dates(t, ["entry_date", "expiry_dt"])
    _normalize_dates(m, ["date", "expiry_dt"])

    if "entry_price" not in t.columns:
        t["entry_price"] = pd.NA

    is_opt = (t["instrument"].astype(str) == "OPTIDX") if "instrument" in t.columns else pd.Series(False, index=t.index)

    join_left = ["entry_date", "symbol", "instrument", "expiry_dt"]
    join_right = ["date", "symbol", "instrument", "expiry_dt"]

    missing_needed = [c for c in join_left if c not in t.columns]
    if missing_needed:
        raise AssertionError(f"Cannot derive entry_price; trades_df missing join keys: {missing_needed}")

    keep_cols = join_right + ["settle_used"]
    if "strike_pr" in m.columns:
        keep_cols.append("strike_pr")
    if "option_typ" in m.columns:
        keep_cols.append("option_typ")
    m_lu = m[keep_cols].copy()

    # Non-options
    t_nonopt = t.loc[~is_opt].copy()
    if not t_nonopt.empty:
        merged = t_nonopt.merge(
            m_lu[join_right + ["settle_used"]],
            how="left",
            left_on=join_left,
            right_on=join_right,
        )
        t.loc[t_nonopt.index, "_entry_price_derived"] = merged["settle_used"].values

    # Options
    t_opt = t.loc[is_opt].copy()
    if not t_opt.empty:
        need_opt_cols = [c for c in ["strike_pr", "option_typ"] if c not in t_opt.columns]
        if need_opt_cols:
            raise AssertionError(f"Cannot derive entry_price for OPTIDX legs; trades_df missing: {need_opt_cols}")
        for c in ["strike_pr", "option_typ"]:
            if c not in m_lu.columns:
                raise AssertionError(f"Cannot derive entry_price for OPTIDX legs; market_df missing: {c}")

        left_opt = join_left + ["strike_pr", "option_typ"]
        right_opt = join_right + ["strike_pr", "option_typ"]
        merged = t_opt.merge(
            m_lu[right_opt + ["settle_used"]],
            how="left",
            left_on=left_opt,
            right_on=right_opt,
        )
        t.loc[t_opt.index, "_entry_price_derived"] = merged["settle_used"].values

    t["entry_price"] = t["entry_price"].where(t["entry_price"].notna(), t["_entry_price_derived"])
    t = t.drop(columns=["_entry_price_derived"], errors="ignore")

    if t["entry_price"].isna().any():
        bad = t.loc[
            t["entry_price"].isna(),
            ["strategy_name", "tenor", "trade_id", "leg_id", "symbol", "instrument", "expiry_dt", "entry_date"],
        ].head(20)
        raise AssertionError(
            "Derived entry_price still has NaNs (first 20). Missing market settle_used rows for entry_date+contract keys.\n"
            + bad.to_string(index=False)
        )

    logger.info("Derived entry_price for legs where strategy output did not provide it.")
    return t


def _assert_required_trade_cols(trades_df: pd.DataFrame) -> None:
    required = [
        "strategy_name", "tenor", "trade_id", "leg_id",
        "symbol", "instrument", "expiry_dt",
        "entry_date", "exit_date", "side", "qty_lots",
    ]
    missing = [c for c in required if c not in trades_df.columns]
    if missing:
        raise AssertionError(f"trades_df missing required columns for engine_pnl: {missing}")


def _assert_day0_anchor(legs_pnl_df: pd.DataFrame) -> None:
    if legs_pnl_df.empty:
        return
    df = legs_pnl_df.sort_values(["trade_id", "leg_id", "date"], kind="mergesort").copy()
    first = df.groupby(["trade_id", "leg_id"], as_index=False).head(1)
    sp = pd.to_numeric(first["settle_prev_used"], errors="coerce")
    ep = pd.to_numeric(first["entry_price"], errors="coerce")
    bad = first.loc[(sp.round(10) != ep.round(10)) | sp.isna() | ep.isna()]
    if not bad.empty:
        sample = bad[["trade_id", "leg_id", "date", "settle_prev_used", "entry_price"]].head(20)
        raise AssertionError("Day-0 anchor failed (first 20):\n" + sample.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/curated/derivatives_clean_Q1_2025.csv")
    parser.add_argument("--start-date", type=str, default="")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--symbol", type=str, default="")
    parser.add_argument("--tenor", type=str, default="")
    parser.add_argument("--strategies", type=str, default="")
    parser.add_argument("--coverage-mode", type=str, default="ASOF", choices=["ASOF", "STRICT"])
    parser.add_argument("--as-of-date", type=str, default="")
    parser.add_argument("--max-trades", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="data/derived/smoke/engine_pnl_q1_2025")
    parser.add_argument("--strategy-overrides-json", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    repo_root = _repo_root_from_here()
    _ensure_repo_on_syspath(repo_root)

    from src.validation.market_df_validator import validate_market_df
    from src.engine.settlement_marking import compute_settle_used
    from src.engine.engine_pnl import compute_legs_pnl

    phase2_params = importlib.import_module("src.config.phase2_params")
    run_cfg = phase2_params.get_phase2_default_run_config() if hasattr(phase2_params, "get_phase2_default_run_config") else None

    in_path = Path(args.path)
    logger.info("Reading market dataset: %s", in_path)
    market_df = _read_df(in_path)
    _normalize_dates(market_df, ["date", "expiry_dt"])

    start_date = args.start_date.strip() or None
    end_date = args.end_date.strip() or None
    if start_date or end_date:
        logger.info("Filtering market_df to window: start=%s end=%s", start_date, end_date)
        market_df = _clip_market_window(market_df, start_date, end_date)

    logger.info("Validating market_df contract via src.validation.market_df_validator.validate_market_df")
    validate_market_df(market_df)
    logger.info("market_df validation passed")

    logger.info("Computing settle_used via src.engine.settlement_marking.compute_settle_used")
    market_df = compute_settle_used(market_df)

    symbol = args.symbol.strip().upper() if args.symbol.strip() else (getattr(run_cfg, "symbol", "NIFTY") if run_cfg is not None else "NIFTY")
    tenor = args.tenor.strip().upper() if args.tenor.strip() else (getattr(run_cfg, "tenor", "BOTH") if run_cfg is not None else "BOTH")

    if args.strategies.strip():
        strategies = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
    else:
        strategies = list(getattr(run_cfg, "strategies", ["short_straddle"])) if run_cfg is not None else ["short_straddle"]

    overrides_raw = _parse_json_dict(args.strategy_overrides_json)
    overrides: Dict[str, Dict[str, Any]] = {}
    for k, v in overrides_raw.items():
        if isinstance(v, dict):
            overrides[str(k).strip().lower()] = v
        else:
            raise ValueError(f"Override for '{k}' must be a dict; got {type(v)}")

    max_trades = args.max_trades if args.max_trades and args.max_trades > 0 else None

    logger.info("Building trades via Phase2 strategies: %s", strategies)
    trades_df = _build_trades_phase2(
        market_df=market_df,
        symbol=symbol,
        run_tenor=tenor,
        strategies=strategies,
        run_cfg=run_cfg,
        strategy_overrides=overrides,
        max_trades=max_trades,
    )

    if trades_df.empty:
        raise RuntimeError("No trades produced by strategies. Try --strategies short_straddle or adjust date window.")

    _normalize_dates(trades_df, ["entry_date", "exit_date", "expiry_dt"])
    _assert_required_trade_cols(trades_df)
    trades_df = _ensure_entry_price(trades_df, market_df)

    as_of_date = pd.Timestamp(args.as_of_date).normalize() if args.as_of_date.strip() else None

    logger.info("Running engine_pnl.compute_legs_pnl: coverage_mode=%s as_of_date=%s", args.coverage_mode, as_of_date)
    legs_pnl_df, skips_df = compute_legs_pnl(
        market_df=market_df,
        trades_df=trades_df,
        coverage_mode=args.coverage_mode,
        as_of_date=as_of_date,
    )

    if legs_pnl_df.empty and skips_df.empty:
        raise AssertionError("engine_pnl returned empty legs_pnl_df and empty skips_df")

    if not legs_pnl_df.empty:
        if "settle_prev_used" in legs_pnl_df.columns and legs_pnl_df["settle_prev_used"].isna().any():
            bad = legs_pnl_df.loc[legs_pnl_df["settle_prev_used"].isna(), ["trade_id", "leg_id", "date"]].head(20)
            raise AssertionError("Found NaN settle_prev_used in legs_pnl_df (first 20):\n" + bad.to_string(index=False))
        _assert_day0_anchor(legs_pnl_df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _write_df(market_df, outdir / "market_used.parquet")
    _write_df(trades_df, outdir / "trades_df.parquet")
    _write_df(legs_pnl_df, outdir / "legs_pnl_df.parquet")
    _write_df(skips_df, outdir / "skips_df.parquet")

    summary = pd.DataFrame(
        [{
            "input_path": str(in_path),
            "rows_market_used": int(len(market_df)),
            "rows_trades": int(len(trades_df)),
            "unique_trades": int(trades_df["trade_id"].astype(str).nunique()) if "trade_id" in trades_df.columns else 0,
            "rows_legs_pnl": int(len(legs_pnl_df)),
            "rows_skips": int(len(skips_df)),
            "coverage_mode": args.coverage_mode,
            "as_of_date": str(as_of_date.date()) if as_of_date is not None else "",
            "symbol": symbol,
            "tenor": tenor,
            "strategies": ",".join(strategies),
            "start_date": start_date or "",
            "end_date": end_date or "",
        }]
    )
    _write_df(summary, outdir / "summary.csv")

    logger.info("SMOKE TEST PASSED. Outputs: %s", outdir)
    print("SMOKE TEST PASSED")
    print(f"Outputs: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
