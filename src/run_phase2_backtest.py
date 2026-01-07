from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root (parent of src/)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.reporting.artifacts import write_df, write_json
from src.reporting.positions_summary import build_positions_df
from src.strategies.aggregations import aggregate_legs_to_trade_pnl, aggregate_trades_to_strategy_pnl

logger = logging.getLogger(__name__)


def _read_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension: {path} (use .csv or .parquet)")


def _normalize_dates(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="raise").dt.normalize()


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
    Uses the signature-safe calling approach from the script pack.
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


def _filter_overrides_if_possible(
    phase2_params_mod: Any,
    strategy_name: str,
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Mirrors script pack behavior: if helper exists, filter; else ignore overrides
    to avoid ConfigError.
    """
    if not overrides:
        return {}

    for helper_name in ("_allowed_strategy_param_keys", "allowed_strategy_param_keys", "get_allowed_strategy_param_keys"):
        if hasattr(phase2_params_mod, helper_name):
            helper = getattr(phase2_params_mod, helper_name)
            try:
                allowed = set(helper(strategy_name))
                return {k: v for k, v in overrides.items() if k in allowed}
            except Exception:
                break

    logger.warning("No allowed-keys helper found; ignoring overrides for %s to avoid ConfigError", strategy_name)
    return {}


def _ensure_entry_price(trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive missing entry_price by joining to market_df.settle_used on entry_date+contract keys.
    This is required because engine requires entry_price not-null.
    """
    t = trades_df.copy()
    if "entry_price" not in t.columns:
        t["entry_price"] = pd.NA

    if t["entry_price"].notna().all():
        return t

    required_mkt = ["date", "symbol", "instrument", "expiry_dt", "settle_used"]
    miss_mkt = [c for c in required_mkt if c not in market_df.columns]
    if miss_mkt:
        raise AssertionError(f"Cannot derive entry_price; market_df missing: {miss_mkt}")

    m = market_df.copy()
    _normalize_dates(m, ["date", "expiry_dt"])
    _normalize_dates(t, ["entry_date", "expiry_dt"])

    join_left = ["symbol", "instrument", "expiry_dt", "entry_date"]
    join_right = ["symbol", "instrument", "expiry_dt", "date"]

    t["_entry_price_derived"] = pd.NA

    is_opt = t["instrument"].astype(str) == "OPTIDX"

    # Non-options
    t_non = t.loc[~is_opt].copy()
    if not t_non.empty:
        merged = t_non.merge(
            m[join_right + ["settle_used"]],
            how="left",
            left_on=join_left,
            right_on=join_right,
        )
        t.loc[t_non.index, "_entry_price_derived"] = merged["settle_used"].values

    # Options
    t_opt = t.loc[is_opt].copy()
    if not t_opt.empty:
        need_t = [c for c in ["strike_pr", "option_typ"] if c not in t_opt.columns]
        if need_t:
            raise AssertionError(f"Cannot derive entry_price for OPTIDX legs; trades_df missing: {need_t}")
        need_m = [c for c in ["strike_pr", "option_typ"] if c not in m.columns]
        if need_m:
            raise AssertionError(f"Cannot derive entry_price for OPTIDX legs; market_df missing: {need_m}")

        left_opt = join_left + ["strike_pr", "option_typ"]
        right_opt = join_right + ["strike_pr", "option_typ"]
        merged = t_opt.merge(
            m[right_opt + ["settle_used"]],
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


def compute_market_max_date_symbol(market_df: pd.DataFrame, symbol: str) -> pd.Timestamp:
    if market_df is None or market_df.empty:
        raise ValueError("market_df empty; cannot compute market_max_date")

    if "symbol" not in market_df.columns or "date" not in market_df.columns:
        raise ValueError("market_df must include 'symbol' and 'date' to compute market_max_date")

    sym = str(symbol).strip().upper()
    m = market_df.loc[market_df["symbol"].astype(str).str.upper() == sym].copy()
    if m.empty:
        raise ValueError(f"Market data empty after symbol filtering: {sym}")

    d = pd.to_datetime(m["date"], errors="raise").dt.normalize()
    mx = d.max()
    if pd.isna(mx):
        raise ValueError("market_max_date cannot be determined (no valid dates)")
    return pd.Timestamp(mx).normalize()


def compute_as_of_date_used(
    *,
    user_end_date: Optional[pd.Timestamp],
    market_max_date: pd.Timestamp,
    as_of_override: Optional[pd.Timestamp],
) -> pd.Timestamp:
    if as_of_override is not None:
        return min(pd.Timestamp(as_of_override).normalize(), pd.Timestamp(market_max_date).normalize())
    if user_end_date is None:
        return pd.Timestamp(market_max_date).normalize()
    return min(pd.Timestamp(user_end_date).normalize(), pd.Timestamp(market_max_date).normalize())


def filter_market_for_run_window(
    market_df: pd.DataFrame,
    *,
    symbol: str,
    start_date: pd.Timestamp,
    as_of_date_used: pd.Timestamp,
) -> pd.DataFrame:
    if market_df is None or market_df.empty:
        raise ValueError("market_df empty")

    m = market_df.copy()
    _normalize_dates(m, ["date", "expiry_dt"])

    sym = str(symbol).strip().upper()
    m = m.loc[m["symbol"].astype(str).str.upper() == sym].copy()
    if m.empty:
        raise ValueError(f"Market data empty after symbol filtering: {sym}")

    d = m["date"]
    out = m.loc[(d >= pd.Timestamp(start_date).normalize()) & (d <= pd.Timestamp(as_of_date_used).normalize())].copy()
    return out


def _build_trades_phase2(
    market_df: pd.DataFrame,
    *,
    symbol: str,
    run_tenor: str,
    strategies: Sequence[str],
    run_cfg: Optional[Any],
    strategy_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Returns:
      - trades_df (concat across strategy × tenor)
      - effective_cfgs[strategy_name][tenor] = resolved effective config dict
    """
    phase2_params_mod = importlib.import_module("src.config.phase2_params")
    strategy_overrides = strategy_overrides or {}

    effective_cfgs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_trades: List[pd.DataFrame] = []
    tenors = _expand_run_tenor(run_tenor)

    for strat in strategies:
        strat_name = strat.strip().lower()
        StratCls = _find_strategy_class(strat_name)
        strat_obj = StratCls()

        effective_cfgs.setdefault(strat_name, {})

        for ten in tenors:
            overrides_raw = dict(strategy_overrides.get(strat_name, {}) or {})
            overrides = _filter_overrides_if_possible(phase2_params_mod, strat_name, overrides_raw)

            params = _resolve_params_signature_safe(
                phase2_params_mod,
                strategy_name=strat_name,
                tenor=ten,
                run_cfg=run_cfg,
                user_overrides=overrides,
            )

            # Validate strategy params if available
            if hasattr(phase2_params_mod, "validate_strategy_params"):
                phase2_params_mod.validate_strategy_params(strat_name, params, ten)

            cfg = {"symbol": symbol.strip().upper(), "tenor": ten, **(params or {})}
            effective_cfgs[strat_name][ten] = dict(cfg)

            logger.info("Building trades: strategy=%s tenor=%s", strat_name, ten)
            trades_df = strat_obj.build_trades(market_df=market_df, cfg=cfg)
            if trades_df is None or len(trades_df) == 0:
                logger.info("No trades returned: strategy=%s tenor=%s", strat_name, ten)
                continue
            all_trades.append(trades_df)

    if not all_trades:
        return pd.DataFrame(), effective_cfgs

    out = pd.concat(all_trades, ignore_index=True)
    sort_cols = [c for c in ["strategy_name", "tenor", "trade_id", "leg_id", "entry_date"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return out, effective_cfgs


def build_run_manifest(
    *,
    input_path: str,
    symbol: str,
    tenor: str,
    start_date: pd.Timestamp,
    user_end_date: Optional[pd.Timestamp],
    market_max_date: pd.Timestamp,
    as_of_date_used: pd.Timestamp,
    coverage_mode: str,
    dataset_rows: int,
    dataset_min_date: pd.Timestamp,
    dataset_max_date: pd.Timestamp,
    raw_overrides: Dict[str, Any],
    effective_cfgs: Dict[str, Dict[str, Dict[str, Any]]],
    legs_pnl_df: pd.DataFrame,
    skips_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> Dict[str, Any]:
    # Artifact-driven leg-status counts
    if legs_pnl_df is not None and not legs_pnl_df.empty:
        leg_keys = ["strategy_name", "tenor", "trade_id", "leg_id"]
        leg_level = (
            legs_pnl_df.sort_values(leg_keys + ["date"], kind="mergesort")
            .groupby(leg_keys, as_index=False)
            .agg(status=("status", "first"))
        )
    else:
        leg_level = pd.DataFrame(columns=["strategy_name", "tenor", "trade_id", "leg_id", "status"])

    # Artifact-driven skips-by-reason
    if skips_df is not None and not skips_df.empty:
        need = ["strategy_name", "tenor", "trade_id", "leg_id", "reason"]
        miss = [c for c in need if c not in skips_df.columns]
        if miss:
            raise ValueError(f"skips_df missing required columns for manifest: {miss}")
        skip_level = skips_df[need].drop_duplicates()
    else:
        skip_level = pd.DataFrame(columns=["strategy_name", "tenor", "trade_id", "leg_id", "reason"])

    strat_pairs = set()
    if trades_df is not None and not trades_df.empty and {"strategy_name", "tenor"} <= set(trades_df.columns):
        for x in trades_df[["strategy_name", "tenor"]].dropna().astype(str).drop_duplicates().to_numpy():
            strat_pairs.add((x[0], x[1]))
    for x in leg_level[["strategy_name", "tenor"]].drop_duplicates().astype(str).to_numpy():
        strat_pairs.add((x[0], x[1]))
    for x in skip_level[["strategy_name", "tenor"]].drop_duplicates().astype(str).to_numpy():
        strat_pairs.add((x[0], x[1]))

    strategies_block: Dict[str, Any] = {}

    for strat_name, ten in sorted(strat_pairs):
        # n_trades / n_cycles from trades_df (best-effort, deterministic)
        n_trades = 0
        n_cycles = 0
        if trades_df is not None and not trades_df.empty:
            tsub = trades_df[
                (trades_df["strategy_name"].astype(str) == strat_name)
                & (trades_df["tenor"].astype(str) == ten)
            ]
            if not tsub.empty:
                if "trade_id" in tsub.columns:
                    n_trades = int(tsub["trade_id"].astype(str).nunique())
                if "expiry_dt" in tsub.columns and "entry_date" in tsub.columns:
                    n_cycles = int(tsub[["expiry_dt", "entry_date"]].drop_duplicates().shape[0])
                else:
                    n_cycles = n_trades

        lsub = leg_level[
            (leg_level["strategy_name"].astype(str) == strat_name)
            & (leg_level["tenor"].astype(str) == ten)
        ]
        n_legs_emitted = int(len(lsub))
        n_open_legs = int((lsub["status"].astype(str) == "OPEN").sum()) if not lsub.empty else 0
        n_closed_legs = int((lsub["status"].astype(str) == "CLOSED").sum()) if not lsub.empty else 0

        ssub = skip_level[
            (skip_level["strategy_name"].astype(str) == strat_name)
            & (skip_level["tenor"].astype(str) == ten)
        ]
        n_skipped_legs = int(len(ssub.drop_duplicates(subset=["strategy_name", "tenor", "trade_id", "leg_id"]))) if not ssub.empty else 0
        skips_by_reason = (
            ssub.groupby("reason").size().sort_index().to_dict()
            if (not ssub.empty and "reason" in ssub.columns)
            else {}
        )

        pnl_totals = {}
        if legs_pnl_df is not None and not legs_pnl_df.empty:
            raw = legs_pnl_df[
                (legs_pnl_df["strategy_name"].astype(str) == strat_name)
                & (legs_pnl_df["tenor"].astype(str) == ten)
            ]
            if not raw.empty:
                pnl_totals = {
                    "mtm_pnl_sum": float(pd.to_numeric(raw["mtm_pnl"], errors="coerce").fillna(0.0).sum()),
                    "gap_risk_pnl_proxy_sum": float(
                        pd.to_numeric(raw.get("gap_risk_pnl_proxy", 0.0), errors="coerce").fillna(0.0).sum()
                    ),
                }

        strategies_block.setdefault(strat_name, {})
        strategies_block[strat_name][ten] = {
            "n_cycles": n_cycles,
            "n_trades": n_trades,
            "n_legs_emitted": n_legs_emitted,
            "n_open_legs": n_open_legs,
            "n_closed_legs": n_closed_legs,
            "n_skipped_legs": n_skipped_legs,
            "skips_by_reason": skips_by_reason,
            "pnl_totals": pnl_totals,
            "effective_config": effective_cfgs.get(strat_name, {}).get(ten, {}),
        }

    manifest: Dict[str, Any] = {
        "inputs": {"path": str(input_path)},
        "symbol": symbol,
        "tenor": tenor,
        "start_date": pd.Timestamp(start_date).date().isoformat(),
        "user_end_date": None if user_end_date is None else pd.Timestamp(user_end_date).date().isoformat(),
        # Ticket requires market_max_date computed from symbol-filtered market_df BEFORE date filtering
        "market_max_date": pd.Timestamp(market_max_date).date().isoformat(),
        "as_of_date_used": pd.Timestamp(as_of_date_used).date().isoformat(),
        "coverage_mode": str(coverage_mode),
        "dataset": {
            "rows": int(dataset_rows),
            "min_date": pd.Timestamp(dataset_min_date).date().isoformat(),
            "max_date": pd.Timestamp(dataset_max_date).date().isoformat(),
        },
        "raw_strategy_overrides": raw_overrides,
        "strategies": strategies_block,
        "counts": {
            "rows_trades": int(0 if trades_df is None else len(trades_df)),
            "rows_legs_pnl": int(0 if legs_pnl_df is None else len(legs_pnl_df)),
            "rows_skips": int(0 if skips_df is None else len(skips_df)),
        },
    }
    return manifest


def run_phase2(
    *,
    input_path: Path,
    outdir: Path,
    start_date: pd.Timestamp,
    end_date: Optional[pd.Timestamp],
    symbol: str,
    tenor: str,
    strategies: Sequence[str],
    coverage_mode: str,
    as_of_override: Optional[pd.Timestamp],
    strategy_overrides: Dict[str, Dict[str, Any]],
) -> None:
    from src.validation.market_df_validator import validate_market_df
    from src.engine.settlement_marking import compute_settle_used
    from src.engine.engine_pnl import compute_legs_pnl

    phase2_params = importlib.import_module("src.config.phase2_params")
    run_cfg = phase2_params.get_phase2_default_run_config() if hasattr(phase2_params, "get_phase2_default_run_config") else None

    if hasattr(phase2_params, "validate_run_config") and run_cfg is not None:
        phase2_params.validate_run_config(run_cfg)

    market_df = _read_df(input_path)
    _normalize_dates(market_df, ["date", "expiry_dt"])

    symbol = str(symbol).strip().upper()

    # Compute market_max_date BEFORE any date filtering (critical nuance)
    market_max_date = compute_market_max_date_symbol(market_df, symbol)

    user_end_date = pd.Timestamp(end_date).normalize() if end_date is not None else None
    as_of_date_used = compute_as_of_date_used(
        user_end_date=user_end_date,
        market_max_date=market_max_date,
        as_of_override=as_of_override,
    )

    # Filter market_df for run window: date in [start_date, as_of_date_used] (do not pre-drop)
    market_df_run = filter_market_for_run_window(
        market_df,
        symbol=symbol,
        start_date=start_date,
        as_of_date_used=as_of_date_used,
    )
    if market_df_run.empty:
        raise RuntimeError(
            f"Market data empty after applying run window: symbol={symbol} start={start_date.date().isoformat()} as_of={as_of_date_used.date().isoformat()}"
        )

    # Dataset info after symbol filter (recommended by ticket)
    market_df_sym = market_df.loc[market_df["symbol"].astype(str).str.upper() == symbol].copy()
    dataset_rows = int(len(market_df_sym))
    dataset_min_date = pd.to_datetime(market_df_sym["date"], errors="raise").dt.normalize().min()
    dataset_max_date = pd.to_datetime(market_df_sym["date"], errors="raise").dt.normalize().max()

    validate_market_df(market_df_run)
    market_df_run = compute_settle_used(market_df_run)

    trades_df, effective_cfgs = _build_trades_phase2(
        market_df=market_df_run,
        symbol=symbol,
        run_tenor=tenor,
        strategies=strategies,
        run_cfg=run_cfg,
        strategy_overrides=strategy_overrides,
    )
    if trades_df.empty:
        raise RuntimeError("No trades produced by strategies. Adjust window/strategies/overrides.")

    _normalize_dates(trades_df, ["entry_date", "exit_date", "expiry_dt"])
    trades_df = _ensure_entry_price(trades_df, market_df_run)

    # Engine invocation must pass coverage_mode + as_of_date_used
    legs_pnl_df, skips_df = compute_legs_pnl(
        market_df=market_df_run,
        trades_df=trades_df,
        coverage_mode=coverage_mode,
        as_of_date=as_of_date_used,
    )

    # Stamp runner-level market_max_date into artifacts (the ticket’s definition)
    if legs_pnl_df is not None and not legs_pnl_df.empty:
        legs_pnl_df = legs_pnl_df.copy()
        legs_pnl_df["market_max_date"] = market_max_date
    if skips_df is not None and not skips_df.empty:
        skips_df = skips_df.copy()
        skips_df["market_max_date"] = market_max_date

    trade_pnl_df = aggregate_legs_to_trade_pnl(legs_pnl_df=legs_pnl_df, skips_df=skips_df)
    strategy_pnl_df = aggregate_trades_to_strategy_pnl(trade_pnl_df=trade_pnl_df)

    positions_df = build_positions_df(legs_pnl_df)

    # Positions reconcile check (ticket acceptance criteria)
    if not positions_df.empty:
        pos_total = float(pd.to_numeric(positions_df["realized_pnl"], errors="coerce").fillna(0.0).sum()) + float(
            pd.to_numeric(positions_df["unrealized_pnl"], errors="coerce").fillna(0.0).sum()
        )
        leg_total = float(pd.to_numeric(positions_df["cum_pnl_asof"], errors="coerce").fillna(0.0).sum())
        if abs(pos_total - leg_total) > 1e-6:
            raise AssertionError(f"Positions reconcile failed: realized+unrealized={pos_total} vs cum_pnl_asof={leg_total}")

    manifest = build_run_manifest(
        input_path=str(input_path),
        symbol=symbol,
        tenor=tenor,
        start_date=start_date,
        user_end_date=user_end_date,
        market_max_date=market_max_date,
        as_of_date_used=as_of_date_used,
        coverage_mode=coverage_mode,
        dataset_rows=dataset_rows,
        dataset_min_date=dataset_min_date,
        dataset_max_date=dataset_max_date,
        raw_overrides={k: v for k, v in strategy_overrides.items()},
        effective_cfgs=effective_cfgs,
        legs_pnl_df=legs_pnl_df,
        skips_df=skips_df,
        trades_df=trades_df,
    )

    outdir.mkdir(parents=True, exist_ok=True)

    # Required outputs per ticket
    write_df(legs_pnl_df, outdir / "legs_pnl_df.parquet")
    write_df(skips_df, outdir / "skips_df.parquet")
    write_df(trade_pnl_df, outdir / "trade_pnl_df.parquet")
    write_df(strategy_pnl_df, outdir / "strategy_pnl_df.parquet")
    write_df(positions_df, outdir / "positions_df.parquet")
    write_json(manifest, outdir / "run_manifest.json")

    # Optional but useful debug artifacts
    write_df(market_df_run, outdir / "market_used.parquet")
    write_df(trades_df, outdir / "trades_df.parquet")

    logger.info(
        "Phase2 complete: outdir=%s rows_market_used=%d rows_trades=%d rows_legs=%d rows_skips=%d",
        str(outdir),
        int(len(market_df_run)),
        int(len(trades_df)),
        int(0 if legs_pnl_df is None else len(legs_pnl_df)),
        int(0 if skips_df is None else len(skips_df)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--tenor", type=str, default="BOTH")
    parser.add_argument("--strategies", type=str, default="short_straddle")
    parser.add_argument("--coverage-mode", type=str, default="ASOF", choices=["ASOF", "STRICT"])
    parser.add_argument("--as-of-date", type=str, default="")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--strategy-overrides-json", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    try:
        input_path = Path(args.path)
        outdir = Path(args.outdir)
        start_date = pd.Timestamp(args.start_date).normalize()

        end_date = pd.Timestamp(args.end_date).normalize() if args.end_date.strip() else None
        as_of_override = pd.Timestamp(args.as_of_date).normalize() if args.as_of_date.strip() else None

        symbol = args.symbol.strip().upper()
        if not symbol:
            raise ValueError("--symbol must be non-empty")

        tenor = args.tenor.strip().upper() if args.tenor.strip() else "BOTH"
        strategies = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
        if not strategies:
            raise ValueError("--strategies must include at least one strategy")

        overrides_raw = _parse_json_dict(args.strategy_overrides_json)
        overrides: Dict[str, Dict[str, Any]] = {}
        for k, v in overrides_raw.items():
            if isinstance(v, dict):
                overrides[str(k).strip().lower()] = v
            else:
                raise ValueError(f"Override for '{k}' must be a dict; got {type(v)}")

        run_phase2(
            input_path=input_path,
            outdir=outdir,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            tenor=tenor,
            strategies=strategies,
            coverage_mode=args.coverage_mode,
            as_of_override=as_of_override,
            strategy_overrides=overrides,
        )

        print(f"PASS: wrote artifacts to {outdir}")
        return 0

    except Exception as e:
        logger.exception("Phase2 failed: %s", str(e))
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
