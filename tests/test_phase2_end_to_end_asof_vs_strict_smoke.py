from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


LEG_KEY_COLS = ["strategy_name", "tenor", "trade_id", "leg_id"]


def _find_dataset_path() -> Path:
    candidates = [
        _REPO_ROOT / "data" / "curated" / "derivatives_clean_Q1_2025.csv",
        _REPO_ROOT / "data" / "curated" / "derivatives_clean_Q1_2025.parquet",
        _REPO_ROOT / "data" / "curated" / "derivatives_clean_Q1_2025.snappy.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Curated dataset not found. Expected one of: "
        + ", ".join(str(x) for x in candidates)
    )


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def _normalize_dates(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="raise").dt.normalize()


def _pick_symbol(df: pd.DataFrame) -> str:
    if "symbol" not in df.columns:
        raise AssertionError("market_df missing required column: symbol")
    syms = (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.upper()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    for preferred in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
        if preferred in syms:
            return preferred
    if not syms:
        raise AssertionError("No symbols found in dataset.")
    return syms[0]


def _import_runner_callable() -> Tuple[Optional[Callable[..., Any]], Optional[str]]:
    for mod_name in ["src.run_phase2_backtests", "src.run_phase2_backtest"]:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        for fn_name in ["run_phase2", "run", "run_backtest", "run_phase2_backtest"]:
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn, mod_name

        return None, mod_name

    return None, None


def _call_runner(
    *,
    runner_fn: Optional[Callable[..., Any]],
    runner_mod_for_cli: Optional[str],
    input_path: Path,
    outdir: Path,
    symbol: str,
    tenor: str,
    start_date: str,
    end_date: str,
    coverage_mode: str,
    strategies: Sequence[str],
    strategy_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if runner_fn is not None:
        sig = inspect.signature(runner_fn)
        kwargs: Dict[str, Any] = {}

        mapping: Dict[str, Any] = {
            "input_path": input_path,
            "path": str(input_path),
            "outdir": outdir,
            "start_date": pd.Timestamp(start_date),
            "end_date": pd.Timestamp(end_date),
            "symbol": symbol,
            "tenor": tenor,
            "strategies": list(strategies),
            "coverage_mode": coverage_mode,
            "as_of_override": None,
            "as_of_date": None,
            "strategy_overrides": strategy_overrides or {},
        }

        for k, v in mapping.items():
            if k in sig.parameters:
                kwargs[k] = v

        try:
            runner_fn(**kwargs)
            return
        except TypeError as e:
            raise AssertionError(
                f"Runner callable signature mismatch. Callable={runner_fn} "
                f"kwargs={sorted(kwargs.keys())}. Underlying error: {e}"
            ) from e

    if not runner_mod_for_cli:
        raise AssertionError("No runner callable found and no CLI module available.")

    cmd = [
        sys.executable,
        "-m",
        runner_mod_for_cli,
        "--path",
        str(input_path),
        "--symbol",
        symbol,
        "--tenor",
        tenor,
        "--strategies",
        ",".join(strategies),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--coverage-mode",
        coverage_mode,
        "--outdir",
        str(outdir),
    ]

    if strategy_overrides:
        cmd += ["--strategy-overrides-json", json.dumps(strategy_overrides)]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise AssertionError(
            "Runner CLI failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )


def _require_artifact(outdir: Path, filename: str) -> Path:
    p = outdir / filename
    assert p.exists(), f"Required artifact missing: {p}"
    return p


def _read_artifacts(outdir: Path) -> Dict[str, Any]:
    paths = {
        "legs": _require_artifact(outdir, "legs_pnl_df.parquet"),
        "skips": _require_artifact(outdir, "skips_df.parquet"),
        "trade": _require_artifact(outdir, "trade_pnl_df.parquet"),
        "strategy": _require_artifact(outdir, "strategy_pnl_df.parquet"),
        "positions": _require_artifact(outdir, "positions_df.parquet"),
        "manifest": _require_artifact(outdir, "run_manifest.json"),
    }

    legs = pd.read_parquet(paths["legs"])
    skips = pd.read_parquet(paths["skips"])
    trade = pd.read_parquet(paths["trade"])
    strategy = pd.read_parquet(paths["strategy"])
    positions = pd.read_parquet(paths["positions"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    return {
        "paths": paths,
        "legs": legs,
        "skips": skips,
        "trade": trade,
        "strategy": strategy,
        "positions": positions,
        "manifest": manifest,
    }


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _get_trade_total_col(df: pd.DataFrame) -> str:
    for c in ["total_mtm_pnl", "mtm_pnl", "pnl", "total_pnl"]:
        if c in df.columns:
            return c
    raise AssertionError(f"Could not find a total PnL column in df columns={list(df.columns)}")


def _manifest_block(manifest: Dict[str, Any], strat_name: str, tenor: str) -> Dict[str, Any]:
    if isinstance(manifest.get("strategies"), dict):
        s = manifest["strategies"]
        if strat_name in s and isinstance(s[strat_name], dict) and tenor in s[strat_name]:
            block = s[strat_name][tenor]
            if isinstance(block, dict):
                return block
    return manifest


def _manifest_expected_counts(
    *,
    legs_pnl_df: pd.DataFrame,
    skips_df: pd.DataFrame,
    strat_name: str,
    tenor: str,
) -> Dict[str, Any]:
    legs_leglevel = (
        legs_pnl_df[LEG_KEY_COLS + ["status"]].drop_duplicates()
        if (legs_pnl_df is not None and not legs_pnl_df.empty)
        else pd.DataFrame(columns=LEG_KEY_COLS + ["status"])
    )
    legs_leglevel = legs_leglevel[
        (legs_leglevel["strategy_name"].astype(str) == str(strat_name))
        & (legs_leglevel["tenor"].astype(str) == str(tenor))
    ]
    n_open = int((legs_leglevel["status"].astype(str) == "OPEN").sum()) if not legs_leglevel.empty else 0
    n_closed = int((legs_leglevel["status"].astype(str) == "CLOSED").sum()) if not legs_leglevel.empty else 0

    if skips_df is not None and not skips_df.empty:
        need = ["strategy_name", "tenor", "trade_id", "leg_id"]
        missing = [c for c in need if c not in skips_df.columns]
        assert not missing, f"skips_df missing required columns: {missing}"
        ssub = skips_df[
            (skips_df["strategy_name"].astype(str) == str(strat_name))
            & (skips_df["tenor"].astype(str) == str(tenor))
        ].copy()
        n_skipped = int(len(ssub[need].drop_duplicates()))
        if "reason" in ssub.columns:
            skips_by_reason = ssub.groupby("reason").size().sort_index().to_dict()
        else:
            skips_by_reason = {}
    else:
        n_skipped = 0
        skips_by_reason = {}

    return {
        "n_open_legs": n_open,
        "n_closed_legs": n_closed,
        "n_skipped_legs": n_skipped,
        "skips_by_reason": skips_by_reason,
    }


def _find_asof_run_with_open_legs(
    *,
    tmp_path: Path,
    dataset_path: Path,
    market_df: pd.DataFrame,
    runner_fn: Optional[Callable[..., Any]],
    runner_mod: Optional[str],
    symbol: str,
    strategies: Sequence[str],
) -> Tuple[str, str, str, Dict[str, Any]]:
    df = market_df.copy()
    _normalize_dates(df, ["date", "expiry_dt"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    sym_df = df[df["symbol"] == symbol].copy()
    assert not sym_df.empty, f"Symbol {symbol!r} not present in dataset."

    trade_dates = sym_df["date"].dropna().drop_duplicates().sort_values().to_list()
    assert len(trade_dates) >= 30, "Not enough trading dates in dataset."

    tenors = ["WEEKLY", "MONTHLY"]
    start_buffers = [90, 60, 40]
    end_offsets = [5, 7, 10, 3]

    tried: list[tuple[str, str, str]] = []

    start_indices = list(range(20, max(21, len(trade_dates) - 15), 7))

    for tenor in tenors:
        for start_i in start_indices:
            for buf in start_buffers:
                start_idx = max(0, start_i - buf)
                start_dt = pd.Timestamp(trade_dates[start_idx]).normalize()

                for off in end_offsets:
                    end_idx = start_i + off
                    if end_idx >= len(trade_dates):
                        continue
                    end_dt = pd.Timestamp(trade_dates[end_idx]).normalize()

                    start_date = start_dt.date().isoformat()
                    end_date = end_dt.date().isoformat()

                    tried.append((tenor, start_date, end_date))

                    outdir = tmp_path / f"probe_asof_{tenor}_{start_date}_{end_date}".replace("-", "")
                    try:
                        _call_runner(
                            runner_fn=runner_fn,
                            runner_mod_for_cli=runner_mod,
                            input_path=dataset_path,
                            outdir=outdir,
                            symbol=symbol,
                            tenor=tenor,
                            start_date=start_date,
                            end_date=end_date,
                            coverage_mode="ASOF",
                            strategies=strategies,
                            strategy_overrides={},
                        )
                    except Exception:
                        continue

                    try:
                        run = _read_artifacts(outdir)
                    except Exception:
                        continue

                    legs = run["legs"]
                    if legs is None or legs.empty:
                        continue
                    if "status" not in legs.columns:
                        continue
                    if (legs["status"].astype(str) == "OPEN").any():
                        return tenor, start_date, end_date, run

    tried_preview = "\n".join([f"  - {t} {s}..{e}" for (t, s, e) in tried[:20]])
    raise AssertionError(
        "Could not find an ASOF run that emits at least one OPEN leg.\n"
        "This usually means either:\n"
        "  - the curated dataset window cannot produce trades for the tested strategy/tenor, or\n"
        "  - ASOF open-leg behavior is not reachable with the current runner/strategy.\n"
        "Tried first 20 windows:\n"
        f"{tried_preview}"
    )


@pytest.mark.integration
def test_phase2_end_to_end_asof_vs_strict_smoke(tmp_path: Path) -> None:
    dataset_path = _find_dataset_path()
    market_df = _read_df(dataset_path)
    _normalize_dates(market_df, ["date", "expiry_dt"])

    symbol = _pick_symbol(market_df)
    strategies = ["short_straddle"]

    runner_fn, runner_mod = _import_runner_callable()
    if runner_fn is None and runner_mod is None:
        pytest.skip("Phase 2 runner module not found (expected src.run_phase2_backtest(s)).")

    tenor, start_date, end_date, asof = _find_asof_run_with_open_legs(
        tmp_path=tmp_path,
        dataset_path=dataset_path,
        market_df=market_df,
        runner_fn=runner_fn,
        runner_mod=runner_mod,
        symbol=symbol,
        strategies=strategies,
    )

    legs_asof = asof["legs"]
    skips_asof = asof["skips"]
    pos_asof = asof["positions"]

    assert isinstance(skips_asof, pd.DataFrame), "skips_df must be a DataFrame (can be empty)."
    assert legs_asof is not None and not legs_asof.empty, "ASOF run must emit non-empty legs_pnl_df."

    required_cols = {
        "as_of_date_used",
        "end_date_used",
        "status",
        "is_open",
        "market_max_date",
        "coverage_mode",
        "date",
        "entry_date",
        "exit_date",
        "mtm_pnl",
        "units",
        "settle_used",
        "settle_prev_used",
        "entry_price",
    }
    missing = sorted(required_cols - set(legs_asof.columns))
    assert not missing, f"legs_pnl_df missing required columns: {missing}"

    status_nuniq = legs_asof.groupby(LEG_KEY_COLS)["status"].nunique(dropna=False).reset_index(name="nuniq")
    bad = status_nuniq[status_nuniq["nuniq"] > 1]
    assert bad.empty, "status must be constant within a leg (per strategy_name+tenor+trade_id+leg_id)."

    open_rows = legs_asof[legs_asof["status"].astype(str) == "OPEN"].copy()
    assert not open_rows.empty, "ASOF run must include at least one OPEN leg (search ensured this)."

    open_leg_maxdate = (
        open_rows.groupby(LEG_KEY_COLS)
        .agg(
            max_date=("date", "max"),
            end_date_used=("end_date_used", "first"),
            as_of_date_used=("as_of_date_used", "first"),
        )
        .reset_index()
    )
    open_leg_maxdate["max_date"] = pd.to_datetime(open_leg_maxdate["max_date"]).dt.normalize()
    open_leg_maxdate["end_date_used"] = pd.to_datetime(open_leg_maxdate["end_date_used"]).dt.normalize()
    open_leg_maxdate["as_of_date_used"] = pd.to_datetime(open_leg_maxdate["as_of_date_used"]).dt.normalize()

    mism1 = open_leg_maxdate[open_leg_maxdate["max_date"] != open_leg_maxdate["end_date_used"]]
    assert mism1.empty, (
        "For OPEN legs, max(date) must equal end_date_used. "
        f"First mismatch:\n{mism1.head(5).to_string(index=False)}"
    )

    mism2 = open_leg_maxdate[open_leg_maxdate["end_date_used"] != open_leg_maxdate["as_of_date_used"]]
    assert mism2.empty, (
        "For OPEN legs, end_date_used must equal as_of_date_used. "
        f"First mismatch:\n{mism2.head(5).to_string(index=False)}"
    )

    assert not pos_asof.empty, "positions_df must be written and non-empty for this run."
    for c in ["status", "realized_pnl", "unrealized_pnl", "cum_pnl_asof"]:
        assert c in pos_asof.columns, f"positions_df missing required column: {c}"

    pos_open = pos_asof[pos_asof["status"].astype(str) == "OPEN"].copy()
    assert not pos_open.empty, "positions_df must include OPEN positions when ASOF emits OPEN legs."

    realized_open = _to_num(pos_open["realized_pnl"])
    assert float(np.abs(realized_open).max()) <= 1e-6, "OPEN legs must have realized_pnl == 0."

    unreal_open = _to_num(pos_open["unrealized_pnl"])
    cum_asof_open = _to_num(pos_open["cum_pnl_asof"])
    assert np.allclose(unreal_open.to_numpy(), cum_asof_open.to_numpy(), atol=1e-6), (
        "For OPEN legs, unrealized_pnl must equal cum_pnl_asof."
    )

    realized_total = float(_to_num(pos_asof["realized_pnl"]).sum())
    unreal_total = float(_to_num(pos_asof["unrealized_pnl"]).sum())
    cum_total = float(_to_num(pos_asof["cum_pnl_asof"]).sum())
    assert abs((realized_total + unreal_total) - cum_total) <= 1e-6, (
        "Portfolio reconcile failed: realized_total + unrealized_total must equal total cum_pnl_asof."
    )

    out_strict = tmp_path / "phase2_strict"
    _call_runner(
        runner_fn=runner_fn,
        runner_mod_for_cli=runner_mod,
        input_path=dataset_path,
        outdir=out_strict,
        symbol=symbol,
        tenor=tenor,
        start_date=start_date,
        end_date=end_date,
        coverage_mode="STRICT",
        strategies=strategies,
        strategy_overrides={},
    )
    strict = _read_artifacts(out_strict)

    legs_strict = strict["legs"]
    skips_strict = strict["skips"]

    open_leg_keys = open_rows[LEG_KEY_COLS].drop_duplicates().astype(str).to_records(index=False).tolist()
    strict_leg_keys = (
        legs_strict[LEG_KEY_COLS].drop_duplicates().astype(str).to_records(index=False).tolist()
        if (legs_strict is not None and not legs_strict.empty)
        else []
    )
    overlap = set(open_leg_keys) & set(strict_leg_keys)
    assert not overlap, (
        "STRICT run must not emit legs that were OPEN in ASOF for the same window. "
        f"Overlapping leg keys (up to 10): {list(overlap)[:10]}"
    )

    assert "reason" in skips_strict.columns, "STRICT skips_df must include 'reason' column."
    strict_reason = "MARKET_WINDOW_END_BEFORE_EXIT_STRICT"
    n_strict_reason = int((skips_strict["reason"].astype(str) == strict_reason).sum())
    assert n_strict_reason > 0, (
        f"STRICT skips_df must include reason {strict_reason!r} with count > 0."
    )

    trade_asof = asof["trade"]
    strat_asof = asof["strategy"]

    legs_sum = float(_to_num(legs_asof["mtm_pnl"]).sum())
    trade_col = _get_trade_total_col(trade_asof)
    strat_col = _get_trade_total_col(strat_asof)

    trade_sum = float(_to_num(trade_asof[trade_col]).sum())
    strat_sum = float(_to_num(strat_asof[strat_col]).sum())

    assert abs(legs_sum - trade_sum) <= 1e-6, (
        f"Reconciliation failed: sum(legs.mtm_pnl)={legs_sum} vs sum(trade.{trade_col})={trade_sum}"
    )
    assert abs(trade_sum - strat_sum) <= 1e-6, (
        f"Reconciliation failed: sum(trade.{trade_col})={trade_sum} vs sum(strategy.{strat_col})={strat_sum}"
    )

    legs_sorted = legs_asof.sort_values(LEG_KEY_COLS + ["date"], kind="mergesort").copy()
    first_rows = legs_sorted.groupby(LEG_KEY_COLS, as_index=False).head(1).copy()
    sample = first_rows.iloc[0]

    sp = float(pd.to_numeric(pd.Series([sample["settle_prev_used"]]), errors="coerce").iloc[0])
    ep = float(pd.to_numeric(pd.Series([sample["entry_price"]]), errors="coerce").iloc[0])
    assert abs(sp - ep) <= 1e-6, (
        f"Day-0 anchor failed: settle_prev_used={sp} must equal entry_price={ep} "
        f"for leg {tuple(sample[c] for c in LEG_KEY_COLS)}"
    )

    units = float(pd.to_numeric(pd.Series([sample["units"]]), errors="coerce").iloc[0])
    settle_used = float(pd.to_numeric(pd.Series([sample["settle_used"]]), errors="coerce").iloc[0])
    mtm = float(pd.to_numeric(pd.Series([sample["mtm_pnl"]]), errors="coerce").iloc[0])
    expected_mtm = units * (settle_used - ep)
    assert abs(mtm - expected_mtm) <= 1e-6, (
        "Day-0 MTM formula failed: "
        f"mtm_pnl={mtm} vs units*(settle_used-entry_price)={expected_mtm}"
    )

    def _assert_manifest_matches_artifacts(run_obj: Dict[str, Any]) -> None:
        legs = run_obj["legs"]
        skips = run_obj["skips"]
        manifest = run_obj["manifest"]

        strat_name = strategies[0]
        block = _manifest_block(manifest, strat_name=strat_name, tenor=tenor)

        expected = _manifest_expected_counts(
            legs_pnl_df=legs,
            skips_df=skips,
            strat_name=strat_name,
            tenor=tenor,
        )

        for k in ["n_open_legs", "n_closed_legs", "n_skipped_legs", "skips_by_reason"]:
            assert k in block, f"run_manifest.json missing required key {k!r} in manifest block."

        assert int(block.get("n_open_legs")) == expected["n_open_legs"], "Manifest n_open_legs must match artifacts."
        assert int(block.get("n_closed_legs")) == expected["n_closed_legs"], "Manifest n_closed_legs must match artifacts."
        assert int(block.get("n_skipped_legs")) == expected["n_skipped_legs"], "Manifest n_skipped_legs must match artifacts."
        assert dict(block.get("skips_by_reason")) == expected["skips_by_reason"], "Manifest skips_by_reason must match artifacts."

    _assert_manifest_matches_artifacts(asof)
    _assert_manifest_matches_artifacts(strict)
