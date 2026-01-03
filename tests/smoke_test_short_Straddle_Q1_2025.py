import sys
from pathlib import Path
import argparse
import logging
from typing import Iterable, Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.short_straddle import ShortStraddleStrategy
from src.strategies import expiry_selectors as es
from src.strategies import contract_selectors as cs


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, default="data/curated/derivatives_clean_Q1_2025.csv")
    p.add_argument("--symbol", type=str, default="NIFTY")
    p.add_argument("--tenor", type=str, default="BOTH", choices=["WEEKLY", "MONTHLY", "BOTH"])
    p.add_argument("--full-print", action="store_true")
    p.add_argument("--debug-cycles", type=int, default=3, help="Print chain stats for first N cycles per tenor")
    p.add_argument("--save-out", type=str, default="outputs/short_straddle_trades_Q1_2025.csv")
    return p.parse_args()


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    # Handle "True"/"False", "1"/"0", 1/0, True/False
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "nan": pd.NA, "none": pd.NA, "": pd.NA})
        .astype("boolean")
    )


def _coerce_market_df_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dates that are required by schema / selectors
    for col in ["date", "expiry_dt", "timestamp", "opt_weekly_expiry", "opt_monthly_expiry", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()

    # Critical boolean flags for expiry cycle builder
    for col in ["is_opt_weekly_expiry", "is_opt_monthly_expiry", "is_trading_day", "is_trade_calendar_date"]:
        if col in df.columns:
            df[col] = _coerce_bool_series(df[col])

    return df


def _base_cfg(symbol: str, tenor: str) -> dict:
    strike_band_n = 10 if tenor == "WEEKLY" else 15
    return {
        "symbol": symbol,
        "tenor": tenor,
        "qty_lots": 1,
        "strike_band_n": strike_band_n,
        "max_atm_search_steps": 3,
        "liquidity_mode": "OFF",
        "min_contracts": 1,
        "min_open_int": 1,
        "liquidity_percentile": 50,
        "exit_rule": "EXPIRY",
        "exit_k_days": None,
        "fees_bps": 0.0,
        "fixed_fee_per_lot": 0.0,
        # required-but-nullable selection params
        "width_points": None,
        "otm_distance_points": None,
    }


def _tenors(arg: str) -> list[str]:
    return ["WEEKLY", "MONTHLY"] if arg == "BOTH" else [arg]


def _print_df(df: pd.DataFrame, title: str, full: bool) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("(empty)")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    if full:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)
        print(df.to_string(index=False))
    else:
        print(df.head(20).to_string(index=False))
        print("\n--- tail(20) ---")
        print(df.tail(20).to_string(index=False))


def _fingerprint(df: pd.DataFrame) -> None:
    print("\n=== market_df fingerprint ===")
    print(f"rows={len(df)} cols={len(df.columns)}")
    for col in ["date", "expiry_dt"]:
        if col in df.columns:
            print(f"{col}: min={df[col].min()} max={df[col].max()} nulls={df[col].isna().sum()}")
        else:
            print(f"{col}: MISSING")

    for col in ["symbol", "instrument", "option_typ"]:
        if col in df.columns:
            print(f"{col} uniques: {sorted([x for x in df[col].dropna().unique()])[:20]}")

    if "instrument" in df.columns:
        print("instrument counts:")
        print(df["instrument"].value_counts(dropna=False).to_string())

    for col in ["is_opt_weekly_expiry", "is_opt_monthly_expiry", "is_trading_day"]:
        if col in df.columns:
            print(f"{col} dtype={df[col].dtype} counts:")
            print(df[col].value_counts(dropna=False).to_string())


def main() -> None:
    _setup_logging()
    log = logging.getLogger("smoke_short_straddle_q1_2025")

    args = _parse_args()
    csv_path = _PROJECT_ROOT / args.path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log.info("Loading %s", csv_path)
    raw = pd.read_csv(csv_path)
    market_df = _coerce_market_df_types(raw)

    _fingerprint(market_df)

    strategy = ShortStraddleStrategy()
    out_frames: list[pd.DataFrame] = []

    for tenor in _tenors(args.tenor):
        cfg = _base_cfg(args.symbol, tenor)

        print(f"\n==================== RUN tenor={tenor} symbol={args.symbol} ====================")

        # 1) cycles diagnostics
        cycles_df = es.build_expiry_cycles(market_df=market_df, symbol=args.symbol, tenor=tenor)
        _print_df(cycles_df, f"cycles_df ({tenor})", full=False)
        print(f"cycles_df rows={len(cycles_df)}")

        if cycles_df.empty:
            print(
                "No cycles produced. This usually means your is_opt_weekly_expiry / is_opt_monthly_expiry "
                "flags are not True anywhere for this symbol in this CSV slice."
            )
            continue

        # 2) chain diagnostics for first N cycles
        n_dbg = max(int(args.debug_cycles), 0)
        if n_dbg > 0:
            sample = cycles_df.sort_values(["expiry_dt", "entry_date"]).head(n_dbg)
            for i, r in sample.iterrows():
                expiry_dt = pd.Timestamp(r["expiry_dt"]).normalize()
                entry_date = pd.Timestamp(r["entry_date"]).normalize()
                chain_df = cs.get_chain(market_df=market_df, symbol=args.symbol, expiry_dt=expiry_dt, entry_date=entry_date)
                print(
                    f"cycle dbg: expiry_dt={expiry_dt.date()} entry_date={entry_date.date()} "
                    f"chain_rows={len(chain_df)}"
                )
                if not chain_df.empty:
                    cols = [c for c in ["spot_close", "strike_pr", "option_typ", "contracts", "open_int", "settle_pr"] if c in chain_df.columns]
                    print("chain cols subset:", cols)
                    if "spot_close" in chain_df.columns:
                        print("spot_close sample:", chain_df["spot_close"].dropna().head(1).tolist())
                    if "option_typ" in chain_df.columns:
                        print("option_typ counts:", chain_df["option_typ"].value_counts(dropna=False).to_dict())
                    if "strike_pr" in chain_df.columns:
                        uniq = sorted(chain_df["strike_pr"].dropna().unique().tolist())
                        print("unique strikes (first 10):", uniq[:10])

        # 3) run strategy
        trades_df = strategy.build_trades(market_df=market_df, cfg=cfg)
        print(f"trades_df rows={len(trades_df)}")
        if not trades_df.empty:
            # invariants
            per_trade = trades_df.groupby("trade_id").size()
            bad = per_trade[per_trade != 2]
            if len(bad) > 0:
                print("WARNING: trade_ids without exactly 2 legs:", bad.to_dict())

        _print_df(trades_df, f"trades_df ({tenor})", full=args.full_print)
        out_frames.append(trades_df)

    combined = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    out_path = _PROJECT_ROOT / args.save_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nSaved combined output rows={len(combined)} to: {out_path}")

    if combined.empty:
        print(
            "\nCombined output is EMPTY.\n"
            "Use the diagnostics above:\n"
            "- If cycles_df is empty: your expiry flags are not True for this symbol/tenor in this CSV slice.\n"
            "- If cycles_df has rows but chain_rows=0 for all: date/expiry_dt dtype mismatch or get_chain filter mismatch.\n"
        )


if __name__ == "__main__":
    main()
