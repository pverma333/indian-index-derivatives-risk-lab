from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("smoke_test.derivatives_clean")

REQUIRED_COLS: Tuple[str, ...] = (
    "date",
    "symbol",
    "instrument",
    "expiry_dt",
    "strike_pr",
    "option_typ",  # must exist, but can be null/blank for FUTIDX
    "is_opt_weekly_expiry",
    "is_opt_monthly_expiry",
    "cal_days_to_expiry",
    "is_trading_day",
    "settle_pr",
    "spot_close",
    "index_open_price",
    "lot_size",
    "rate_182d",
    "rate_364d",
    "chg_in_oi",
)

RECOMMENDED_COLS: Tuple[str, ...] = ("rate_91d", "vix_close", "contracts", "open_int")

FUT_OPTION_TYP_CANON = "XX"
ALLOWED_SYMBOLS = {"NIFTY", "BANKNIFTY"}
ALLOWED_INSTRUMENTS = {"FUTIDX", "OPTIDX"}
ALLOWED_OPT_TYPES = {"CE", "PE"}
ALLOWED_OPTION_TYP_NORM = {"CE", "PE", "XX"}

KEY_COLS: Tuple[str, ...] = ("date", "symbol", "instrument", "expiry_dt", "strike_pr", "option_typ_norm")

# NOTE: option_typ is nullable in schema; do NOT include in required not-null enforcement
NOT_NULL_REQUIRED_COLS: Tuple[str, ...] = tuple(c for c in REQUIRED_COLS if c != "option_typ")


@dataclass(frozen=True)
class SmokeTestFailure(Exception):
    message: str
    missing_columns: List[str]
    first_violations: pd.DataFrame

    def __str__(self) -> str:
        parts = [self.message]
        if self.missing_columns:
            parts.append(f"Missing columns ({len(self.missing_columns)}): {self.missing_columns}")
        if self.first_violations is not None and not self.first_violations.empty:
            parts.append("First 20 violating rows (subset):")
            parts.append(self.first_violations.to_string(index=False))
        return "\n".join(parts)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SmokeTestFailure(f"CSV file not found: {path}", [], pd.DataFrame())
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise SmokeTestFailure(f"CSV loaded but is empty: {path}", [], pd.DataFrame())
    return df


def _validate_columns_present(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SmokeTestFailure("Missing required columns for Phase 2 market_df contract", missing, pd.DataFrame())

    missing_rec = [c for c in RECOMMENDED_COLS if c not in df.columns]
    if missing_rec:
        logger.warning("Missing recommended columns (non-fatal): %s", missing_rec)


def _coerce_dates_and_option_typ(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["expiry_dt"] = pd.to_datetime(out["expiry_dt"], errors="coerce").dt.normalize()

    # Normalize option_typ into Phase-2 domain {CE, PE, XX}
    # FUTIDX often has blank/null option_typ in curated sources.
    opt = out["option_typ"].astype("string").fillna("").str.strip().str.upper()
    out["option_typ_norm"] = opt.mask(opt == "", FUT_OPTION_TYP_CANON)

    return out


def _coerce_booleans(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    def to_bool_series(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s
        x = s.astype("string").str.strip().str.lower()
        mapped = x.map(
            {
                "1": True,
                "0": False,
                "true": True,
                "false": False,
                "t": True,
                "f": False,
                "yes": True,
                "no": False,
            }
        )
        numeric = pd.to_numeric(s, errors="coerce")
        mapped2 = numeric.map({1.0: True, 0.0: False})
        return mapped.fillna(mapped2)

    for c in cols:
        out[c] = to_bool_series(out[c])

    return out


def _fail_if_any(msg: str, viol_mask: pd.Series, viol_view: pd.DataFrame) -> None:
    if bool(viol_mask.any()):
        raise SmokeTestFailure(msg, [], viol_view.loc[viol_mask].head(20).copy())


def _validate_invariants(df: pd.DataFrame) -> None:
    # Date parse must succeed
    _fail_if_any(
        "date parse failed (NaT found)",
        df["date"].isna(),
        df[["date", "symbol", "instrument", "expiry_dt", "strike_pr", "option_typ", "option_typ_norm"]].copy(),
    )
    _fail_if_any(
        "expiry_dt parse failed (NaT found)",
        df["expiry_dt"].isna(),
        df[["date", "symbol", "instrument", "expiry_dt", "strike_pr", "option_typ", "option_typ_norm"]].copy(),
    )

    # Not-null checks (excluding option_typ)
    for c in NOT_NULL_REQUIRED_COLS:
        _fail_if_any(
            f"Required column has nulls: {c}",
            df[c].isna(),
            df[list(KEY_COLS) + [c]].copy(),
        )

    # Enums
    _fail_if_any(
        f"symbol outside allowed set: {sorted(ALLOWED_SYMBOLS)}",
        ~df["symbol"].astype(str).isin(ALLOWED_SYMBOLS),
        df[list(KEY_COLS) + ["symbol"]].copy(),
    )
    _fail_if_any(
        f"instrument outside allowed set: {sorted(ALLOWED_INSTRUMENTS)}",
        ~df["instrument"].astype(str).isin(ALLOWED_INSTRUMENTS),
        df[list(KEY_COLS) + ["instrument"]].copy(),
    )

    # Boolean flags parseable
    for c in ["is_trading_day", "is_opt_weekly_expiry", "is_opt_monthly_expiry"]:
        _fail_if_any(
            f"Boolean flag not parseable (null after coercion): {c}",
            df[c].isna(),
            df[list(KEY_COLS) + [c]].copy(),
        )

    # cal_days_to_expiry >= 0
    cal = pd.to_numeric(df["cal_days_to_expiry"], errors="coerce")
    _fail_if_any("cal_days_to_expiry not numeric", cal.isna(), df[list(KEY_COLS) + ["cal_days_to_expiry"]].copy())
    _fail_if_any("cal_days_to_expiry < 0 (invalid)", cal < 0, df[list(KEY_COLS) + ["cal_days_to_expiry"]].copy())

    # option_typ_norm domain (Phase 2 wants {CE,PE,XX})
    _fail_if_any(
        f"option_typ_norm outside allowed set: {sorted(ALLOWED_OPTION_TYP_NORM)}",
        ~df["option_typ_norm"].astype(str).isin(ALLOWED_OPTION_TYP_NORM),
        df[list(KEY_COLS) + ["option_typ", "option_typ_norm"]].copy(),
    )

    strike = pd.to_numeric(df["strike_pr"], errors="coerce")
    _fail_if_any("strike_pr not numeric", strike.isna(), df[list(KEY_COLS) + ["strike_pr"]].copy())

    is_fut = df["instrument"].astype(str) == "FUTIDX"
    is_opt = df["instrument"].astype(str) == "OPTIDX"

    # FUTIDX invariants: option_typ_norm == XX, strike_pr == 0.0
    _fail_if_any(
        "FUTIDX rows must have strike_pr == 0.0",
        is_fut & ~(np.isclose(strike, 0.0, equal_nan=False)),
        df[list(KEY_COLS) + ["strike_pr", "instrument", "option_typ", "option_typ_norm"]].copy(),
    )
    _fail_if_any(
        f"FUTIDX rows must have option_typ_norm == {FUT_OPTION_TYP_CANON}",
        is_fut & (df["option_typ_norm"].astype(str) != FUT_OPTION_TYP_CANON),
        df[list(KEY_COLS) + ["instrument", "option_typ", "option_typ_norm"]].copy(),
    )

    # OPTIDX invariants: option_typ_norm in {CE,PE}
    _fail_if_any(
        f"OPTIDX rows must have option_typ in {sorted(ALLOWED_OPT_TYPES)}",
        is_opt & ~df["option_typ_norm"].astype(str).isin(ALLOWED_OPT_TYPES),
        df[list(KEY_COLS) + ["instrument", "option_typ", "option_typ_norm"]].copy(),
    )

    # Key uniqueness on normalized grain
    dup_mask = df.duplicated(subset=list(KEY_COLS), keep=False)
    _fail_if_any(
        f"Key uniqueness violated on {KEY_COLS}",
        dup_mask,
        df[list(KEY_COLS) + ["settle_pr"]].copy(),
    )


def run_smoke_test(csv_path: Path) -> None:
    logger.info("Loading CSV: %s", csv_path)
    df_raw = _read_csv(csv_path)
    logger.info("Loaded: rows=%d cols=%d", len(df_raw), len(df_raw.columns))

    _validate_columns_present(df_raw)

    df = _coerce_dates_and_option_typ(df_raw)
    df = _coerce_booleans(df, ["is_trading_day", "is_opt_weekly_expiry", "is_opt_monthly_expiry"])

    logger.info("Date range: %s → %s", df["date"].min(skipna=True), df["date"].max(skipna=True))
    logger.info("Symbols: %s", sorted(set(df["symbol"].astype(str).unique())))

    _validate_invariants(df)
    logger.info("SMOKE TEST PASSED ")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test for derivatives_clean_Q1_2025.csv")
    parser.add_argument("--csv", type=str, default="data/curated/derivatives_clean_Q1_2025.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    try:
        run_smoke_test(Path(args.csv))
        return 0
    except SmokeTestFailure as e:
        logger.error("SMOKE TEST FAILED \n%s", str(e))
        return 1
    except Exception:
        logger.exception("Unhandled exception during smoke test")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
