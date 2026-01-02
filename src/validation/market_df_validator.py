from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

# Allow running this file directly in VS Code ("Run Python File") without install/PYTHONPATH.
# Repo root is two levels above this file: repo_root/src/validation/market_df_validator.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.exceptions import DataIntegrityError  # noqa: E402

REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "instrument",
    "expiry_dt",
    "strike_pr",
    "option_typ",
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
]

RECOMMENDED_COLUMNS = ["rate_91d", "vix_close", "contracts", "open_int"]

KEY_COLS = ["date", "symbol", "instrument", "expiry_dt", "option_typ", "strike_pr"]

OPTION_TYP_ALLOWED = {"CE", "PE", "XX"}
OPTIDX_ALLOWED = {"CE", "PE"}


@dataclass(frozen=True)
class ValidationWindow:
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    return logger if logger is not None else logging.getLogger(__name__)


def _missing_columns(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    have = set(df.columns)
    return [c for c in cols if c not in have]


def _parse_date(value: Any, field_name: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="raise")
    if isinstance(ts, pd.DatetimeIndex):
        raise ValueError(f"{field_name} must be scalar, got DatetimeIndex")
    return pd.Timestamp(ts).normalize()


def _normalize_datetime_column(df: pd.DataFrame, col: str) -> Tuple[int, int]:
    raw = df[col]
    parsed = pd.to_datetime(raw, errors="coerce")
    parse_failures = int(parsed.isna().sum())

    normalized = parsed.dt.normalize()
    had_time_component = int((parsed.notna() & (parsed != normalized)).sum())

    df[col] = normalized
    return parse_failures, had_time_component


def _build_window(start_date: Any, end_date: Any) -> ValidationWindow:
    start_ts = _parse_date(start_date, "start_date") if start_date is not None else None
    end_ts = _parse_date(end_date, "end_date") if end_date is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(f"start_date ({start_ts.date()}) > end_date ({end_ts.date()})")
    return ValidationWindow(start=start_ts, end=end_ts)


def _apply_window(df: pd.DataFrame, window: ValidationWindow) -> pd.DataFrame:
    if window.start is None and window.end is None:
        return df
    mask = pd.Series(True, index=df.index)
    if window.start is not None:
        mask &= df["date"] >= window.start
    if window.end is not None:
        mask &= df["date"] <= window.end
    return df.loc[mask]


def _preview_violations(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    cols = [c for c in KEY_COLS if c in df.columns]
    return df.loc[mask, cols].head(20).copy()


def _raise_rule(df: pd.DataFrame, rule: str, mask: pd.Series, details: str) -> None:
    count = int(mask.sum())
    preview = _preview_violations(df, mask)
    raise DataIntegrityError(rule=rule, details=f"{details}. violations={count}", violations_preview=preview)


def _coerce_futidx_option_typ(df: pd.DataFrame) -> int:
    fut_mask = df["instrument"] == "FUTIDX"
    opt = df["option_typ"]

    as_str = opt.fillna("").astype(str).str.strip()
    blank_or_null = as_str.eq("") | opt.isna()
    to_coerce = fut_mask & blank_or_null
    coerced = int(to_coerce.sum())
    if coerced > 0:
        df.loc[to_coerce, "option_typ"] = "XX"
    return coerced


def _validate_boolean_flags(df: pd.DataFrame) -> None:
    for col in ["is_opt_weekly_expiry", "is_opt_monthly_expiry"]:
        s = df[col]
        if is_bool_dtype(s):
            invalid = s.isna()
        else:
            invalid = s.isna() | ~s.map(lambda v: isinstance(v, (bool, np.bool_)))
        if invalid.any():
            _raise_rule(
                df,
                rule="FLAG_NOT_BOOLEAN",
                mask=invalid,
                details=f"{col} must be boolean (True/False) with no nulls",
            )


def validate_market_df(
    market_df: pd.DataFrame,
    start_date: Any = None,
    end_date: Any = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Phase 2 gatekeeper for market_df used by all backtests.

    Notes:
      - Mutates `market_df` in-place for `date` and `expiry_dt` normalization and FUTIDX option_typ coercion.
      - Raises DataIntegrityError on fail-fast violations.
    """
    log = _get_logger(logger)

    missing_req = _missing_columns(market_df, REQUIRED_COLUMNS)
    if missing_req:
        raise DataIntegrityError(
            rule="REQUIRED_COLUMNS_MISSING",
            details=f"Missing required columns: {missing_req}",
            violations_preview=None,
        )

    missing_rec = _missing_columns(market_df, RECOMMENDED_COLUMNS)
    if missing_rec:
        log.warning("market_df missing recommended columns", extra={"missing": missing_rec})

    report: Dict[str, Any] = {
        "rows_total": int(len(market_df)),
        "rows_in_scope": None,
        "window_start": None,
        "window_end": None,
        "missing_recommended_columns": missing_rec,
        "date_parse_failures": 0,
        "date_had_time_component": 0,
        "expiry_dt_parse_failures": 0,
        "futidx_option_typ_coerced": 0,
    }

    date_fail, date_timecnt = _normalize_datetime_column(market_df, "date")
    report["date_parse_failures"] = date_fail
    report["date_had_time_component"] = date_timecnt
    if date_fail > 0:
        bad_preview = market_df.loc[market_df["date"].isna(), [c for c in KEY_COLS if c in market_df.columns]].head(20)
        raise DataIntegrityError(
            rule="DATE_PARSE_FAILED",
            details=f"Failed to parse {date_fail} 'date' values",
            violations_preview=bad_preview,
        )

    exp_fail, _ = _normalize_datetime_column(market_df, "expiry_dt")
    report["expiry_dt_parse_failures"] = exp_fail
    if exp_fail > 0:
        bad_preview = market_df.loc[
            market_df["expiry_dt"].isna(), [c for c in KEY_COLS if c in market_df.columns]
        ].head(20)
        raise DataIntegrityError(
            rule="EXPIRY_DT_PARSE_FAILED",
            details=f"Failed to parse {exp_fail} 'expiry_dt' values",
            violations_preview=bad_preview,
        )

    window = _build_window(start_date, end_date)
    report["window_start"] = window.start.date().isoformat() if window.start is not None else None
    report["window_end"] = window.end.date().isoformat() if window.end is not None else None

    df_scope = _apply_window(market_df, window)
    report["rows_in_scope"] = int(len(df_scope))

    report["futidx_option_typ_coerced"] = _coerce_futidx_option_typ(market_df)

    invalid_domain = ~market_df["option_typ"].isin(list(OPTION_TYP_ALLOWED))
    if invalid_domain.any():
        _raise_rule(
            market_df,
            rule="OPTION_TYP_DOMAIN",
            mask=invalid_domain,
            details=f"option_typ must be one of {sorted(OPTION_TYP_ALLOWED)}",
        )

    strike = pd.to_numeric(market_df["strike_pr"], errors="coerce")
    if strike.isna().any():
        _raise_rule(market_df, rule="STRIKE_PR_NOT_NUMERIC", mask=strike.isna(), details="strike_pr must be numeric")

    fut_mask = market_df["instrument"] == "FUTIDX"
    fut_viol = fut_mask & ((market_df["option_typ"] != "XX") | ~np.isclose(strike, 0.0, atol=1e-12))
    if fut_viol.any():
        _raise_rule(
            market_df,
            rule="FUTIDX_INVARIANTS",
            mask=fut_viol,
            details="FUTIDX requires option_typ == 'XX' and strike_pr == 0.0",
        )

    opt_mask = market_df["instrument"] == "OPTIDX"
    opt_viol = opt_mask & (~market_df["option_typ"].isin(list(OPTIDX_ALLOWED)) | ~(strike > 0))
    if opt_viol.any():
        _raise_rule(
            market_df,
            rule="OPTIDX_INVARIANTS",
            mask=opt_viol,
            details="OPTIDX requires option_typ in {'CE','PE'} and strike_pr > 0",
        )

    null_iop = df_scope["index_open_price"].isna()
    if null_iop.any():
        _raise_rule(
            df_scope,
            rule="INDEX_OPEN_PRICE_NULL_IN_SCOPE",
            mask=null_iop,
            details="index_open_price must be non-null in filtered backtest window",
        )

    _validate_boolean_flags(market_df)

    log.info("market_df validation passed", extra={"report": report})


def _read_market_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file type: {p.suffix} (expected .csv or .parquet)")


def _configure_console_logging() -> logging.Logger:
    logger = logging.getLogger("market_df_validator")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate market_df contract for derivatives_clean data.")
    ap.add_argument(
        "--path",
        default="data/curated/derivatives_clean_Q1_2025.csv",
        help="Path to .csv or .parquet (default: data/curated/derivatives_clean_Q1_2025.csv)",
    )
    ap.add_argument("--start-date", default=None, help="Optional start date (YYYY-MM-DD)")
    ap.add_argument("--end-date", default=None, help="Optional end date (YYYY-MM-DD)")
    args = ap.parse_args()

    log = _configure_console_logging()
    df = _read_market_file(args.path)
    validate_market_df(df, start_date=args.start_date, end_date=args.end_date, logger=log)
