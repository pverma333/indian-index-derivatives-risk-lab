from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("build_derivatives_curated")


@dataclass(frozen=True)
class ETLConfig:
    derivatives_csv: Path
    market_data_parquet: Path
    lot_size_map_parquet: Path
    treasury_curve_parquet: Path
    trade_calendar_parquet: Path
    output_parquet: Path

    chunksize: int = 750_000

    allowed_instruments: Tuple[str, ...] = ("FUTIDX", "OPTIDX")
    allowed_symbols: Tuple[str, ...] = ("NIFTY", "BANKNIFTY")

    # Fail-fast thresholds
    max_date_parse_failure_rate: float = 0.001  # 0.1%
    max_numeric_parse_failure_rate: float = 0.01  # 1%
    max_drop_negative_tte_rate: float = 0.002  # 0.2%


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def to_snake_case(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"__+", "_", name)
    return name.lower().strip("_")


def normalize_symbol(value: str) -> str:
    if value is None:
        return value
    s = str(value).strip().upper()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ").strip()

    if s in {"NIFTY BANK", "BANK NIFTY", "NIFTYBANK"}:
        return "BANKNIFTY"
    if s in {"NIFTY 50", "NIFTY"}:
        return "NIFTY"
    if s == "BANKNIFTY":
        return "BANKNIFTY"

    return s.replace(" ", "")


def parse_mixed_date_series(values: pd.Series, col_name: str) -> pd.Series:
    # No dayfirst=True (per ticket); allow mixed formats.
    parsed = pd.to_datetime(values, errors="coerce", dayfirst=False).dt.normalize()
    failures = int(parsed.isna().sum())
    if failures:
        LOGGER.warning("Date parse failures for %s: %d of %d", col_name, failures, len(values))
    return parsed


def coerce_float(values: pd.Series, col_name: str) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce").astype("float64")
    failures = int(out.isna().sum() - values.isna().sum())
    if failures:
        LOGGER.warning("Numeric parse failures for %s: %d of %d", col_name, failures, len(values))
    return out


def read_and_filter_derivatives(cfg: ETLConfig) -> pd.DataFrame:
    required_cols = [
        "INSTRUMENT",
        "SYMBOL",
        "EXPIRY_DT",
        "STRIKE_PR",
        "OPTION_TYP",
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "SETTLE_PR",
        "CONTRACTS",
        "OPEN_INT",
        "CHG_IN_OI",
        "TIMESTAMP",
    ]

    LOGGER.info("Reading derivatives CSV: %s", cfg.derivatives_csv)
    frames: List[pd.DataFrame] = []
    total_rows = 0

    for chunk in pd.read_csv(
        cfg.derivatives_csv,
        usecols=required_cols,
        chunksize=cfg.chunksize,
        dtype="string",
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk = chunk[chunk["INSTRUMENT"].isin(cfg.allowed_instruments)]
        chunk = chunk[chunk["SYMBOL"].isin(cfg.allowed_symbols)]
        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        raise ValueError("No rows matched instrument/symbol filters in derivatives CSV.")

    df = pd.concat(frames, ignore_index=True)
    LOGGER.info("Derivatives rows: read=%d filtered=%d", total_rows, len(df))

    df.columns = [to_snake_case(c) for c in df.columns]
    df["instrument"] = df["instrument"].astype("string")
    df["symbol"] = df["symbol"].map(normalize_symbol).astype("string")

    df["timestamp"] = parse_mixed_date_series(df["timestamp"], "timestamp")
    df["expiry_dt"] = parse_mixed_date_series(df["expiry_dt"], "expiry_dt")
    df["date"] = df["timestamp"]

    for col in ("timestamp", "expiry_dt"):
        failure_rate = float(df[col].isna().mean())
        if failure_rate > cfg.max_date_parse_failure_rate:
            raise ValueError(f"Too many date parse failures in {col}: {failure_rate:.4%}")

    numeric_cols = [
        "strike_pr",
        "open",
        "high",
        "low",
        "close",
        "settle_pr",
        "contracts",
        "open_int",
        "chg_in_oi",
    ]
    for c in numeric_cols:
        df[c] = coerce_float(df[c], c)

    numeric_failure_rate = float(df[numeric_cols].isna().sum().sum()) / float(len(df) * len(numeric_cols))
    if numeric_failure_rate > cfg.max_numeric_parse_failure_rate:
        raise ValueError(f"Too many numeric parse failures: {numeric_failure_rate:.4%}")

    # Futures consistency
    fut_mask = df["instrument"] == "FUTIDX"
    df.loc[fut_mask, "strike_pr"] = 0.0

    # Option type normalization
    df["option_typ"] = df["option_typ"].fillna("").astype("string").str.upper().str.strip()

    return df


def load_market_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("date", "spot_close", "vix_close"):
        if c not in df.columns:
            raise ValueError(f"market_data.parquet missing '{c}'")

    df = df.copy()
    df["date"] = parse_mixed_date_series(df["date"].astype("string"), "market_data.date")

    if "symbol" not in df.columns:
        if "index_name" not in df.columns:
            raise ValueError("market_data.parquet missing both 'symbol' and 'index_name'")
        df["symbol"] = df["index_name"].map(normalize_symbol)

    df["symbol"] = df["symbol"].map(normalize_symbol).astype("string")
    df["spot_close"] = df["spot_close"].astype("float64")
    df["vix_close"] = df["vix_close"].astype("float64")
    if "div_yield" in df.columns:
        df["div_yield"] = df["div_yield"].astype("float64")
    else:
        df["div_yield"] = np.nan

    return df[["date", "symbol", "spot_close", "vix_close", "div_yield"]].drop_duplicates()


def validate_no_overlap_lot_map(lot: pd.DataFrame) -> None:
    lot = lot.sort_values(["symbol", "start_date"]).reset_index(drop=True)
    for sym, g in lot.groupby("symbol", sort=False):
        g = g.copy()
        g["end_eff"] = g["end_date"].fillna(pd.Timestamp.max.normalize())
        prev_end = g["end_eff"].shift(1)
        curr_start = g["start_date"]
        overlap = prev_end.notna() & (curr_start <= prev_end)
        if overlap.any():
            bad = g.loc[overlap, ["symbol", "start_date", "end_date"]]
            raise ValueError(f"Overlapping lot_size_map ranges for symbol={sym}: {bad.to_dict('records')}")


def load_lot_size_map(path: Path) -> pd.DataFrame:
    lot = pd.read_parquet(path).copy()
    for c in ("symbol", "start_date", "end_date", "lot_size"):
        if c not in lot.columns:
            raise ValueError(f"lot_size_map.parquet missing '{c}'")

    lot["symbol"] = lot["symbol"].map(normalize_symbol).astype("string")
    lot["start_date"] = pd.to_datetime(lot["start_date"], errors="coerce").dt.normalize()
    lot["end_date"] = pd.to_datetime(lot["end_date"], errors="coerce").dt.normalize()
    lot["lot_size"] = lot["lot_size"].astype("int64")

    if lot["start_date"].isna().any():
        raise ValueError("lot_size_map contains unparseable start_date")

    validate_no_overlap_lot_map(lot)
    return lot.sort_values(["symbol", "start_date"]).reset_index(drop=True)


def attach_lot_size(deriv: pd.DataFrame, lot: pd.DataFrame) -> pd.DataFrame:
    # Range join: date >= start_date AND (date <= end_date OR end_date is null)
    out_parts: List[pd.DataFrame] = []
    deriv = deriv.sort_values(["symbol", "date"]).reset_index(drop=True)

    for sym, g in deriv.groupby("symbol", sort=False):
        lm = lot[lot["symbol"] == sym].sort_values("start_date")
        if lm.empty:
            raise ValueError(f"No lot_size_map rows for symbol={sym}")

        merged = pd.merge_asof(
            g.sort_values("date"),
            lm[["start_date", "end_date", "lot_size"]].sort_values("start_date"),
            left_on="date",
            right_on="start_date",
            direction="backward",
            allow_exact_matches=True,
        )

        valid = merged["start_date"].notna() & (merged["end_date"].isna() | (merged["date"] <= merged["end_date"]))
        merged.loc[~valid, ["lot_size", "end_date"]] = np.nan

        # Keep end_date for schema/debug; drop start_date.
        merged = merged.drop(columns=["start_date"])
        out_parts.append(merged)

    out = pd.concat(out_parts, ignore_index=True)
    nulls = int(out["lot_size"].isna().sum())
    if nulls:
        raise ValueError(f"Unresolved lot_size for {nulls} rows after lot size mapping.")

    out["lot_size"] = out["lot_size"].astype("int64")
    # end_date remains datetime64[ns] with NaT allowed
    return out


def load_treasury_curve(path: Path) -> pd.DataFrame:
    curve = pd.read_parquet(path).copy()
    for c in ("date", "tenor", "rate"):
        if c not in curve.columns:
            raise ValueError(f"treasury_curve.parquet missing '{c}'")

    curve["date"] = parse_mixed_date_series(curve["date"].astype("string"), "treasury_curve.date")
    curve["tenor"] = curve["tenor"].astype("string").str.upper().str.strip()
    curve["rate"] = curve["rate"].astype("float64")

    def tenor_to_bucket(t: str) -> Optional[str]:
        if t is None:
            return None
        m = re.search(r"(\d+)", str(t))
        if not m:
            return None
        n = int(m.group(1))
        if n in (91, 182, 364):
            return f"rate_{n}d"
        return None

    curve["bucket"] = curve["tenor"].map(tenor_to_bucket)
    curve = curve[curve["bucket"].isin(["rate_91d", "rate_182d", "rate_364d"])].copy()

    wide = curve.pivot_table(index="date", columns="bucket", values="rate", aggfunc="last").reset_index()

    for c in ("rate_91d", "rate_182d", "rate_364d"):
        if c not in wide.columns:
            wide[c] = np.nan

        gt1 = wide[c] > 1.0
        if gt1.any():
            LOGGER.info("Treasury rate appears in percent for %s: converting %d rows to decimals", c, int(gt1.sum()))
            wide.loc[gt1, c] = wide.loc[gt1, c] / 100.0

        wide[c] = wide[c].astype("float64")

    wide = wide.sort_values("date").reset_index(drop=True)
    # Optional: fill inside treasury itself if some tenors missing on certain days
    wide[["rate_91d", "rate_182d", "rate_364d"]] = wide[["rate_91d", "rate_182d", "rate_364d"]].ffill()

    return wide


def load_trade_calendar(path: Path) -> pd.DataFrame:
    cal = pd.read_parquet(path).copy()
    for c in ("TradeDate", "Symbol", "Opt_Weekly_Expiry", "Opt_Monthly_Expiry"):
        if c not in cal.columns:
            raise ValueError(f"trade_calendar.parquet missing '{c}'")

    cal["date"] = pd.to_datetime(cal["TradeDate"], errors="coerce").dt.normalize()
    cal["symbol"] = cal["Symbol"].map(normalize_symbol).astype("string")
    cal["opt_weekly_expiry"] = pd.to_datetime(cal["Opt_Weekly_Expiry"], errors="coerce").dt.normalize()
    cal["opt_monthly_expiry"] = pd.to_datetime(cal["Opt_Monthly_Expiry"], errors="coerce").dt.normalize()

    if cal["date"].isna().any():
        raise ValueError("trade_calendar has unparseable TradeDate")

    return cal[["date", "symbol", "opt_weekly_expiry", "opt_monthly_expiry"]].drop_duplicates()


def build_trading_days_index(cal: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Returns dict(symbol -> sorted int64 day-ordinals) where ordinals are numpy datetime64[D] -> int64.

    NOTE: We avoid pandas Series.astype("datetime64[D]") because pandas raises:
      TypeError: Cannot cast DatetimeArray to dtype datetime64[D]
    """
    out: Dict[str, np.ndarray] = {}
    for sym, g in cal.groupby("symbol", sort=False):
        # numpy-safe conversion
        dates = g["date"].dropna().to_numpy(dtype="datetime64[ns]")
        day_ord = np.sort(dates.astype("datetime64[D]").astype("int64"))
        out[str(sym)] = day_ord
    return out


def _series_day_ordinal(s: pd.Series) -> np.ndarray:
    # Convert datetime64[ns] series to numpy int64 day ordinals (datetime64[D] -> int64)
    arr = s.to_numpy(dtype="datetime64[ns]")
    return arr.astype("datetime64[D]").astype("int64")


def compute_trading_days_to_expiry(df: pd.DataFrame, trading_days: Dict[str, np.ndarray]) -> pd.Series:
    # Count of trading days strictly after date and <= expiry_dt
    result = np.empty(len(df), dtype=np.int32)

    date_ord = _series_day_ordinal(df["date"])
    exp_ord = _series_day_ordinal(df["expiry_dt"])
    sym_arr = df["symbol"].astype("string").to_numpy()

    for sym in pd.unique(sym_arr):
        mask = sym_arr == sym
        idx = np.flatnonzero(mask)
        td = trading_days.get(str(sym))
        if td is None or len(td) == 0:
            raise ValueError(f"No trade_calendar trading days for symbol={sym}")

        d = date_ord[idx]
        e = exp_ord[idx]

        start_pos = np.searchsorted(td, d, side="right")
        end_pos = np.searchsorted(td, e, side="right")
        result[idx] = (end_pos - start_pos).astype(np.int32)

    return pd.Series(result, index=df.index, dtype="int32")


def compute_is_trading_day(df: pd.DataFrame, trading_days: Dict[str, np.ndarray]) -> pd.Series:
    out = np.empty(len(df), dtype=bool)
    date_ord = _series_day_ordinal(df["date"])
    sym_arr = df["symbol"].astype("string").to_numpy()

    for sym in pd.unique(sym_arr):
        mask = sym_arr == sym
        idx = np.flatnonzero(mask)
        td = trading_days.get(str(sym))
        if td is None or len(td) == 0:
            raise ValueError(f"No trade_calendar trading days for symbol={sym}")
        out[idx] = np.isin(date_ord[idx], td)

    return pd.Series(out, index=df.index, dtype="bool")


def compute_expiry_rank(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby(["date", "symbol", "instrument"], sort=False)["expiry_dt"]
        .rank(method="dense", ascending=True)
        .astype("int16")
    )


def compute_moneyness(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")

    opt = df["instrument"].eq("OPTIDX")
    if not opt.any():
        return out

    opt_typ = df["option_typ"].astype("string").str.upper().str.strip()

    calls = opt & opt_typ.isin(["CE", "C", "CALL"])
    puts = opt & opt_typ.isin(["PE", "P", "PUT"])

    # Unknown option types only among OPTIDX
    unknown = opt & ~(calls | puts)
    if unknown.any():
        bad = df.loc[unknown, "option_typ"].value_counts(dropna=False).to_dict()
        raise ValueError(f"Unknown option_typ values encountered: {bad}")

    out.loc[calls] = df.loc[calls, "spot_close"] - df.loc[calls, "strike_pr"]
    out.loc[puts] = df.loc[puts, "strike_pr"] - df.loc[puts, "spot_close"]
    return out



def validate_output(df: pd.DataFrame) -> None:
    bad_fut = int(df.loc[df["instrument"] == "FUTIDX", "strike_pr"].ne(0.0).sum())
    if bad_fut:
        raise AssertionError(f"Found {bad_fut} FUTIDX rows where strike_pr != 0.0")

    required_nonnull = ["spot_close", "vix_close", "lot_size", "rate_91d", "rate_182d", "rate_364d"]
    nulls = df[required_nonnull].isna().sum()
    if nulls.any():
        raise AssertionError(f"Nulls detected in required joined columns: {nulls[nulls > 0].to_dict()}")

    if (df["cal_days_to_expiry"] < 0).any():
        raise AssertionError("cal_days_to_expiry contains negative values after filtering")
    if (df["trading_days_to_expiry"] < 0).any():
        raise AssertionError("trading_days_to_expiry contains negative values")

    opt = df["instrument"] == "OPTIDX"
    if opt.any() and df.loc[opt, "moneyness"].isna().any():
        raise AssertionError("moneyness has nulls for OPTIDX rows")


def build_curated_derivatives(cfg: ETLConfig) -> pd.DataFrame:
    deriv = read_and_filter_derivatives(cfg)
    market = load_market_data(cfg.market_data_parquet)
    lot = load_lot_size_map(cfg.lot_size_map_parquet)
    treas = load_treasury_curve(cfg.treasury_curve_parquet)
    cal = load_trade_calendar(cfg.trade_calendar_parquet)

    trading_days = build_trading_days_index(cal)

    LOGGER.info("Joining market_data (spot/vix) ...")
    out = deriv.merge(market, on=["date", "symbol"], how="left", validate="m:1")
    if out["spot_close"].isna().any() or out["vix_close"].isna().any():
        missing = out[["spot_close", "vix_close"]].isna().sum().to_dict()
        raise ValueError(f"Missing spot/vix after join: {missing}")

    LOGGER.info("Attaching lot_size (effective-dated) ...")
    out = attach_lot_size(out, lot)

    LOGGER.info("Joining treasury curve (as-of backward on date) ...")
    out = out.sort_values("date").reset_index(drop=True)
    treas_sorted = treas.sort_values("date").reset_index(drop=True)
    out = pd.merge_asof(out, treas_sorted, on="date", direction="backward", allow_exact_matches=True)
    # Fail-fast if still missing
    if out[["rate_91d", "rate_182d", "rate_364d"]].isna().any().any():
        missing = out[["rate_91d", "rate_182d", "rate_364d"]].isna().sum().to_dict()
        raise ValueError(f"Missing treasury rates after merge_asof: {missing}")

    LOGGER.info("Joining trade_calendar expiry context (left join + flags) ...")
    out = out.merge(cal, on=["date", "symbol"], how="left", validate="m:1", indicator=True)
    out["is_trade_calendar_date"] = out["_merge"].eq("both")
    out = out.drop(columns=["_merge"])

    # Holiday/weekend flag (same meaning here)
    out["is_trading_day"] = compute_is_trading_day(out, trading_days)

    # Expiry-type flags (False when calendar row missing)
    out["is_opt_weekly_expiry"] = out["expiry_dt"].eq(out["opt_weekly_expiry"])
    out["is_opt_monthly_expiry"] = out["expiry_dt"].eq(out["opt_monthly_expiry"])
    out["is_opt_weekly_expiry"] = out["is_opt_weekly_expiry"].fillna(False).astype("bool")
    out["is_opt_monthly_expiry"] = out["is_opt_monthly_expiry"].fillna(False).astype("bool")

    LOGGER.info("Computing time-to-expiry metrics ...")
    out["cal_days_to_expiry"] = (out["expiry_dt"] - out["date"]).dt.days.astype("int32")

    neg_mask = out["cal_days_to_expiry"] < 0
    neg_cnt = int(neg_mask.sum())
    if neg_cnt:
        rate = neg_cnt / len(out)
        LOGGER.warning("Dropping rows with negative cal_days_to_expiry: %d (%.4f%%)", neg_cnt, rate * 100)
        if rate > cfg.max_drop_negative_tte_rate:
            raise ValueError(f"Too many negative cal_days_to_expiry rows: {rate:.4%}")
        out = out.loc[~neg_mask].copy()

    out["trading_days_to_expiry"] = compute_trading_days_to_expiry(out, trading_days)

    LOGGER.info("Computing expiry_rank ...")
    out["expiry_rank"] = compute_expiry_rank(out)

    LOGGER.info("Computing moneyness ...")
    out["moneyness"] = compute_moneyness(out)

    validate_output(out)

    preferred = [
        "date",
        "timestamp",
        "symbol",
        "instrument",
        "expiry_dt",
        "expiry_rank",
        "strike_pr",
        "option_typ",
        "open",
        "high",
        "low",
        "close",
        "settle_pr",
        "contracts",
        "open_int",
        "chg_in_oi",
        "spot_close",
        "vix_close",
        "div_yield",
        "lot_size",
        "end_date",
        "rate_91d",
        "rate_182d",
        "rate_364d",
        "cal_days_to_expiry",
        "trading_days_to_expiry",
        "opt_weekly_expiry",
        "opt_monthly_expiry",
        "is_trade_calendar_date",
        "is_trading_day",
        "is_opt_weekly_expiry",
        "is_opt_monthly_expiry",
        "moneyness",
    ]
    existing = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in existing]
    out = out[existing + remaining].copy()

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build curated derivatives dataset (golden source).")
    p.add_argument("--derivatives-csv", default="data/processed/Nifty_Historical_Derivatives.csv")
    p.add_argument("--market-data", default="data/processed/market_data.parquet")
    p.add_argument("--lot-size-map", default="data/processed/lot_size_map.parquet")
    p.add_argument("--treasury-curve", default="data/processed/treasury_curve.parquet")
    p.add_argument("--trade-calendar", default="data/processed/trade_calendar.parquet")
    p.add_argument("--output", default="data/curated/derivatives_clean.parquet")
    p.add_argument("--chunksize", type=int, default=750_000)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    cfg = ETLConfig(
        derivatives_csv=Path(args.derivatives_csv),
        market_data_parquet=Path(args.market_data),
        lot_size_map_parquet=Path(args.lot_size_map),
        treasury_curve_parquet=Path(args.treasury_curve),
        trade_calendar_parquet=Path(args.trade_calendar),
        output_parquet=Path(args.output),
        chunksize=int(args.chunksize),
    )

    LOGGER.info("Output path: %s", cfg.output_parquet)
    cfg.output_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = build_curated_derivatives(cfg)
    LOGGER.info("Writing parquet: rows=%d cols=%d", len(df), df.shape[1])
    df.to_parquet(cfg.output_parquet, index=False)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
