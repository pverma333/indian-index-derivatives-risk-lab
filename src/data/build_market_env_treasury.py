from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger("build_market_env_treasury")


@dataclass(frozen=True)
class BuildConfig:
    raw_dir: Path
    processed_dir: Path

    spot_csv: str = "Index_Spot_Prices.csv"
    vix_csv: str = "India_VIX_Historical.csv"

    bond_files: Tuple[Tuple[str, str], ...] = (
        ("3-MonBondYield.csv", "91D"),
        ("6-MonBondYield.csv", "182D"),
        ("1-YearBondYield.csv", "364D"),
    )

    market_out: str = "market_data.parquet"
    treasury_out: str = "treasury_curve.parquet"

    vix_ffill_limit_days: int = 5
    div_ffill_limit_days: int = 5


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip header whitespace deterministically (raw CSVs often contain trailing spaces)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _require_columns(df: pd.DataFrame, required: Iterable[str], ctx: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx}: missing required columns {missing}. Found: {list(df.columns)}")


def _find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lowered:
            return lowered[key]
    return None


def _parse_date_series_strict(s: pd.Series, fmt: str, ctx: str) -> pd.Series:
    parsed = pd.to_datetime(s, format=fmt, errors="coerce")
    bad = int(parsed.isna().sum())
    if bad:
        sample = s[parsed.isna()].head(5).tolist()
        raise ValueError(f"{ctx}: date parse failures={bad}/{len(s)} using fmt={fmt}. Sample bad values={sample}")
    return parsed.dt.strftime("%Y-%m-%d")


def normalize_index_name(raw: str) -> str:
    """
    Normalize different naming conventions to two canonical keys:
    - NIFTY
    - BANKNIFTY
    """
    s = "" if raw is None else str(raw).upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    if "BANK" in s:
        return "BANKNIFTY"
    if "NIFTY" in s:
        return "NIFTY"
    return s


def percent_to_decimal(series: pd.Series, ctx: str) -> pd.Series:
    """
    Convert percent-quoted values to decimals.
    Examples:
      "7.123"  -> 0.07123
      "7.123%" -> 0.07123
    """
    s = series.astype("string").str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    out = pd.to_numeric(s, errors="coerce").astype("float64") / 100.0
    bad = int(out.isna().sum())
    if bad:
        LOGGER.warning("%s: numeric parse failures=%d/%d", ctx, bad, len(out))
    return out


def write_parquet_stable(
    df: pd.DataFrame,
    out_path: Path,
    *,
    sort_cols: Iterable[str],
    column_types: Dict[str, str],
    compression: str = "zstd",
    row_group_size: int = 100_000,
) -> None:
    """
    Stable Parquet writer (deterministic within the same runtime environment):
    - stable sort order
    - explicit dtypes
    - stripped schema metadata
    - fixed writer options
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()

    for col, t in column_types.items():
        if col not in out.columns:
            raise KeyError(f"write_parquet_stable: missing column '{col}'. Found={list(out.columns)}")
        if t == "string":
            out[col] = out[col].astype("string")
        elif t == "float64":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        else:
            raise ValueError(f"write_parquet_stable: unsupported dtype '{t}' for column '{col}'")

    out = out.sort_values(list(sort_cols), kind="mergesort").reset_index(drop=True)

    table = pa.Table.from_pandas(out, preserve_index=False).replace_schema_metadata({})
    pq.write_table(
        table,
        where=str(out_path),
        compression=compression,
        row_group_size=row_group_size,
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
    )


def load_spot_prices(processed_dir: Path, spot_csv: str) -> pd.DataFrame:
    path = processed_dir / spot_csv
    LOGGER.info("Loading spot prices: %s", path)
    df = _strip_columns(pd.read_csv(path))

    date_col = _find_first_column(df, ["date", "Date", "DATE"])
    idx_col = _find_first_column(df, ["index_name", "Index", "INDEX", "Symbol", "SYMBOL", "Name", "Index Name"])
    close_col = _find_first_column(df, ["spot_close", "close", "Close", "CLOSE", "SpotClose", "Spot Close"])
    open_col = _find_first_column(df, ["open", "Open", "OPEN"])

    if not (date_col and idx_col and close_col and open_col):
        raise ValueError(f"Spot prices: could not infer date/index/close/open columns. columns={list(df.columns)}")

    out = df[[date_col, idx_col, close_col, open_col]].rename(
        columns={date_col: "date", idx_col: "index_name", close_col: "spot_close", open_col: "index_open_price"}
    )

    parsed = pd.to_datetime(out["date"], errors="coerce")
    bad = int(parsed.isna().sum())
    if bad:
        raise ValueError(f"Spot prices: unparseable dates={bad}/{len(out)}")
    out["date"] = parsed.dt.strftime("%Y-%m-%d")

    out["index_name"] = out["index_name"].map(normalize_index_name)
    out["spot_close"] = pd.to_numeric(out["spot_close"], errors="coerce").astype("float64")
    out["index_open_price"] = pd.to_numeric(out["index_open_price"], errors="coerce").astype("float64")

    if out["spot_close"].isna().any():
        raise ValueError("Spot prices: spot_close contains nulls; spot is base and must be complete.")

    out = out[out["index_name"].isin(["NIFTY", "BANKNIFTY"])].copy()
    if out.empty:
        raise ValueError("Spot prices: no rows for NIFTY/BANKNIFTY after normalization/filtering.")

    return out[["date", "index_name", "spot_close", "index_open_price"]]


def load_vix(processed_dir: Path, vix_csv: str) -> pd.DataFrame:
    path = processed_dir / vix_csv
    LOGGER.info("Loading VIX: %s", path)
    df = _strip_columns(pd.read_csv(path))

    date_col = _find_first_column(df, ["date", "Date", "DATE"])
    close_col = _find_first_column(df, ["vix_close", "close", "Close", "CLOSE", "VIX Close", "VIX_Close"])
    if not (date_col and close_col):
        raise ValueError(f"VIX: could not infer date/close columns. columns={list(df.columns)}")

    out = df[[date_col, close_col]].rename(columns={date_col: "date", close_col: "vix_close"})

    parsed = pd.to_datetime(out["date"], errors="coerce")
    bad = int(parsed.isna().sum())
    if bad:
        raise ValueError(f"VIX: unparseable dates={bad}/{len(out)}")
    out["date"] = parsed.dt.strftime("%Y-%m-%d")
    out["vix_close"] = pd.to_numeric(out["vix_close"], errors="coerce").astype("float64")

    return out[["date", "vix_close"]]


def load_dividend_yields(raw_dir: Path) -> pd.DataFrame:
    """
    Process all NIFTY/BANKNIFTY yield CSVs under raw_dir matching '*yield-*.csv'.
    Date format: dd-mmm-yyyy (e.g., 01-Jan-2020)
    Dividend yield is percent -> convert to decimal.
    """
    files = sorted(raw_dir.glob("*yield-*.csv"))
    if not files:
        raise ValueError(f"No dividend yield files found under {raw_dir} matching '*yield-*.csv'")

    frames = []
    for fp in files:
        df = _strip_columns(pd.read_csv(fp))

        date_col = _find_first_column(df, ["date", "Date", "DATE"])
        yld_col = _find_first_column(
            df,
            [
                "Div Yield%", "Div Yield %",
                "Div Yield",
                "Dividend Yield", "DividendYield",
                "Yield", "YIELD",
            ],
        )
        if not (date_col and yld_col):
            raise ValueError(f"Dividend yields: could not infer columns in {fp.name}. columns={list(df.columns)}")

        idx_name = normalize_index_name(fp.name)

        tmp = df[[date_col, yld_col]].rename(columns={date_col: "date", yld_col: "div_yield"})
        tmp["index_name"] = idx_name

        tmp["date"] = _parse_date_series_strict(tmp["date"], fmt="%d-%b-%Y", ctx=f"Dividend yields ({fp.name})")
        tmp["div_yield"] = percent_to_decimal(tmp["div_yield"], ctx=f"Dividend yields ({fp.name})")

        frames.append(tmp[["date", "index_name", "div_yield"]])

    out = pd.concat(frames, ignore_index=True)
    out = out[out["index_name"].isin(["NIFTY", "BANKNIFTY"])].copy()
    if out.empty:
        raise ValueError("Dividend yields: no rows for NIFTY/BANKNIFTY after filtering.")

    before = len(out)
    out = out.sort_values(["index_name", "date"], kind="mergesort").drop_duplicates(
        subset=["index_name", "date"], keep="last"
    )
    after = len(out)
    if after != before:
        LOGGER.info("Dividend yields: dropped duplicates %d -> %d", before, after)

    return out


def load_bond_yield_file(raw_dir: Path, filename: str, tenor: str) -> pd.DataFrame:
    """
    Logic to handle mixed Price/Yield reporting in raw Treasury files:
    1. If value is 0-15: consider it Yield % and convert to decimal (val/100).
    2. If value is 60-105: consider it Price and convert to Yield ((100-price)/100).
    """
    path = raw_dir / filename
    LOGGER.info("Loading bond yield: %s (tenor=%s)", path, tenor)
    df = _strip_columns(pd.read_csv(path))

    _require_columns(df, ["Date", "Price"], f"Bond yield ({filename})")

    out = df[["Date", "Price"]].rename(columns={"Date": "date", "Price": "raw_val"})
    out["date"] = _parse_date_series_strict(out["date"], fmt="%d-%m-%Y", ctx=f"Bond yield ({filename})")

    # Clean raw string value
    s = out["raw_val"].astype("string").str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    vals = pd.to_numeric(s, errors="coerce")

    def process_rate(v):
        if pd.isna(v):
            return np.nan
        # 1. 0-15 -> Yield % -> decimal
        if 0 <= v <= 15:
            return v / 100.0
        # 2. 60-105 -> Price -> (100-price)/100
        if 60 <= v <= 105:
            return (100.0 - v) / 100.0
        return np.nan

    out["rate"] = vals.apply(process_rate)
    out["tenor"] = tenor

    return out[["date", "tenor", "rate"]]


def build_treasury_curve(raw_dir: Path, spot_min_date: str, spot_max_date: str, cfg: BuildConfig) -> pd.DataFrame:
    """
    Treasury curve construction:
    - Starts strictly from July 1, 2019.
    - Forward-fills across calendar days through the end of the spot range.
    """
    parts = [load_bond_yield_file(raw_dir, fn, tenor) for fn, tenor in cfg.bond_files]
    tdf = pd.concat(parts, ignore_index=True)

    # Constraint: Start filling from July 1 2019
    target_start = "2019-07-01"

    treasury_min = tdf["date"].min()
    if target_start < treasury_min:
        LOGGER.warning("Requested start %s is earlier than first treasury data point %s. Starting from %s.",
                       target_start, treasury_min, treasury_min)
        actual_start = treasury_min
    else:
        actual_start = target_start

    end_date = max(spot_max_date, tdf["date"].max())

    if actual_start > end_date:
        raise ValueError(f"Treasury curve: invalid date range start={actual_start} end={end_date}")

    all_dates = pd.date_range(start=actual_start, end=end_date, freq="D").strftime("%Y-%m-%d")
    wide = tdf.pivot(index="date", columns="tenor", values="rate").reindex(all_dates).sort_index()

    nulls_before = int(wide.isna().sum().sum())
    wide = wide.ffill()  # forward-fill across weekends/holidays
    nulls_after = int(wide.isna().sum().sum())

    LOGGER.info("Treasury curve: forward-fill across calendar. nulls before=%d after=%d", nulls_before, nulls_after)

    # Start row check
    if wide.iloc[0].isna().any():
        LOGGER.warning("Treasury curve: Start row still has nulls. First data point might be after %s. Attempting minor bfill.", actual_start)
        wide = wide.bfill(limit=30)

    if wide.isna().any().any():
        sample = wide[wide.isna().any(axis=1)].head(10)
        raise ValueError(f"Treasury curve: still has nulls after ffill/bfill. Sample rows:\n{sample}")

    out = wide.reset_index().rename(columns={"index": "date"}).melt(
        id_vars=["date"],
        var_name="tenor",
        value_name="rate",
    )
    out["rate"] = out["rate"].astype("float64")
    return out[["date", "tenor", "rate"]]


def build_market_data(spot: pd.DataFrame, vix: pd.DataFrame, div: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    LOGGER.info(
        "Merging market context (spot base). spot=%d vix=%d div=%d",
        len(spot), len(vix), len(div),
    )

    merged = spot.merge(vix, on="date", how="left").merge(div, on=["date", "index_name"], how="left")
    merged = merged.sort_values(["index_name", "date"], kind="mergesort").reset_index(drop=True)

    for col, lim in [("vix_close", cfg.vix_ffill_limit_days), ("div_yield", cfg.div_ffill_limit_days)]:
        before = int(merged[col].isna().sum())
        merged[col] = merged.groupby("index_name", sort=False)[col].ffill(limit=lim)
        after = int(merged[col].isna().sum())
        LOGGER.info("Market data: ffill %s limit=%d nulls %d -> %d", col, lim, before, after)

    if merged["spot_close"].isna().any():
        raise ValueError("Market data: spot_close has nulls (unexpected with spot base).")

    if merged["vix_close"].isna().any():
        n = int(merged["vix_close"].isna().sum())
        sample = merged.loc[merged["vix_close"].isna(), ["date", "index_name"]].head(10)
        raise ValueError(f"Market data: vix_close still has nulls={n} after ffill. Sample:\n{sample}")

    return merged[["date", "index_name", "spot_close", "index_open_price", "vix_close", "div_yield"]]


def run(cfg: BuildConfig) -> None:
    spot = load_spot_prices(cfg.processed_dir, cfg.spot_csv)
    vix = load_vix(cfg.processed_dir, cfg.vix_csv)
    div = load_dividend_yields(cfg.raw_dir)

    market = build_market_data(spot, vix, div, cfg)

    spot_min = spot["date"].min()
    spot_max = spot["date"].max()
    treasury = build_treasury_curve(cfg.raw_dir, spot_min, spot_max, cfg)

    market_path = cfg.processed_dir / cfg.market_out
    treasury_path = cfg.processed_dir / cfg.treasury_out

    LOGGER.info("Writing: %s", market_path)
    write_parquet_stable(
        market,
        market_path,
        sort_cols=["date", "index_name"],
        column_types={
            "date": "string",
            "index_name": "string",
            "spot_close": "float64",
            "index_open_price": "float64",
            "vix_close": "float64",
            "div_yield": "float64",
        },
    )

    LOGGER.info("Writing: %s", treasury_path)
    write_parquet_stable(
        treasury,
        treasury_path,
        sort_cols=["date", "tenor"],
        column_types={
            "date": "string",
            "tenor": "string",
            "rate": "float64",
        },
    )

    LOGGER.info("Done. Outputs written:\n- %s\n- %s", market_path, treasury_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build market environment + treasury curve processed layer.")
    p.add_argument("--raw-dir", default="data/raw", type=str)
    p.add_argument("--processed-dir", default="data/processed", type=str)
    p.add_argument("--log-level", default="INFO", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = BuildConfig(raw_dir=Path(args.raw_dir), processed_dir=Path(args.processed_dir))
    run(cfg)


if __name__ == "__main__":
    main()
