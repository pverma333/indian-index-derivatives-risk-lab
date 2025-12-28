from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditConfig:
    parquet_path: str = "data/processed/derivatives_clean.parquet"

    # Contract grain / uniqueness key (ticket explicitly suggests this structure)
    unique_contract_key: Tuple[str, ...] = (
        "date",
        "symbol",
        "instrument",
        "expiry_dt",
        "option_typ",
        "strike_pr",
    )

    # Thresholds
    basis_spike_frac_threshold: float = 0.02   # abs(basis) > 2% * spot_close is flagged
    basis_spike_rate_threshold: float = 0.02   # if >2% of days spiky -> assert
    max_reasonable_rate: float = 1.0           # rate scaling check: must be decimals
    vix_min_std: float = 1e-6                  # must not be constant/zero
    random_seed: int = 42                      # deterministic "random" dates
    density_num_dates: int = 5
    density_num_strikes_each_side: int = 10

    # Project truth table (edit here if needed)
    # Interpretation: allowed lot sizes for (symbol, year).
    # If a year has multiple lot sizes (mid-year change), list both.
    lot_size_truth_table: Dict[str, Dict[int, List[int]]] = dataclasses.field(
        default_factory=lambda: {
            "NIFTY": {
                2019: [75],
                2020: [50],
                2021: [25],
                2022: [25],
                2023: [25],
                2024: [25],
                2025: [25],  # <- Acceptance requires explicit assertion for 2025.
            },
            "BANKNIFTY": {
                # Placeholder defaults; update if your project truth table differs
                2019: [20],
                2020: [25],
                2021: [15],
                2022: [15],
                2023: [15],
                2024: [15],
                2025: [15],
            },
        }
    )


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_derivatives_clean(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Ensure types are consistent/deterministic
    for col in ["date", "timestamp", "expiry_dt", "opt_weekly_expiry", "opt_monthly_expiry"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)

    if "instrument" in df.columns:
        df["instrument"] = df["instrument"].astype(str)

    if "option_typ" in df.columns:
        # Keep None/NaN, normalize blanks to empty string for grouping stability
        df["option_typ"] = df["option_typ"].replace({np.nan: None})
        df["option_typ"] = df["option_typ"].apply(lambda x: x if x is None else str(x).strip())

    return df


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing required columns: {missing}")


def assert_no_duplicate_contract_keys(df: pd.DataFrame, key_cols: Tuple[str, ...]) -> pd.DataFrame:
    require_columns(df, key_cols)
    dup = df.duplicated(list(key_cols), keep=False)
    dup_df = df.loc[dup, list(key_cols)].sort_values(list(key_cols))
    if not dup_df.empty:
        raise AssertionError(
            f"Holiday Drift / join duplication: found {len(dup_df)} duplicated rows "
            f"by unique contract key {key_cols}. Example:\n{dup_df.head(20)}"
        )
    return dup_df


def validate_time_to_expiry_non_negative(df: pd.DataFrame) -> None:
    require_columns(df, ["cal_days_to_expiry"])
    min_val = int(df["cal_days_to_expiry"].min())
    if min_val < 0:
        raise AssertionError(f"Negative Time: min(cal_days_to_expiry)={min_val} < 0")


def validate_rate_scaling(df: pd.DataFrame, cfg: AuditConfig) -> None:
    rate_cols = ["rate_91d", "rate_182d", "rate_364d"]
    require_columns(df, rate_cols)

    max_rate = float(df[rate_cols].max().max())
    if max_rate >= cfg.max_reasonable_rate:
        raise AssertionError(
            f"Rate Scaling: max rate observed={max_rate:.6f} >= {cfg.max_reasonable_rate}. "
            f"Rates should be decimals (e.g., 0.07 for 7%)."
        )


def validate_vix_variation(df: pd.DataFrame, cfg: AuditConfig) -> None:
    require_columns(df, ["vix_close", "symbol"])
    # If VIX join failed, it might be constant 0 or constant value. Check per symbol.
    by_sym = (
        df.groupby("symbol")["vix_close"]
        .agg(std="std", min="min", max="max", n="size")
        .reset_index()
    )
    bad = by_sym[by_sym["std"].fillna(0.0) <= cfg.vix_min_std]
    if not bad.empty:
        raise AssertionError(
            f"VIX Decay: vix_close appears constant/zero for symbols:\n{bad}"
        )


def validate_futures_strike_zero(df: pd.DataFrame) -> None:
    require_columns(df, ["instrument", "strike_pr"])
    fut = df[df["instrument"] == "FUTIDX"]
    if fut.empty:
        LOGGER.warning("No FUTIDX rows found; skipping futures strike=0 validation.")
        return
    bad = fut.loc[fut["strike_pr"] != 0.0, ["date", "symbol", "expiry_dt", "strike_pr"]]
    if not bad.empty:
        raise AssertionError(
            f"FUTIDX physicality: expected 100% strike_pr==0.0, found violations:\n{bad.head(50)}"
        )


def validate_moneyness_polarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    CE: moneyness = spot_close - strike_pr  (positive when spot > strike)
    PE: moneyness = strike_pr - spot_close  (positive when strike > spot)
    """
    require_columns(df, ["instrument", "option_typ", "spot_close", "strike_pr", "moneyness"])

    opt = df[df["instrument"] == "OPTIDX"].copy()
    if opt.empty:
        LOGGER.warning("No OPTIDX rows found; skipping moneyness polarity validation.")
        return pd.DataFrame()

    ce = opt[opt["option_typ"] == "CE"].copy()
    pe = opt[opt["option_typ"] == "PE"].copy()

    ce_expected = ce["spot_close"] - ce["strike_pr"]
    pe_expected = pe["strike_pr"] - pe["spot_close"]

    ce_bad = ce.loc[~np.isclose(ce["moneyness"], ce_expected, rtol=0, atol=1e-9)]
    pe_bad = pe.loc[~np.isclose(pe["moneyness"], pe_expected, rtol=0, atol=1e-9)]

    bad = pd.concat([ce_bad, pe_bad], axis=0, ignore_index=True)
    if not bad.empty:
        sample = bad[["date", "symbol", "expiry_dt", "option_typ", "spot_close", "strike_pr", "moneyness"]].head(50)
        raise AssertionError(f"Moneyness polarity mismatch vs definition. Sample:\n{sample}")

    return bad


def validate_expiry_rank_near(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["date", "symbol", "instrument", "expiry_dt", "expiry_rank"])

    # For each (date,symbol,instrument), expiry_rank==1 should match min(expiry_dt)
    gcols = ["date", "symbol", "instrument"]
    min_exp = df.groupby(gcols)["expiry_dt"].min().rename("min_expiry_dt").reset_index()
    rank1 = df[df["expiry_rank"] == 1][gcols + ["expiry_dt"]].drop_duplicates()

    merged = rank1.merge(min_exp, on=gcols, how="left")
    bad = merged[merged["expiry_dt"] != merged["min_expiry_dt"]]

    if not bad.empty:
        raise AssertionError(
            "Expiry rank check failed: expiry_rank==1 is not the near contract "
            f"for some groups. Sample:\n{bad.head(50)}"
        )
    return bad


def validate_expiry_day_rows(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["date", "expiry_dt", "cal_days_to_expiry"])
    exp0 = df[df["cal_days_to_expiry"] == 0].copy()
    if exp0.empty:
        raise AssertionError("Expected cal_days_to_expiry == 0 rows (expiry-day) but found none.")
    # Physical definition: date should equal expiry_dt on expiry day
    bad = exp0[exp0["date"] != exp0["expiry_dt"]]
    if not bad.empty:
        raise AssertionError(
            f"Expiry-day definition broken: cal_days_to_expiry==0 but date!=expiry_dt. Sample:\n{bad.head(50)}"
        )
    return exp0


def monthly_expiry_counts(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - per_year counts: (symbol, year) -> count of is_opt_monthly_expiry==True
      - per_month uniqueness: (symbol, year, month) -> count
    """
    require_columns(df, ["symbol", "expiry_dt", "is_opt_monthly_expiry"])

    m = df[df["is_opt_monthly_expiry"] == True].copy()  # noqa: E712
    if m.empty:
        raise AssertionError("No rows with is_opt_monthly_expiry==True found.")

    m["year"] = m["expiry_dt"].dt.year
    m["month"] = m["expiry_dt"].dt.month

    per_year = m.groupby(["symbol", "year"]).size().rename("monthly_expiry_events").reset_index()
    per_month = m.groupby(["symbol", "year", "month"]).size().rename("events_in_month").reset_index()
    return per_year, per_month


def assert_exactly_12_monthlies(per_year: pd.DataFrame) -> None:
    bad = per_year[per_year["monthly_expiry_events"] != 12]
    if not bad.empty:
        raise AssertionError(
            "Expected exactly 12 monthly expiry events per year per symbol. Violations:\n"
            f"{bad}"
        )


def assert_one_monthly_per_month(per_month: pd.DataFrame) -> None:
    bad = per_month[per_month["events_in_month"] != 1]
    if not bad.empty:
        raise AssertionError(
            "Expected exactly 1 monthly expiry per month per symbol. Violations:\n"
            f"{bad.head(50)}"
        )


def compute_basis_near_futures(df: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    require_columns(df, ["instrument", "expiry_rank", "settle_pr", "spot_close", "date", "symbol", "expiry_dt"])

    fut_near = df[(df["instrument"] == "FUTIDX") & (df["expiry_rank"] == 1)].copy()
    if fut_near.empty:
        raise AssertionError("No near futures rows found (instrument=FUTIDX, expiry_rank=1).")

    fut_near["basis"] = fut_near["settle_pr"] - fut_near["spot_close"]
    fut_near["basis_abs_frac"] = (fut_near["basis"].abs() / fut_near["spot_close"].replace(0.0, np.nan))

    fut_near["is_basis_spike"] = fut_near["basis_abs_frac"] > cfg.basis_spike_frac_threshold
    return fut_near


def assert_basis_spikes_reasonable(fut_near: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    spikes = fut_near[fut_near["is_basis_spike"]].copy()

    # Evaluate spike rate per symbol across dates (dedupe date/expiry just in case)
    spike_rate = (
        fut_near.drop_duplicates(["date", "symbol", "expiry_dt"])
        .groupby("symbol")["is_basis_spike"]
        .mean()
        .rename("spike_rate")
        .reset_index()
    )
    bad = spike_rate[spike_rate["spike_rate"] > cfg.basis_spike_rate_threshold]
    if not bad.empty:
        raise AssertionError(
            f"Basis spikes too frequent (>{cfg.basis_spike_rate_threshold:.2%}) for:\n{bad}\n"
            f"Sample spikes:\n{spikes[['date','symbol','expiry_dt','basis','basis_abs_frac']].head(50)}"
        )
    return spikes


def treasury_curve_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["rate_91d", "rate_364d", "date", "symbol"])
    bad = df[df["rate_364d"] < df["rate_91d"]][["date", "symbol", "rate_91d", "rate_364d"]].copy()
    return bad


def lot_size_audit_by_year(df: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    require_columns(df, ["symbol", "date", "lot_size"])
    tmp = df.copy()
    tmp["year"] = pd.to_datetime(tmp["date"]).dt.year
    out = (
        tmp.groupby(["symbol", "year"])["lot_size"]
        .agg(
            unique_lot_sizes=lambda s: sorted(set(int(x) for x in s.dropna().unique().tolist())),
            min_lot_size="min",
            max_lot_size="max",
            n="size",
        )
        .reset_index()
    )
    return out


def assert_lot_sizes_match_truth_table(lot_by_year: pd.DataFrame, cfg: AuditConfig) -> None:
    # Enforce only for years in the truth table; ignore others.
    violations: List[str] = []
    for symbol, year_map in cfg.lot_size_truth_table.items():
        for year, allowed in year_map.items():
            rows = lot_by_year[(lot_by_year["symbol"] == symbol) & (lot_by_year["year"] == year)]
            if rows.empty:
                violations.append(f"{symbol} {year}: no data found")
                continue
            observed = rows.iloc[0]["unique_lot_sizes"]
            if sorted(observed) != sorted(allowed):
                violations.append(f"{symbol} {year}: observed {observed} vs expected {allowed}")

    if violations:
        msg = "Lot size truth table violations:\n" + "\n".join(f"- {v}" for v in violations)
        raise AssertionError(msg)


def pick_deterministic_sample_dates(df: pd.DataFrame, cfg: AuditConfig) -> List[pd.Timestamp]:
    require_columns(df, ["date"])
    dates = sorted(pd.to_datetime(df["date"]).dropna().unique().tolist())
    if len(dates) == 0:
        raise AssertionError("No dates found in dataset.")
    rng = np.random.RandomState(cfg.random_seed)
    k = min(cfg.density_num_dates, len(dates))
    idx = rng.choice(len(dates), size=k, replace=False)
    return [pd.to_datetime(dates[i]).normalize() for i in sorted(idx)]


def infer_strike_step(strikes: np.ndarray) -> Optional[float]:
    strikes = np.array(sorted(set(float(x) for x in strikes if pd.notna(x) and float(x) > 0)))
    if len(strikes) < 3:
        return None
    diffs = np.diff(strikes)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    # Most common increment (robust-ish)
    return float(pd.Series(diffs).round(6).mode().iloc[0])


def strike_density_around_atm(
    df: pd.DataFrame,
    sample_date: pd.Timestamp,
    cfg: AuditConfig,
) -> pd.DataFrame:
    """
    Returns one row per (symbol, expiry_dt) for the chosen date summarizing
    observed strikes around ATM and missing holes in the expected grid.
    """
    require_columns(df, ["date", "symbol", "instrument", "expiry_rank", "option_typ", "strike_pr", "spot_close", "expiry_dt"])
    day = df[pd.to_datetime(df["date"]).dt.normalize() == pd.to_datetime(sample_date).normalize()].copy()
    day = day[(day["instrument"] == "OPTIDX") & (day["expiry_rank"] == 1)]
    if day.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    for (symbol, expiry_dt), g in day.groupby(["symbol", "expiry_dt"]):
        spot = float(g["spot_close"].iloc[0])
        strikes = np.array(sorted(set(float(x) for x in g["strike_pr"].dropna().unique().tolist() if float(x) > 0)))
        if len(strikes) == 0:
            continue

        step = infer_strike_step(strikes)
        if step is None or step <= 0:
            continue

        atm = float(strikes[np.argmin(np.abs(strikes - spot))])
        target = np.array([atm + i * step for i in range(-cfg.density_num_strikes_each_side, cfg.density_num_strikes_each_side + 1)])
        observed_set = set(np.round(strikes, 6))
        missing = [float(x) for x in np.round(target, 6) if float(x) not in observed_set]

        rows.append(
            {
                "date": pd.to_datetime(sample_date).normalize(),
                "symbol": symbol,
                "expiry_dt": pd.to_datetime(expiry_dt).normalize(),
                "spot_close": spot,
                "atm_strike": atm,
                "inferred_step": step,
                "expected_strikes": len(target),
                "observed_strikes_total": len(strikes),
                "missing_in_atm_window": len(missing),
                "missing_strikes": missing[:20],  # keep small
            }
        )
    return pd.DataFrame(rows)


def run_all_audits(df: pd.DataFrame, cfg: AuditConfig) -> Dict[str, pd.DataFrame]:
    configure_logging()

    LOGGER.info("Rows: %d | Cols: %d", len(df), df.shape[1])
    assert_no_duplicate_contract_keys(df, cfg.unique_contract_key)
    validate_time_to_expiry_non_negative(df)
    validate_rate_scaling(df, cfg)
    validate_vix_variation(df, cfg)
    validate_futures_strike_zero(df)
    validate_moneyness_polarity(df)
    validate_expiry_rank_near(df)
    expiry_day = validate_expiry_day_rows(df)

    per_year, per_month = monthly_expiry_counts(df)
    assert_exactly_12_monthlies(per_year)
    assert_one_monthly_per_month(per_month)

    fut_near = compute_basis_near_futures(df, cfg)
    basis_spikes = assert_basis_spikes_reasonable(fut_near, cfg)

    curve_viol = treasury_curve_monotonicity(df)
    lot_by_year = lot_size_audit_by_year(df, cfg)
    assert_lot_sizes_match_truth_table(lot_by_year, cfg)

    # Strike density (returns per-date summaries; notebook will plot)
    sample_dates = pick_deterministic_sample_dates(df, cfg)
    density_frames = []
    for d in sample_dates:
        density_frames.append(strike_density_around_atm(df, d, cfg))
    density = pd.concat(density_frames, ignore_index=True) if density_frames else pd.DataFrame()

    return {
        "monthly_expiry_per_year": per_year,
        "monthly_expiry_per_month": per_month,
        "expiry_day_rows": expiry_day[["date", "symbol", "instrument", "expiry_dt", "settle_pr", "close", "spot_close"]].head(200),
        "near_futures_basis": fut_near[["date", "symbol", "expiry_dt", "settle_pr", "spot_close", "basis", "basis_abs_frac", "is_basis_spike"]],
        "near_futures_basis_spikes": basis_spikes[["date", "symbol", "expiry_dt", "basis", "basis_abs_frac"]].sort_values(["symbol", "date"]),
        "treasury_curve_violations": curve_viol.sort_values(["symbol", "date"]),
        "lot_size_by_year": lot_by_year.sort_values(["symbol", "year"]),
        "strike_density_samples": density.sort_values(["date", "symbol"]),
    }
