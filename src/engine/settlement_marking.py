from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../<repo_root>
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging
from typing import Final

import numpy as np
import pandas as pd

from src.utils.errors import DataIntegrityError, _sample_indices

logger = logging.getLogger(__name__)

REQUIRED_COLS: Final[list[str]] = [
    "instrument",
    "option_typ",
    "strike_pr",
    "settle_pr",
    "spot_close",
    "cal_days_to_expiry",
]

PRICE_METHOD_SETTLE: Final[str] = "SETTLE_PR"
PRICE_METHOD_INTRINSIC: Final[str] = "INTRINSIC_ON_EXPIRY"


def compute_settle_used(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic EOD marking used by MTM and entry anchoring.

    Default (all rows):
      settle_used = settle_pr
      price_method = SETTLE_PR

    OPTIDX expiry override (instrument == OPTIDX and cal_days_to_expiry == 0):
      CE: max(spot_close - strike_pr, 0)
      PE: max(strike_pr - spot_close, 0)
      price_method = INTRINSIC_ON_EXPIRY

    FUTIDX:
      always uses settle_pr (no intrinsic override)
      price_method remains SETTLE_PR

    Raises DataIntegrityError on:
      - missing required columns
      - OPTIDX expiry rows with missing strike_pr or spot_close
      - OPTIDX expiry rows with option_typ not in {CE, PE}
      - settle_pr missing where price_method == SETTLE_PR
      - any negative settle_used
    """
    missing_cols = [c for c in REQUIRED_COLS if c not in market_df.columns]
    if missing_cols:
        raise DataIntegrityError(
            "market_df missing required columns for settle_used computation",
            details={"missing_cols": missing_cols},
        )

    df = market_df.copy()

    # Defaults
    df["settle_used"] = df["settle_pr"]
    df["price_method"] = PRICE_METHOD_SETTLE

    is_optidx = df["instrument"].astype(str) == "OPTIDX"
    is_expiry = df["cal_days_to_expiry"] == 0
    opt_expiry = is_optidx & is_expiry

    logger.info(
        "compute_settle_used: rows=%d opt_expiry_rows=%d",
        int(len(df)),
        int(opt_expiry.sum()),
    )

    if opt_expiry.any():
        missing_spot = opt_expiry & df["spot_close"].isna()
        missing_strike = opt_expiry & df["strike_pr"].isna()

        if missing_spot.any() or missing_strike.any():
            raise DataIntegrityError(
                "OPTIDX expiry rows require non-null spot_close and strike_pr",
                details={
                    "missing_spot_count": int(missing_spot.sum()),
                    "missing_strike_count": int(missing_strike.sum()),
                    "sample_indices_missing_spot": _sample_indices(missing_spot),
                    "sample_indices_missing_strike": _sample_indices(missing_strike),
                },
            )

        opt_typ = df.loc[opt_expiry, "option_typ"].astype(str)
        invalid_local = ~opt_typ.isin(["CE", "PE"])
        if invalid_local.any():
            invalid_idx = opt_typ.index[invalid_local]
            bad_vals = sorted(set(opt_typ.loc[invalid_idx].tolist()))
            raise DataIntegrityError(
                "OPTIDX expiry rows require option_typ in {CE, PE}",
                details={
                    "bad_option_typ_values": bad_vals[:10],
                    "bad_option_typ_count": int(invalid_local.sum()),
                    "sample_indices": [int(i) for i in list(invalid_idx[:10])],
                },
            )

        spot = df.loc[opt_expiry, "spot_close"].astype(float).to_numpy()
        strike = df.loc[opt_expiry, "strike_pr"].astype(float).to_numpy()
        is_ce = (opt_typ.to_numpy() == "CE")

        intrinsic = np.where(is_ce, spot - strike, strike - spot)
        intrinsic = np.maximum(intrinsic, 0.0)

        df.loc[opt_expiry, "settle_used"] = intrinsic
        df.loc[opt_expiry, "price_method"] = PRICE_METHOD_INTRINSIC

    # settle_pr required wherever using settle-based marking
    settle_rows = df["price_method"] == PRICE_METHOD_SETTLE
    missing_settle = settle_rows & df["settle_pr"].isna()
    if missing_settle.any():
        raise DataIntegrityError(
            "settle_pr is missing for rows marked with SETTLE_PR",
            details={
                "missing_settle_pr_count": int(missing_settle.sum()),
                "sample_indices": _sample_indices(missing_settle),
            },
        )

    # No negative settle_used
    negative = df["settle_used"].astype(float) < 0
    if negative.any():
        raise DataIntegrityError(
            "Negative settle_used detected (should never happen)",
            details={
                "negative_count": int(negative.sum()),
                "sample_indices": _sample_indices(negative),
            },
        )

    return df


if __name__ == "__main__":
    # Optional tiny smoke-run so calling the file directly doesn't look "dead".
    logging.basicConfig(level=logging.INFO)
    demo = pd.DataFrame(
        [
            {
                "instrument": "OPTIDX",
                "option_typ": "CE",
                "strike_pr": 20000.0,
                "settle_pr": 10.0,
                "spot_close": 20100.0,
                "cal_days_to_expiry": 0,
            }
        ]
    )
    out = compute_settle_used(demo)
    print(out[["settle_used", "price_method"]])
