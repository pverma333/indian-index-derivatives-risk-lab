"""
fetch_eod.py

Skeleton for NSE EOD data fetching utilities for the Indian Index Derivatives Risk Lab.

Responsibilities (future):
- Download NIFTY spot index EOD data from NSE (2019-07-01 to present).
- Download NSE derivatives bhavcopy files (FUTIDX, OPTIDX, SYMBOL=NIFTY).
- Save raw files under data/raw/ (spot and derivatives separated).

For now:
- Provide clean function stubs and path handling only.
"""

from pathlib import Path
from datetime import date
from typing import Optional


# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_SPOT_DIR = RAW_DIR / "spot"
RAW_DERIVATIVES_DIR = RAW_DIR / "derivatives"


def ensure_directories() -> None:
    """
    Ensure all required raw data directories exist.
    This is safe to call multiple times.
    """
    RAW_SPOT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)


def fetch_nifty_spot_eod(
    start: date,
    end: date,
    index_symbol: str = "NIFTY 50",
) -> None:
    """
    Stub: Fetch NIFTY spot EOD index data from NSE between [start, end].

    Future behaviour:
    - Call NSE index EOD endpoint for each date or a date range.
    - Save combined raw CSV under data/raw/spot/.

    For now:
    - Just validate inputs and raise NotImplementedError.
    """
    if start > end:
        raise ValueError("start date must be <= end date")

    # TODO: implement actual HTTP calls to NSE index endpoint
    raise NotImplementedError("fetch_nifty_spot_eod is not implemented yet.")


def fetch_nse_derivatives_bhavcopy_for_range(
    start: date,
    end: date,
    symbol: str = "NIFTY",
) -> None:
    """
    Stub: Fetch NSE derivatives bhavcopy files (FUTIDX, OPTIDX) for NIFTY
    between [start, end] and save them into data/raw/derivatives/.

    Future behaviour:
    - For each date in range:
      - Build the NSE bhavcopy URL for that date.
      - Download and unzip (if needed).
      - Save the raw CSV file.
    - Only keep rows where INSTRUMENT in {FUTIDX, OPTIDX} and SYMBOL matches.

    For now:
    - Validate inputs and raise NotImplementedError.
    """
    if start > end:
        raise ValueError("start date must be <= end date")

    # TODO: implement actual HTTP calls to NSE derivatives bhavcopy endpoint
    raise NotImplementedError(
        "fetch_nse_derivatives_bhavcopy_for_range is not implemented yet."
    )


def main(
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> None:
    """
    Entry point for running this module as a script.

    For now:
    - Ensure directories exist.
    - Print a message so we know the skeleton works.

    Later:
    - Parse CLI args for date range.
    - Call fetch_nifty_spot_eod and fetch_nse_derivatives_bhavcopy_for_range.
    """
    ensure_directories()
    print("fetch_eod skeleton: directories ensured. No data fetched yet.")


if __name__ == "__main__":
    # Default: just run skeleton without dates
    main()

