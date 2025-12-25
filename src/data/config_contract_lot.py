import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import io
import random
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# Path logic: src/data -> ../../ (root) -> data/raw
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Updated Target Path for the CSV as requested
CONFIG_CSV_PATH = RAW_DIR / "lot_size_map.csv"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- HARDCODED DATA (FALLBACK) ---
# [cite: 33] Lot sizes and contract specifications are critical for P&L calculation.
DEFAULT_CONFIG_CSV = """symbol,start_date,end_date,lot_size
NIFTY,2019-01-01,2021-04-29,75
NIFTY,2021-04-30,2024-04-25,50
NIFTY,2024-04-26,2024-11-19,25
NIFTY,2024-11-20,2025-10-27,75
NIFTY,2025-10-28,,65
BANKNIFTY,2019-01-01,2020-05-03,20
BANKNIFTY,2020-05-04,2023-04-27,25
BANKNIFTY,2023-04-28,2024-11-19,15
BANKNIFTY,2024-11-20,2025-04-24,30
BANKNIFTY,2025-04-25,2025-10-27,35
BANKNIFTY,2025-10-28,,30
"""

def ensure_config_exists():
    """
    Checks if lot_size_map.csv exists in data/raw.
    If not, creates it using the hardcoded default data.
    """
    if not CONFIG_CSV_PATH.exists():
        print(f"[WARN] Config not found at {CONFIG_CSV_PATH}")
        print("[INFO] Creating default lot_size_map.csv in data/raw...")

        with open(CONFIG_CSV_PATH, "w") as f:
            f.write(DEFAULT_CONFIG_CSV)

        print("[SUCCESS] Created default configuration file.")
    else:
        print(f"[INFO] Found existing config at: {CONFIG_CSV_PATH}")

def process_lot_size_config():
    """
    Reads the CSV config from data/raw and saves to data/processed as Parquet.
    """
    ensure_config_exists()

    print(f"[INFO] Processing Lot Size Config...")
    df_config = pd.read_csv(CONFIG_CSV_PATH)

    # Normalize Dates
    df_config['start_date'] = pd.to_datetime(df_config['start_date'])
    df_config['end_date'] = pd.to_datetime(df_config['end_date'])

    # Basic Validation [cite: 387]
    for symbol in df_config['symbol'].unique():
        symbol_rows = df_config[df_config['symbol'] == symbol]
        print(f"Loaded config for {symbol}: {len(symbol_rows)} periods.")

    # Save to processed
    output_path = PROCESSED_DIR / "lot_size_map.parquet"
    df_config.to_parquet(output_path)
    print(f"[SUCCESS] Lot Size Map saved to: {output_path}")
    return df_config

def get_lot_size(target_date, symbol, df_lot_config):
    """
    HELPER FUNCTION: Implements logic: Start <= Target <= End (OR End is Null)
    Essential for calculating Rupee P&L accurately[cite: 35].
    """
    target_date = pd.Timestamp(target_date)

    # Filter by symbol
    subset = df_lot_config[df_lot_config['symbol'] == symbol].copy()

    # Logic: start_date <= target_date <= end_date (or end_date is NaN)
    mask = (
        (subset['start_date'] <= target_date) &
        ((subset['end_date'] >= target_date) | (subset['end_date'].isnull()))
    )

    match = subset[mask]

    if match.empty:
        raise ValueError(f"No Lot Size defined for {symbol} on {target_date}")

    if len(match) > 1:
        match = match.sort_values('start_date', ascending=False)
        warnings.warn(f"Multiple lot sizes found for {symbol} on {target_date}. Using latest start_date.")

    return int(match.iloc[0]['lot_size'])

def get_random_date(start_str, end_str):
    """
    Returns a random datetime string for testing over the project horizon[cite: 251].
    """
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    delta = end - start
    random_days = random.randint(0, delta.days)

    random_dt = start + timedelta(days=random_days)
    return random_dt.strftime("%Y-%m-%d")

if __name__ == "__main__":
    try:
        # 1. Process Config from raw to processed
        df_config = process_lot_size_config()

        # 2. Smoke Tests
        print("\n--- Testing Lot Size Logic (Smoke Test) ---")

        # Test Case 1: Hardcoded Historical (Historical NIFTY lot was 75)
        date_1 = "2020-01-01"
        size_1 = get_lot_size(date_1, "NIFTY", df_config)
        print(f"Test 1 (Fixed): NIFTY on {date_1} -> {size_1} (Expected: 75)")

        # Test Case 2: Random Date NIFTY
        date_2 = get_random_date("2019-07-01", "2025-12-18")
        size_2 = get_lot_size(date_2, "NIFTY", df_config)
        print(f"Test 2 (Random): NIFTY on {date_2} -> {size_2}")

        # Test Case 3: Random Date BANKNIFTY
        date_3 = get_random_date("2019-07-01", "2025-12-18")
        size_3 = get_lot_size(date_3, "BANKNIFTY", df_config)
        print(f"Test 3 (Random): BANKNIFTY on {date_3} -> {size_3}")

    except Exception as e:
        print(f"[ERROR] Failed: {e}")
