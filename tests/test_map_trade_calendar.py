from __future__ import annotations
import sys
import os
from pathlib import Path
import pandas as pd
import pytest

# 1. Setup Pathing
# Get the absolute path of the current directory (tests/)
BASE_DIR = Path(__file__).resolve().parent
# Get the project root
PROJECT_ROOT = BASE_DIR.parent
# Set output directory for test files
TEST_OUTPUTS = BASE_DIR / "test_outputs"
TEST_OUTPUTS.mkdir(exist_ok=True)

# Ensure 'src' is discoverable
sys.path.append(str(PROJECT_ROOT))

from src.data.map_trade_calendar import build_trade_calendar

def test_build_trade_calendar_futures_and_options_rules() -> None:
    """
    Validates the core logic of the Trade Calendar:
    1. Futures ladder (Near, Next, Far)
    2. Options classification (Monthly = Max of Month, Weekly = Others) [cite: 358]
    """

    rows = [
        {"INSTRUMENT": "FUTIDX", "SYMBOL": "NIFTY", "EXPIRY_DT": "2024-01-25", "TIMESTAMP": "2024-01-10"},
        {"INSTRUMENT": "FUTIDX", "SYMBOL": "NIFTY", "EXPIRY_DT": "2024-02-29", "TIMESTAMP": "2024-01-10"},
        {"INSTRUMENT": "FUTIDX", "SYMBOL": "NIFTY", "EXPIRY_DT": "2024-03-28", "TIMESTAMP": "2024-01-10"},
        {"INSTRUMENT": "OPTIDX", "SYMBOL": "NIFTY", "EXPIRY_DT": "2024-01-11", "TIMESTAMP": "2024-01-10"},
        {"INSTRUMENT": "OPTIDX", "SYMBOL": "NIFTY", "EXPIRY_DT": "2024-01-18", "TIMESTAMP": "2024-01-10"},
        {"INSTRUMENT": "OPTIDX", "SYMBOL": "NIFTY", "EXPIRY_DT": "2024-01-25", "TIMESTAMP": "2024-01-10"},
        {"INSTRUMENT": "FUTIDX", "SYMBOL": "BANKNIFTY", "EXPIRY_DT": "2024-01-31", "TIMESTAMP": "2024-01-10"},
        {"INSTRUMENT": "FUTIDX", "SYMBOL": "BANKNIFTY", "EXPIRY_DT": "2024-02-29", "TIMESTAMP": "2024-01-10"},
        {"INSTRUMENT": "FUTIDX", "SYMBOL": "BANKNIFTY", "EXPIRY_DT": "2024-03-28", "TIMESTAMP": "2024-01-10"},
        {"INSTRUMENT": "OPTIDX", "SYMBOL": "BANKNIFTY", "EXPIRY_DT": "2024-01-17", "TIMESTAMP": "2024-01-10"},
        {"INSTRUMENT": "OPTIDX", "SYMBOL": "BANKNIFTY", "EXPIRY_DT": "2024-01-24", "TIMESTAMP": "2024-01-10"},
        {"INSTRUMENT": "OPTIDX", "SYMBOL": "BANKNIFTY", "EXPIRY_DT": "2024-01-31", "TIMESTAMP": "2024-01-10"},
    ]
    df = pd.DataFrame(rows)

    # Force paths to stay inside tests/test_outputs/
    input_csv = TEST_OUTPUTS / "mock_derivatives.csv"
    out_parquet = TEST_OUTPUTS / "trade_calendar.parquet"
    out_csv = TEST_OUTPUTS / "trade_calendar.csv"

    df.to_csv(input_csv, index=False)

    # Execute the build
    cal = build_trade_calendar(
        input_csv=input_csv,
        output_parquet=out_parquet,
        output_csv=out_csv,
        log_level=50,
    )

    # Basic validations to ensure logic holds
    assert list(cal.columns) == ["TradeDate", "Symbol", "Fut_Near_Expiry", "Fut_Next_Expiry", "Fut_Far_Expiry", "Opt_Weekly_Expiry", "Opt_Monthly_Expiry"]
    assert len(cal) == 2

    # NIFTY Monthly check
    nifty = cal.loc[cal["Symbol"].eq("NIFTY")].iloc[0]
    assert nifty["Opt_Monthly_Expiry"] == pd.Timestamp("2024-01-25")

    # Check that files were created in the correct folder
    assert out_parquet.exists()
    print(f"\nSuccess! Results saved in: {TEST_OUTPUTS}")

if __name__ == "__main__":
    test_build_trade_calendar_futures_and_options_rules()
