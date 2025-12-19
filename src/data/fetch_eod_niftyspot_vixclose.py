import pandas as pd
import requests
import io
import time
import os
from datetime import date, timedelta
from pathlib import Path

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

def fetch_daily_archive(target_date):
    """Downloads daily consolidated index reports from NSE Archive."""
    date_str = target_date.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/content/indices/ind_close_all_{date_str}.csv"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            # NSE Archive CSVs often have leading/trailing spaces in headers
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = [c.strip() for c in df.columns]
            return df
        return None
    except Exception:
        return None

def main():
    # SET YOUR FULL TIMELINE HERE
    start_date = date(2019, 1, 1)
    end_date = date(2025, 12, 17)

    spot_data = []
    vix_data = []
    curr = start_date

    print(f"Starting Master Fetch (Spot + VIX) from {start_date}...")

    while curr <= end_date:
        # Skip weekends
        if curr.weekday() < 5:
            df = fetch_daily_archive(curr)
            if df is not None:
                # 1. Process Spot Prices (Nifty 50 & Bank Nifty)
                # Note: Archive uses 'Index Name', 'Open Index Value', etc.
                spot_filter = df[df['Index Name'].isin(['Nifty 50', 'Nifty Bank'])].copy()
                if not spot_filter.empty:
                    spot_filter['Date'] = curr.strftime('%Y-%m-%d')
                    # Map the long archive names to our standard OHLC
                    temp_spot = spot_filter[['Date', 'Index Name', 'Open Index Value', 'High Index Value', 'Low Index Value', 'Closing Index Value']]
                    temp_spot.columns = ['Date', 'Index', 'Open', 'High', 'Low', 'Close']
                    spot_data.append(temp_spot)

                # 2. Process India VIX
                vix_filter = df[df['Index Name'] == 'India VIX'].copy()
                if not vix_filter.empty:
                    vix_filter['Date'] = curr.strftime('%Y-%m-%d')
                    temp_vix = vix_filter[['Date', 'Closing Index Value']]
                    temp_vix.columns = ['Date', 'VIX_Close']
                    vix_data.append(temp_vix)

                print(f"Captured: {curr}")

            # Politeness delay to avoid IP block
            time.sleep(0.1)
        curr += timedelta(days=1)

    # Save Spot Prices
    if spot_data:
        final_spot = pd.concat(spot_data, ignore_index=True)
        final_spot.to_csv(PROCESSED_PATH / "Index_Spot_Prices.csv", index=False)
        print(f"--- Saved Spot Prices: {PROCESSED_PATH / 'Index_Spot_Prices.csv'} ---")

    # Save VIX
    if vix_data:
        final_vix = pd.concat(vix_data, ignore_index=True)
        final_vix.to_csv(PROCESSED_PATH / "India_VIX_Historical.csv", index=False)
        print(f"--- Saved VIX Data: {PROCESSED_PATH / 'India_VIX_Historical.csv'} ---")

if __name__ == "__main__":
    main()
