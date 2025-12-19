import time
import requests
import zipfile
import io
import os
from datetime import date, timedelta, datetime
import pandas as pd
from pathlib import Path

# 1. Setup Robust Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
MASTER_FILE = PROCESSED_PATH / "Nifty_Historical_Derivatives.csv"

# THE TARGET 14-COLUMN SCHEMA
COLS = [
    'INSTRUMENT', 'SYMBOL', 'EXPIRY_DT', 'STRIKE_PR', 'OPTION_TYP',
    'OPEN', 'HIGH', 'LOW', 'CLOSE', 'SETTLE_PR', 'CONTRACTS',
    'OPEN_INT', 'CHG_IN_OI', 'TIMESTAMP'
]

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "*/*", "Connection": "keep-alive"}

def format_expiry_date(date_val):
    """Converts ISO dates (2024-07-25) to Legacy format (25-Jul-2024)"""
    try:
        # If it's already a string in ISO format
        dt = pd.to_datetime(date_val)
        return dt.strftime('%d-%b-%Y')
    except:
        return date_val

def get_bhavcopy_url(target_date):
    date_str = target_date.strftime("%Y%m%d")
    return f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"

def download_and_extract(target_date):
    url = get_bhavcopy_url(target_date)
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 404: return "HOLIDAY"
        if response.status_code != 200: raise Exception(f"HTTP {response.status_code}")

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
                return df.loc[:, ~df.columns.duplicated()]
    except Exception as e: raise e

def fetch_and_format_data(start_date, end_date):
    curr = start_date
    batch_data = []

    mapping = {
        'TradDt': 'TIMESTAMP', 'BizDt': 'TIMESTAMP', 'FinInstrmId': 'INSTRUMENT',
        'TckrSymb': 'SYMBOL', 'XpryDt': 'EXPIRY_DT', 'StrkPric': 'STRIKE_PR',
        'OptnTp': 'OPTION_TYP', 'OpnPric': 'OPEN', 'HghPric': 'HIGH', 'LwPric': 'LOW',
        'ClsPric': 'CLOSE', 'SttlmPric': 'SETTLE_PR', 'TtlTradgVol': 'CONTRACTS',
        'OpnIntrst': 'OPEN_INT', 'ChngInOpnIntrst': 'CHG_IN_OI'
    }

    while curr <= end_date:
        if curr.weekday() >= 5:
            curr += timedelta(days=1); continue
        try:
            result = download_and_extract(curr)

            if isinstance(result, str) and result == "HOLIDAY":
                print(f"Skipping {curr}: Holiday")
            else:
                df = result.rename(columns=mapping)
                df = df.loc[:, ~df.columns.duplicated()]
                df.columns = [c.strip() for c in df.columns]

                # FIX 1: Robust Conversion for STRIKE_PR to find FUTIDX
                df['STRIKE_PR'] = pd.to_numeric(df['STRIKE_PR'], errors='coerce').fillna(0)

                # Filter NIFTY/BANKNIFTY
                df = df[df['SYMBOL'].isin(['NIFTY', 'BANKNIFTY'])].copy()

                # Logic to convert numerical Instrument IDs back to text labels
                df['INSTRUMENT'] = df.apply(lambda x: 'FUTIDX' if x['STRIKE_PR'] == 0 else 'OPTIDX', axis=1)

                # FIX 2: Reformat EXPIRY_DT to match V1 (25-Jul-2024)
                df['EXPIRY_DT'] = df['EXPIRY_DT'].apply(format_expiry_date)

                df['TIMESTAMP'] = curr.strftime('%Y-%m-%d')

                # REINDEX to the 14-column COLS list
                df = df.reindex(columns=COLS)

                batch_data.append(df)
                print(f"Processed: {curr} (Found {len(df[df['INSTRUMENT']=='FUTIDX'])} Futures)")

            if len(batch_data) >= 30:
                save_to_master(batch_data)
                batch_data = []

            time.sleep(1.0)
        except Exception as e:
            print(f"Error on {curr}: {str(e)}")

        curr += timedelta(days=1)

    if batch_data:
        save_to_master(batch_data)

def save_to_master(data_list):
    final_df = pd.concat(data_list, ignore_index=True)
    final_df = final_df[COLS]
    header = not MASTER_FILE.exists()
    final_df.to_csv(MASTER_FILE, mode='a', index=False, header=header)
    print(f"--- Checkpoint: Saved 14-column batch to {MASTER_FILE.name} ---")

if __name__ == "__main__":
    START = date(2024, 7, 1)
    END = date(2025, 12, 17)
    fetch_and_format_data(START, END)
