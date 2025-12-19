import time
from jugaad_data.nse import bhavcopy_fo_save
from datetime import date, timedelta
import pandas as pd
import os
from pathlib import Path

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
MASTER_FILE = PROCESSED_PATH / "Nifty_Historical_Derivatives.csv"

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

def fetch_and_format_data(start_date, end_date):
    curr = start_date
    batch_data = []

    column_mapping = {
        'TradDt': 'TIMESTAMP', 'BizDt': 'TIMESTAMP', 'InstrmntType': 'INSTRUMENT',
        'Symbl': 'SYMBOL', 'XpryDt': 'EXPIRY_DT', 'StrkPric': 'STRIKE_PR',
        'OptnTyp': 'OPTION_TYP', 'OpnPric': 'OPEN', 'HghPric': 'HIGH',
        'LwPric': 'LOW', 'ClsPric': 'CLOSE', 'SttlmPric': 'SETTLE_PR',
        'TtlTradgVol': 'CONTRACTS', 'TtlTrfVal': 'VAL_IN_LAKH',
        'OpnIntrst': 'OPEN_INT', 'ChngInOpnIntrst': 'CHG_IN_OI'
    }

    while curr <= end_date:
        if curr.weekday() >= 5:
            curr += timedelta(days=1)
            continue

        try:
            expected_filename = f"fo{curr.strftime('%d%b%Y').lower()}bhav.csv"
            file_path = RAW_PATH / expected_filename

            if not file_path.exists():
                file_path = bhavcopy_fo_save(curr, str(RAW_PATH))
                # Increased delay to 1.0s for safer historical fetching
                time.sleep(1.0)

            df = pd.read_csv(file_path)
            df = df.rename(columns=column_mapping)
            df.columns = [c.strip() for c in df.columns]

            df = df[df['SYMBOL'].isin(['NIFTY', 'BANKNIFTY'])]
            df['TIMESTAMP'] = curr.strftime('%Y-%m-%d')
            batch_data.append(df)
            print(f"Processed: {curr}")

            # Incremental Save to protect RAM
            if len(batch_data) >= 30:
                save_to_master(batch_data)
                batch_data = []

        except Exception as e:
            print(f"Unavailable {curr}: {e}")

        curr += timedelta(days=1)

    if batch_data:
        save_to_master(batch_data)

def save_to_master(data_list):
    final_df = pd.concat(data_list, ignore_index=True)
    cols = ['INSTRUMENT', 'SYMBOL', 'EXPIRY_DT', 'STRIKE_PR', 'OPTION_TYP',
            'OPEN', 'HIGH', 'LOW', 'CLOSE', 'SETTLE_PR', 'CONTRACTS',
            'VAL_IN_LAKH', 'OPEN_INT', 'CHG_IN_OI', 'TIMESTAMP']
    final_df = final_df[[c for c in cols if c in final_df.columns]]

    header = not MASTER_FILE.exists()
    final_df.to_csv(MASTER_FILE, mode='a', index=False, header=header)
    print(f"--- Checkpoint: Saved to {MASTER_FILE.name} ---")

if __name__ == "__main__":
    fetch_and_format_data(date(2019, 7, 1), date(2024, 6, 30))
