── indian-index-derivatives-risk-lab
│   │       ├── data
│   │       │   ├── curated
│   │       │   │   ├── derivatives_clean.parquet
│   │       │   ├── processed
│   │       │   │   ├── Index_Spot_Prices.csv
│   │       │   │   ├── India_VIX_Historical.csv
│   │       │   │   ├── lot_size_map.parquet
│   │       │   │   ├── market_data.parquet
│   │       │   │   ├── Nifty_Historical_Derivatives.csv
│   │       │   │   ├── trade_calendar.parquet
│   │       │   │   └── treasury_curve.parquet
│   │       │   └── raw
│   │       │       ├── 1-YearBondYield.csv
│   │       │       ├── 3-MonBondYield.csv
│   │       │       ├── 6-MonBondYield.csv
│   │       │       ├── lot_size_map.csv
│   │       │       ├── NIFTY 50-yield-01-01-2020-to-31-12-2020.csv
│   │       │       ├── NIFTY 50-yield-01-01-2021-to-31-12-2021.csv
│   │       │       ├── NIFTY 50-yield-01-01-2022-to-31-12-2022.csv
│   │       │       ├── NIFTY 50-yield-01-01-2023-to-31-12-2023.csv
│   │       │       ├── NIFTY 50-yield-01-01-2024-to-31-12-2024.csv
│   │       │       ├── NIFTY 50-yield-01-01-2025-to-19-12-2025.csv
│   │       │       ├── NIFTY 50-yield-01-07-2019-to-31-12-2019.csv
│   │       │       ├── NIFTY BANK-yield-01-01-2020-to-31-12-2020.csv
│   │       │       ├── NIFTY BANK-yield-01-01-2021-to-31-12-2021.csv
│   │       │       ├── NIFTY BANK-yield-01-01-2022-to-31-12-2022.csv
│   │       │       ├── NIFTY BANK-yield-01-01-2023-to-31-12-2023.csv
│   │       │       ├── NIFTY BANK-yield-01-01-2024-to-31-12-2024.csv
│   │       │       ├── NIFTY BANK-yield-01-01-2025-to-19-12-2025.csv
│   │       │       ├── NIFTY BANK-yield-01-07-2019-to-31-12-2019.csv
│   │       │       └── trade_calendar.csv
│   │       ├── notebooks
│   │       │       ├── phase_1_data_validation.ipynb
│   │       ├── docs
│   │       │   └──
│   │       │       ├── devlog.md
│   │       │       └── repo_structure.md
│   │       │       └── data_dict.md
│   │       ├── README.md
│   │       ├── schemas
│   │       │   ├── curated
│   │       │   │   ├── schema.derivatives_clean.json
│   │       │   ├── processed
│   │       │   └── raw
│   │       ├── src
│   │       │   ├── data
│   │       │   │   ├── __pycache__
│   │       │   │   ├── build_market_env_treasury.cpython-312.pyc
│   │       │   │   ├── build_market_env_treasury.py
│   │       │   │   ├── config_contract_lot.py
│   │       │   │   ├── fetch_eod_niftyspot_vixclose.py
│   │       │   │   ├── fetch_eod_v1.py
│   │       │   │   ├── fetch_eod_v2.py
│   │       │   │   └── map_trade_calendar.py
│   │       │   │   └── phase_1_data_validation_utils.py
│   │       │   ├── greeks
│   │       │   ├── portfolio
│   │       │   ├── risk
│   │       │   └── strategies
│   │       └── tests
│   │           ├── __pycache__
│   │           │   └── test_build_market_env_treasury.cpython-312-pytest-7.4.4.pyc
│   │           ├── test_build_market_env_treasury.py
│   │           ├── test_map_trade_calendar.py
│   │           ├── test_phase_1_data_validation_utils.py
│   │           └── test_outputs
│   │               ├── mock_derivatives.csv
│   │               ├── trade_calendar.csv
│   │               └── trade_calendar.parquet
