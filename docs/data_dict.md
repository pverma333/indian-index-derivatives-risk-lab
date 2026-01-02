### File: Index_Spot_Prices.csv
**Path:** `data/processed/Index_Spot_Prices.csv`

| Column | Data Type |
| --- | --- |
| Date | object |
| Index | object |
| Open | object |
| High | object |
| Low | object |
| Close | object |

---
### File: trade_calendar.parquet
**Path:** `data/processed/trade_calendar.parquet`

| Column | Data Type |
| --- | --- |
| TradeDate | datetime64[ns] |
| Symbol | object |
| Fut_Near_Expiry | datetime64[ns] |
| Fut_Next_Expiry | datetime64[ns] |
| Fut_Far_Expiry | datetime64[ns] |
| Opt_Weekly_Expiry | datetime64[ns] |
| Opt_Monthly_Expiry | datetime64[ns] |

---
### File: lot_size_map.parquet
**Path:** `data/processed/lot_size_map.parquet`

| Column | Data Type |
| --- | --- |
| symbol | object |
| start_date | datetime64[ns] |
| end_date | datetime64[ns] |
| lot_size | int64 |

---
### File: market_data.parquet
**Path:** `data/processed/market_data.parquet`

| Column | Data Type |
| --- | --- |
| date | object |
| index_name | object |
| spot_close | float64 |
| vix_close | float64 |
| div_yield | float64 |

---
### File: India_VIX_Historical.csv
**Path:** `data/processed/India_VIX_Historical.csv`

| Column | Data Type |
| --- | --- |
| Date | object |
| VIX_Close | object |

---
### File: treasury_curve.parquet
**Path:** `data/processed/treasury_curve.parquet`

| Column | Data Type |
| --- | --- |
| date | object |
| tenor | object |
| rate | float64 |

---
### File: Nifty_Historical_Derivatives.csv
**Path:** `data/processed/Nifty_Historical_Derivatives.csv`

| Column | Data Type |
| --- | --- |
| INSTRUMENT | object |
| SYMBOL | object |
| EXPIRY_DT | object |
| STRIKE_PR | object |
| OPTION_TYP | object |
| OPEN | object |
| HIGH | object |
| LOW | object |
| CLOSE | object |
| SETTLE_PR | object |
| CONTRACTS | object |
| OPEN_INT | object |
| CHG_IN_OI | object |
| TIMESTAMP | object |

---

### File: NIFTY BANK-yield-01-01-2023-to-31-12-2023.csv
**Path:** `data/raw/NIFTY BANK-yield-01-01-2023-to-31-12-2023.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: NIFTY 50-yield-01-01-2024-to-31-12-2024.csv
**Path:** `data/raw/NIFTY 50-yield-01-01-2024-to-31-12-2024.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: NIFTY BANK-yield-01-01-2025-to-19-12-2025.csv
**Path:** `data/raw/NIFTY BANK-yield-01-01-2025-to-19-12-2025.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: NIFTY 50-yield-01-01-2020-to-31-12-2020.csv
**Path:** `data/raw/NIFTY 50-yield-01-01-2020-to-31-12-2020.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: NIFTY BANK-yield-01-01-2022-to-31-12-2022.csv
**Path:** `data/raw/NIFTY BANK-yield-01-01-2022-to-31-12-2022.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: NIFTY 50-yield-01-01-2021-to-31-12-2021.csv
**Path:** `data/raw/NIFTY 50-yield-01-01-2021-to-31-12-2021.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: 3-MonBondYield.csv
**Path:** `data/raw/3-MonBondYield.csv`

| Column | Data Type |
| --- | --- |
| Date | object |
| Price | object |
| Open | object |
| High | object |
| Low | object |
| Change % | object |

---
### File: lot_size_map.csv
**Path:** `data/raw/lot_size_map.csv`

| Column | Data Type |
| --- | --- |
| symbol | object |
| start_date | object |
| end_date | object |
| lot_size | object |

---
### File: NIFTY 50-yield-01-01-2022-to-31-12-2022.csv
**Path:** `data/raw/NIFTY 50-yield-01-01-2022-to-31-12-2022.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: trade_calendar.csv
**Path:** `data/raw/trade_calendar.csv`

| Column | Data Type |
| --- | --- |
| TradeDate | object |
| Symbol | object |
| Fut_Near_Expiry | object |
| Fut_Next_Expiry | object |
| Fut_Far_Expiry | object |
| Opt_Weekly_Expiry | object |
| Opt_Monthly_Expiry | object |

---
### File: 6-MonBondYield.csv
**Path:** `data/raw/6-MonBondYield.csv`

| Column | Data Type |
| --- | --- |
| Date | object |
| Price | object |
| Open | object |
| High | object |
| Low | object |
| Change % | object |

---
### File: NIFTY BANK-yield-01-01-2021-to-31-12-2021.csv
**Path:** `data/raw/NIFTY BANK-yield-01-01-2021-to-31-12-2021.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: NIFTY 50-yield-01-01-2023-to-31-12-2023.csv
**Path:** `data/raw/NIFTY 50-yield-01-01-2023-to-31-12-2023.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: NIFTY BANK-yield-01-01-2020-to-31-12-2020.csv
**Path:** `data/raw/NIFTY BANK-yield-01-01-2020-to-31-12-2020.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: NIFTY 50-yield-01-01-2025-to-19-12-2025.csv
**Path:** `data/raw/NIFTY 50-yield-01-01-2025-to-19-12-2025.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: NIFTY BANK-yield-01-01-2024-to-31-12-2024.csv
**Path:** `data/raw/NIFTY BANK-yield-01-01-2024-to-31-12-2024.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: NIFTY 50-yield-01-07-2019-to-31-12-2019.csv
**Path:** `data/raw/NIFTY 50-yield-01-07-2019-to-31-12-2019.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |

---
### File: 1-YearBondYield.csv
**Path:** `data/raw/1-YearBondYield.csv`

| Column | Data Type |
| --- | --- |
| Date | object |
| Price | object |
| Open | object |
| High | object |
| Low | object |
| Change % | object |

---
### File: NIFTY BANK-yield-01-07-2019-to-31-12-2019.csv
**Path:** `data/raw/NIFTY BANK-yield-01-07-2019-to-31-12-2019.csv`

| Column | Data Type |
| --- | --- |
| Index Name  | object |
| Date  | object |
| P/E  | object |
| P/B  | object |
| Div Yield%  | object |


## derivatives_clean.parquet — Data Dictionary
**Path:** `data/curated/derivatives_clean.parquet`
| Column                 |           Type | Description                                                                           |
| ---------------------- | -------------: | ------------------------------------------------------------------------------------- |
| date                   | datetime64[ns] | Trading date (normalized) derived from `timestamp`                                    |
| timestamp              | datetime64[ns] | Same as `date` (normalized) retained for clarity                                      |
| symbol                 |         string | Normalized symbol (`NIFTY`, `BANKNIFTY`)                                              |
| instrument             |         string | `FUTIDX` or `OPTIDX`                                                                  |
| expiry_dt              | datetime64[ns] | Contract expiry date (normalized)                                                     |
| expiry_rank            |          int16 | Dense rank of `expiry_dt` ascending per `[date,symbol,instrument]` (1=Near)           |
| strike_pr              |        float64 | Strike; **forced to `0.0` for FUTIDX**                                                |
| option_typ             |         string | Option type (`CE`/`PE`) for OPTIDX; blank for FUTIDX                                  |
| open/high/low/close    |        float64 | OHLC from raw                                                                         |
| settle_pr              |        float64 | Settlement price from raw                                                             |
| contracts              |        float64 | Contracts/volume from raw (kept float64 for uniformity)                               |
| open_int               |        float64 | Open interest                                                                         |
| chg_in_oi              |        float64 | Change in OI                                                                          |
| spot_close             |        float64 | Spot close joined from `market_data.parquet`                                          |
| vix_close              |        float64 | India VIX close joined from `market_data.parquet`                                     |
| div_yield              |        float64 | Dividend yield joined from `market_data.parquet` (if present)                         |
| lot_size               |          int64 | Lot size resolved from `lot_size_map.parquet` by effective date range                 |
| rate_91d               |        float64 | 91-day treasury rate (decimal)                                                        |
| rate_182d              |        float64 | 182-day treasury rate (decimal)                                                       |
| rate_364d              |        float64 | 364-day treasury rate (decimal)                                                       |
| cal_days_to_expiry     |          int32 | Calendar days between `date` and `expiry_dt`                                          |
| trading_days_to_expiry |          int32 | Count of trading days in `(date, expiry_dt]` from `trade_calendar.parquet`            |
| is_opt_weekly_expiry   |           bool | `expiry_dt == Opt_Weekly_Expiry` for that `[date,symbol]`                             |
| is_opt_monthly_expiry  |           bool | `expiry_dt == Opt_Monthly_Expiry` for that `[date,symbol]`                            |
| moneyness              |        float64 | OPTIDX only: Calls=`spot_close-strike_pr`, Puts=`strike_pr-spot_close` (ITM positive) |

#### Phase 1 - Audit
| Table                   | Column                |       Type | Meaning                                                                                |
| ----------------------- | --------------------- | ---------: | -------------------------------------------------------------------------------------- |
| monthly_expiry_per_year | monthly_expiry_events |        int | Count of rows with `is_opt_monthly_expiry==True` for that `(symbol, year)`             |
| near_futures_basis      | basis                 |      float | `settle_pr - spot_close` for near futures (`expiry_rank==1`)                           |
| near_futures_basis      | basis_abs_frac        |      float | `abs(basis) / spot_close`                                                              |
| near_futures_basis      | is_basis_spike        |       bool | True if `basis_abs_frac > 0.02`                                                        |
| lot_size_by_year        | unique_lot_sizes      | array<int> | Sorted unique lot sizes observed in that `(symbol, year)`                              |
| strike_density_samples  | inferred_step         |      float | Most common positive difference between sorted unique strikes (per date/symbol/expiry) |
| strike_density_samples  | missing_in_atm_window |        int | Missing strikes count in `[ATM ± 10 * step]` window                                    |


## startegy_mtm_pnl.parquet — Data Dictionary
**Path:** `data/curated/startegy_mtm_pnl.parquet`
| Column          |   Type | Nullable | Description                                                        |
| --------------- | -----: | :------: | ------------------------------------------------------------------ |
| trade_id        | string |    No    | Unique identifier per trade lifecycle                              |
| date            |   date |    No    | Trading date for MTM                                               |
| symbol          | string |    No    | Underlying (e.g., NIFTY/BANKNIFTY)                                 |
| strike_pr       |  float |    No    | Option strike (0 for futures if used)                              |
| option_typ      | string |    Yes   | CE/PE or null for futures                                          |
| entry_price     |  float |    No    | Entry price used for Day-0 MTM (override or entry-day settle_used) |
| settle_pr       |  float |    No    | Settlement used for MTM (settle_pr else close fallback)            |
| daily_pnl_rupee |  float |    No    | MTM P&L for the day, already multiplied by lot_size                |
| cum_pnl_rupee   |  float |    No    | Running sum of daily_pnl_rupee within trade_id                     |



Important date / field consideration

- nifty_historical_derivatives -
----> expiry_dt  = dd-mmm-yyyy
----> timestamp = yyyy-mm-dd

- trade_Calendar.parquet -
----> all columns datetimstamp in format - yyyy-mm-dd 00:00:00

-lot_size.parquet -
----> all date columns datetimestamp in format - yyyy-mm-dd 00:00:00

- market_data.parquet -
----> date - yyyy-mm-dd format
----> div yeild - percentage converted to decimal so # in decimals

- treasury_data.parquet -
-----> date - yyyy-mm-dd format
-----> rate - percentage converted to decimal so # in decimals


| Field                       |   Type | Nullable | Description                                                     |
| --------------------------- | -----: | :------: | --------------------------------------------------------------- |
| strategy_name               | string |    no    | Strategy key (`short_straddle`, etc.)                           |
| tenor                       | string |    no    | `WEEKLY` or `MONTHLY` (single-tenor resolution)                 |
| param_version               | string |    no    | Registry version (e.g., `phase2_v1`)                            |
| qty_lots                    |    int |    no    | Lots per leg                                                    |
| strike_band_n_weekly        |    int |    no    | Weekly strike band default                                      |
| strike_band_n_monthly       |    int |    no    | Monthly strike band default                                     |
| strike_band_n               |    int |    no    | Derived by tenor from weekly/monthly                            |
| max_atm_search_steps        |    int |    no    | ATM selection fallback steps                                    |
| liquidity_mode              | string |    no    | `OFF` / `ABSOLUTE` / `PERCENTILE`                               |
| min_contracts               |    int |    no    | Used when `ABSOLUTE`                                            |
| min_open_int                |    int |    no    | Used when `ABSOLUTE`                                            |
| liquidity_percentile        |    int |    no    | Used when `PERCENTILE` (0–100)                                  |
| exit_rule                   | string |    no    | Phase 2 default `EXPIRY` (stub supports `K_DAYS_BEFORE_EXPIRY`) |
| exit_k_days                 |    int |    yes   | Required if `exit_rule=K_DAYS_BEFORE_EXPIRY`                    |
| fees_bps                    |  float |    no    | Stored for later fee application                                |
| fixed_fee_per_lot           |  float |    no    | Stored for later fee application                                |
| width_points                |    int |    yes   | Spread width (spreads only)                                     |
| otm_distance_points         |    int |    yes   | Strangle distance (strangle only)                               |
| otm_distance_points_weekly  |    int |    yes   | Present for short_strangle (input to derived)                   |
| otm_distance_points_monthly |    int |    yes   | Present for short_strangle (input to derived)                   |
