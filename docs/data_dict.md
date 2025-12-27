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

---
