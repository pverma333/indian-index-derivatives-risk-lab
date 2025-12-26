# Dev Log – Indian Index Derivatives Risk Lab

## 2025-12-16
- Created initial project folder structure.
- Next: implement fetch_eod.py skeleton and basic data pipeline.

## 2025-12-17 && 2025-12-18
- created scripts for fetching - nifty spot price daily, futures and options daily contract details, nifty volatility index daily
- Data ready in csv files for - nifty spot, futures and options, nifty vix

## 2025-12-19
- fetched the Yield and risk free rates (manual step)
- Nifty Dividend yield - https://www.nseindia.com/reports-indices-yield
- T-bill 91 days https://in.investing.com/rates-bonds/india-6-month-bond-yield-historical-data
- T-bill 182 days -https://in.investing.com/rates-bonds/india-6-month-bond-yield-historical-data
- T bill 364 days - https://in.investing.com/rates-bonds/india-1-year-bond-yield-historical-data

## 2025-12-25
- Added `src/data/map_trade_calendar.py` to generate a deterministic trade calendar for NIFTY and BANKNIFTY.
  - Futures: near/next/far expiries selected as 1st/2nd/3rd listed FUTIDX expiry >= TradeDate.
  - Options: monthly expiry classified as max expiry per (Symbol, Year-Month); others weekly (no DTE heuristics).
  - Writes parquet + csv outputs and enforces core invariants via assertions.
- Added unit tests in `tests/test_map_trade_calendar.py`.

```md
## 2025-12-26 - Market Environment & Treasury Curve Construction (#12)

### Added
- `src/data/build_market_env_treasury.py`
  - Builds `market_data.parquet` (spot + VIX + dividend yields; controlled ffill up to 5 days)
  - Builds `treasury_curve.parquet` (91D/182D/364D; percent -> decimal; calendar-day global ffill)

### Notes
- Parquet writing is stabilized via stable sort, fixed dtypes, and stripped schema metadata for repeatable outputs.
- Fixed dividend yield ingestion to handle trailing-space headers and `Div Yield%` naming variant.
- Added regression test using temp CSV mirroring raw header formatting.

### Tested
- percent -> decimal conversion
- VIX ffill limit enforcement
- stable parquet write (bit-identical within same environment)

