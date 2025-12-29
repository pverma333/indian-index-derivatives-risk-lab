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


### 2025-12-27
## Phase 1.2 – Curated “Golden Source” derivatives layer

- Added curated derivatives ETL: `src/data/build_derivatives_curated.py`
  - Reads `Nifty_Historical_Derivatives.csv` in chunks, filters to FUTIDX/OPTIDX and NIFTY/BANKNIFTY
  - Normalizes schema (snake_case), parses mixed date formats without dayfirst=True, casts numerics to float64
  - Enriches with spot/VIX (market_data), lot_size (range join), treasury curve (pivot to 91/182/364d, decimals), and trade_calendar expiry context
  - Computes `cal_days_to_expiry`, `trading_days_to_expiry`, `expiry_rank`, and `moneyness`
  - Writes `data/curated/derivatives_clean.parquet`
- Tests:
  - Added `tests/test_build_derivatives_curated.py` end-to-end fixture covering joins, strike=0 for futures, rates-as-decimals, expiry_rank, moneyness, and TTE invariants

## 2025-12-28
- Added Phase 1 derivatives_clean audit notebook: `notebooks/01_data_validation.ipynb`
- Implemented reusable audit functions in `src/data_validation/derivatives_audit.py`
- Added pytest coverage for key invariants (uniqueness, expiry_rank, moneyness, basis)
- Validates enrichment joins (VIX variance, rates scaling) and lot size truth table
- Adds strike density sampling around ATM for 5 deterministic dates (seed=42)

Tested:
- `pytest -q`
- Notebook runs end-to-end on local parquet
Next:
- Align lot size truth table with finalized project mapping if it differs from defaults.



---
## 2025-12-29 — Vectorized Rupee MTM P&L Engine + Contract Lifecycle Guards

### Added
- `src/strategies/engine_pnl.py`
  - Implemented `BaseStrategy` as the foundational parent class for strategies.
  - Implemented `load_market_data()` with required-column validation and `expiry_rank == 1` filtering.
  - Implemented `identify_entry_days()`:
    - Entry day = first trading day strictly after an `is_opt_monthly_expiry == True` day (per `symbol`).
  - Implemented vectorized MTM Rupee P&L engine `compute_mtm_pnl_rupee()`:
    - Entry-day P&L uses `(settle_t - entry_price) * lot_size`
    - Holding-day P&L uses `(settle_t - settle_{t-1}) * lot_size`
    - `position_sign` supports long (+1) / short (-1)

- Safety guards and bookkeeping:
  - Settlement fallback: if `settle_pr` is `NaN` or `0`, fallback to `close` with warnings and sample rows.
  - Lot size integrity: raises `DataIntegrityError` if `lot_size` changes within a `trade_id`.
  - Expiry exit: enforces that each trade has a market row on `expiry_dt` to force-close using final exchange settlement.
  - Output includes `daily_pnl_rupee` and `cum_pnl_rupee` per trade lifecycle.

- `tests/test_engine_pnl.py`
  - Deterministic unit test confirming:
    - Long MTM path matches expected daily/cumulative P&L
    - Short position generates positive P&L when settlement decreases
    - Lot size change mid-trade raises `DataIntegrityError`

### Why
- Establish a reusable, strategy-agnostic engine for MTM P&L with robust lifecycle handling.
- Ensure correctness and data safety (settlement gaps, bad joins, and expiry closure).

### Tested
- `pytest -q`
- Unit tests cover MTM formulas, sign behavior, and integrity checks.

### Next
- Add a small utility method to persist outputs with consistent naming (optional).
- Add integration-style test against a small real-data sample slice from `derivatives_clean.parquet`.
- Extend blotter schema support for multi-leg strategies (spreads) while keeping vectorization.

## 2025-12-29
- Added ShortStraddleStrategy (monthly ATM short straddle) with:
  - ATM strike resolver + strike-interval rounding
  - Strict CE/PE existence + liquidity validation on entry
  - Leg synchronization guard (drops whole trade if calendars diverge)
  - Expiry intrinsic settlement validation vs spot settlement
  - Strategy-level aggregation output schema

Tested:
- pytest: liquidity abort, large move loss, intrinsic validation failure

How to run short straddly
python -c "
import logging
from src.strategies.short_straddle import ShortStraddleStrategy, ShortStraddleStrategyConfig

logging.basicConfig(level=logging.INFO)

cfg = ShortStraddleStrategyConfig(
    input_parquet_path='data>curated>nifty_options.parquet',
    symbol='NIFTY',
    strike_interval=50,
)
s = ShortStraddleStrategy(cfg)
out = s.run()
print(out.head())
print(out.tail())
print(out.columns.tolist())
"
