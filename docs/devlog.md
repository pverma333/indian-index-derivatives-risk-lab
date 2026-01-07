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


## 2025-02-01
---
DEVLOG entry

Added Phase 2 parameter registry (phase2_params.py) as the single source of truth for defaults/overrides/validations.

Added deterministic effective param resolution by tenor (weekly/monthly).

Added strict validation with ConfigError(errors=[...]) to fail early and clearly.

Added unit tests covering defaults, overrides, validations, JSON-serializability.

Tested: pytest -q.

Added src/validation/market_df_validator.py Phase 2 gatekeeper.

Added DataIntegrityError with standardized violation preview.

Added unit tests for missing columns, option typ domain, FUTIDX invariants, and index_open_price completeness.

Added: deterministic marking logic for settle_used + price_method with OPTIDX expiry intrinsic override.

Why: ensures correct expiry-day valuation and consistent entry anchoring for downstream settle_prev_used logic.

Tested: pytest -q (unit tests for non-expiry, CE/PE intrinsic, FUTIDX unchanged, and missing-input integrity errors).

Next: wire into runner ordering: validate market → compute settle_used → enrich entry_price → engine MTM.


Implemented build_expiry_cycles() in src/strategies/expiry_selectors.py
Uses only dataset flags (no weekday assumptions)
Deterministic next-trading-day entry selection via is_trading_day
Added drop auditing via warning logs (MISSING_ENTRY_DATE) and optional returned dropped_df
Added pytest coverage for monthly, weekly, and missing-entry drop cases


Added src/strategies/expiry_selectors.py:build_expiry_cycles() implementing Phase 2 expiry-cycle construction using dataset flags only.
Added pytest coverage for monthly, weekly, and missing entry-date drop behavior (including log assertions).
Tested with pytest -q tests/test_expiry_selectors.py.
Smoke test with python tests/smoke_test_expiry_cycles_q1_2025.py --symbol NIFTY --tenor BOTH --max_print 10



DEVLOG entry

Added contract_selectors.py implementing chain extraction, strike banding, liquidity filtering (3 modes), and robust ATM/OTM strike selection with deterministic tie-breaks.

Added unit tests covering required behaviors including fallback logic and percentile thresholds.

Tested: pytest -q tests/test_contract_selectors.py

Added src/strategies/trade_schema.py with TradeSchemaError + validate_trades_df.

Added tests covering missing columns, duplicate leg_id fail-fast, invalid side, and bad tenor.

Tested via pytest -q.

Added deterministic short straddle trade generation:

Uses Phase 2 cycle builder and contract selectors for chain extraction, strike banding, liquidity filtering, and ATM selection.

Emits schema-validator-compatible selection parameter columns, plus strike_interval_used.

Added 3 unit tests covering: 2-legs-per-trade, unique leg ids, and skip-on-empty-chain.


Added bull call spread strategy with:

chain → strike band → liquidity filter → ATM selection

OTM call strike selection with deterministic fallback above ATM

stored width_points and strike_interval_used

tests for leg correctness and fallback behavior

Added short strangle strategy with deterministic OTM fallback on both sides.

Added tests for 2-leg emission and fallback tie-breaking.

Added bear put spread strategy + tests validating legs and fallback-below behavior.


Added core engine PnL computation with:

Day-0 settle_prev_used anchor

MTM PnL

Gap-risk proxy (intrinsic-open for options; index-open for futures)

Skip-on-missing-market-rows behavior with logged context

Unit tests for anchoring + gap proxy formulas

Implemented ASOF/STRICT coverage handling in P&L engine:

Engine computes market_max_date, as_of_date_used, and end_date_used per leg.

ASOF emits partial legs as OPEN without skipping solely due to insufficient future coverage.

STRICT skips legs when market_max_date < exit_date.

Added deterministic skips_df output.

Added unit tests for day-0 anchoring, gap proxies, OPEN/CLOSED labeling, capped expected-dates logic, STRICT skip rule, and skip schema.


### DEVLOG entry (append to your `DEVLOG.md`)

```md
## 2026-01-07 — Phase 2 runner dashboard-safe ASOF wiring + positions + manifest (Ticket 10)

### Added
- Deterministic ASOF wiring in Phase 2 runner:
  - compute `market_max_date` from symbol-filtered market_df before date filtering
  - compute `as_of_date_used` bounded by `market_max_date`
  - filter market_df to `[start_date, as_of_date_used]`
  - pass `coverage_mode` and `as_of_date_used` into `compute_legs_pnl(...)`
- New artifact: `positions_df.parquet`
  - leg-level positions summary with realized vs unrealized P&L split using leg `status`
- New artifact: `run_manifest.json`
  - artifact-derived run summary with counts and skips_by_reason computed from `skips_df`
  - reconciliation sanity totals

### Changed
- Aggregations now preserve ASOF metadata (`market_max_date`, `as_of_date_used`, `coverage_mode`)
  and include counts (`n_open_legs`, `n_closed_legs`, `n_skipped_legs`) derived from artifacts.

### Notes
- Manifest and counts are derived from saved artifacts only (no log parsing).
- ASOF runs complete even when user end_date exceeds market coverage by capping to `market_max_date`.



DEVLOG entry

Added Phase 2 end-to-end smoke test to prevent regressions:

ASOF must emit OPEN legs when end_date < exit_date

STRICT must skip incomplete lifecycle legs with reason MARKET_WINDOW_END_BEFORE_EXIT_STRICT

Manifest counts validated against legs_pnl_df / skips_df artifacts (no log-derived counts)
