# indian-index-derivatives-risk-lab
EOD-based Indian index derivatives risk lab: options strategies, Greeks, portfolio VaR/CVaR.

### Trade calendar

Builds a deterministic trade calendar from `data/processed/Nifty_Historical_Derivatives.csv`.

Outputs:
- `data/processed/trade_calendar.parquet`
- `data/raw/trade_calendar.csv`

Run:
```bash
python -m src.data.map_trade_calendar


## Market Environment & Treasury Curve Construction

Builds:
- `data/processed/market_data.parquet`
  - `date,index_name,spot_close,vix_close,div_yield`
  - spot is base; VIX and div-yield are forward-filled up to 5 days (no clipping of VIX spikes)
  - Note: raw CSV headers may contain trailing spaces (e.g., `Date `, `Div Yield% `). The pipeline strips header whitespace before schema inference.

- `data/processed/treasury_curve.parquet`
  - `date,tenor,rate`
  - 91D/182D/364D bond yields are percent-quoted and converted to decimals; rates are forward-filled across calendar days
Run:
```bash
python src/data/build_market_env_treasury.py

### Curated Derivatives (Golden Source)

Build the curated derivatives dataset (joined spot/VIX, lot sizes, treasury rates, expiry context, and TTE metrics):

```bash
python -m src.data.build_derivatives_curated

## Data Validation (Phase 1 Audit)

Run the audit notebook:

1. Ensure the curated dataset exists:
   - `data/processed/derivatives_clean.parquet`

2. Open and run:
   - `notebooks/01_data_validation.ipynb`

The notebook fails fast with AssertionError if:
- duplicates exist by unique contract key (holiday drift)
- `cal_days_to_expiry` is negative anywhere
- treasury rates are not in decimal form (e.g., 7.0 instead of 0.07)
- VIX join is constant/zero per symbol
- FUTIDX has non-zero strike_pr
- moneyness polarity is inconsistent with CE/PE definitions
- `expiry_rank==1` is not the nearest expiry
- monthly expiry events are not exactly 12 per year per symbol
- lot sizes violate the project truth table (including NIFTY 2025)


# Vectorized Rupee MTM P&L Engine

This repo implements a foundational **vectorized Mark-to-Market (MTM) P&L engine** for Indian derivatives data. The core class is `BaseStrategy` in `src/strategies/engine_pnl.py`, intended to be subclassed by specific strategies.

## What it does

Given:
- Curated derivatives market data (`derivatives_clean.parquet`)
- A strategy-produced **trade blotter** (one row per trade)

The engine expands each trade across its contract life (from `entry_date` to `expiry_dt`, inclusive) and computes **daily** and **cumulative** Rupee MTM P&L using exchange settlement prices.

### MTM logic (single leg)

- **Entry day (Day 0):**
  `daily_pnl_rupee = position_sign * (settle_t - entry_price) * lot_size_t`

- **Holding days (Day t):**
  `daily_pnl_rupee = position_sign * (settle_t - settle_{t-1}) * lot_size_t`

Where:
- `position_sign` = `+1` (Long), `-1` (Short)
- `lot_size` is applied **row-level** to preserve accuracy during historical transitions

## Files

- `src/strategies/engine_pnl.py`
  - `BaseStrategyConfig`
  - `DataIntegrityError`
  - `TradeBlotterSchema`
  - `BaseStrategy` with:
    - `load_market_data()`
    - `identify_entry_days()`
    - `compute_mtm_pnl_rupee()`

- `tests/test_engine_pnl.py`
  - pytest unit tests for MTM logic and integrity guards

## Input data

- Parquet path (provided via config):
  - `data>curated>derivatives_clean.parquet` (also supports normal `data/curated/...` paths)

Expected market columns (subset):
- `date`, `symbol`, `instrument`, `expiry_dt`, `strike_pr`, `option_typ`
- `close`, `settle_pr`, `lot_size`
- `expiry_rank`, `is_trading_day`, `is_opt_monthly_expiry`

## Trade blotter format

A strategy must create a DataFrame with **one row per trade** containing at least:

- `trade_id`
- `symbol`, `instrument`, `expiry_dt`, `strike_pr`, `option_typ`
- `entry_date`
- `position_sign` (`+1` long / `-1` short)

Optional:
- `entry_price` (if omitted, engine uses `settle_pr` on `entry_date`)

## Output

`compute_mtm_pnl_rupee(...)` returns a **pandas DataFrame** (it does not write files automatically).

Columns:
- `date`, `symbol`, `strike_pr`, `option_typ`
- `entry_price` (resolved)
- `settle_pr` (the value actually used; may be fallback-to-close)
- `daily_pnl_rupee`
- `cum_pnl_rupee`
- `trade_id`

**Grain:** one row per `trade_id` per `date`.

## Safety guards

- **Missing settlement:** if `settle_pr` is `NaN` or `0`, fallback to `close` and log a warning (includes examples).
- **Lot size integrity:** if `lot_size` changes within the same `trade_id`, raise `DataIntegrityError`.
- **Expiry exit:** requires a market row on `expiry_dt` for each trade (force-close using exchange-provided final settlement).

## Example usage

```python
import pandas as pd
from src.strategies.engine_pnl import BaseStrategy, BaseStrategyConfig

class MyStrategy(BaseStrategy):
    def build_trades(self, market_df: pd.DataFrame, entry_days: pd.DataFrame) -> pd.DataFrame:
        # build your blotter here (one row per trade)
        raise NotImplementedError

engine = MyStrategy(BaseStrategyConfig(input_parquet_path="data>curated>derivatives_clean.parquet"))

mkt = engine.load_market_data()
entry_days = engine.identify_entry_days(mkt)

trades = engine.build_trades(mkt, entry_days)
pnl_df = engine.compute_mtm_pnl_rupee(mkt, trades)

# Save if desired
pnl_df.to_parquet("data/outputs/mtm_pnl.parquet", index=False)
