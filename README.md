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
