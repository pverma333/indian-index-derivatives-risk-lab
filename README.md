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

