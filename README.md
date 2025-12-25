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
