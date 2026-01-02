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


# Run unit tests
pytest -q

# (Example usage in runner code)
python -c "
from src.config.phase2_params import resolve_effective_strategy_params
print(resolve_effective_strategy_params('short_strangle','WEEKLY',{'qty_lots':2}))
"
README snippet (how to use)

Phase 2 Parameter Registry

Defaults and validations live in src/config/phase2_params.py.

Runner should:

run_cfg = get_phase2_default_run_config() then merge CLI overrides.

For each (strategy, tenor) call resolve_effective_strategy_params(strategy, tenor, user_overrides) and pass the resulting dict into the strategy.

Write raw overrides + effective configs into run_manifest.json.

validate_market_df() enforces Phase 2 market contract (required columns, option type domain, FUTIDX/OPTIDX invariants, index_open_price non-null in window, boolean expiry flags).

python -c "import pandas as pd; from validation.market_df_validator import validate_market_df; df=pd.read_csv('data/curated/derivatives_clean_Q1_2025.csv'); validate_market_df(df)"

Use it immediately after loading derivatives_clean.parquet before any strategy backtest.

README snippet (update)

Replace references to src/engine/marking.py with:

README snippet (add under “Engine / Marking”)

compute_settle_used(market_df) enriches the market dataset with:

settle_used: deterministic EOD marking used for MTM and entry anchoring

price_method: "SETTLE_PR" by default; "INTRINSIC_ON_EXPIRY" for OPTIDX on expiry day (cal_days_to_expiry == 0)

This is required so enrichment can set entry_price = settle_used(entry_date) for the Day-0 settle_prev_used anchor in the engine.

Run for smoke test on integrated data - pytest -q -s tests/test_settle_used_q1_2025.py

Add a section:

Expiry cycle construction (Phase 2)
Cycles are built deterministically from dataset flags:

WEEKLY cycles use is_opt_weekly_expiry == True

MONTHLY cycles use is_opt_monthly_expiry == True

entry_date is the next is_trading_day == True date strictly after expiry_dt

Cycles with no available entry_date are dropped and logged as MISSING_ENTRY_DATE

build_expiry_cycles(market_df, symbol, tenor) creates expiry cycles from dataset flags:

WEEKLY uses is_opt_weekly_expiry == True

MONTHLY uses is_opt_monthly_expiry == True

entry_date is the next trading day strictly after expiry_dt where is_trading_day == True

cycles with missing entry_date are dropped and logged as MISSING_ENTRY_DATE
python -m pytest -q tests/test_expiry_selectors.py
