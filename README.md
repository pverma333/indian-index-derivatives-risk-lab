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



README snippet (add under “Strategies → Contract selection utilities”)

New module: src/strategies/contract_selectors.py

Provides deterministic utilities:

get_chain() (skip-safe empty df)

apply_strike_band() (ATM ± N strikes by count)

apply_liquidity_filters() (OFF/ABSOLUTE/PERCENTILE)

select_atm_strike() (requires both CE+PE; fallback steps)

select_otm_strike_above/below() (exact-or-nearest fallback)

dd a short section:

validate_trades_df(trades_df) enforces:

required columns (keys/contract/position/lifecycle/selection params)

leg_id unique across the entire dataframe (prevents join mixing)

side ∈ {+1,-1}, qty_lots integer and >=1

entry_date <= exit_date

tenor ∈ {WEEKLY,MONTHLY}, liquidity_mode ∈ {OFF,ABSOLUTE,PERCENTILE}

nullable: width_points, otm_distance_points, exit_k_days (unless K-days rule)


ShortStraddleStrategy.build_trades(market_df, cfg) produces a leg-level trades dataframe for Phase 2 engine consumption.

Each expiry cycle produces exactly two short option legs at the selected ATM strike (CE + PE).

Cycles are skipped (with logs) when no chain exists on entry_date or when liquidity/selection prevents finding an ATM strike with both CE and PE.

To smoke test short_Straddle on Q1 data run it in repo root - python tests/manual_run_short_straddle_q1_2025.py --tenor BOTH --full-print

BullCallSpreadStrategy emits a 2-leg call debit spread per expiry cycle:

Buy ATM CE, sell OTM CE (ATM + width_points), with deterministic fallback to the nearest available strike above ATM if the exact preferred strike is missing.

Output trades are leg-level and are later expanded by the engine into daily marks, PnL, and risk metrics.

pytest -q tests/test_bull_call_spread.py
python tests/smoke_test_bull_call_spread_q1_2025.py --tenor BOTH --full-print



README snippet

ShortStrangleStrategy emits 2 short legs per cycle:

short OTM CE at ATM + otm_distance_points (fallback nearest above ATM)

short OTM PE at ATM - otm_distance_points (fallback nearest below ATM)

pytest -q tests/test_short_strangle.py
python tests/smoke_test_short_strangle_q1_2025.py --tenor BOTH --full-print

BearPutSpreadStrategy emits 2 put legs per cycle:


Buy ATM PE
Sell OTM PE at ATM - width_points, with deterministic fallback to nearest available strike below ATM when exact strike missing.
pytest -q tests/test_bear_put_spread.py
python tests/smoke_test_bear_put_spread_q1_2025.py --tenor WEEKLY --width_points 225 --full-print



README snippet (add under “Engine / P&L”)

compute_legs_pnl(market_df, trades_df, coverage_mode="ASOF", as_of_date=None) -> (legs_pnl_df, skips_df)

coverage_mode="ASOF": never skip due solely to market window ending before exit_date; instead label leg OPEN and value to end_date_used

coverage_mode="STRICT": skip if market_max_date < exit_date with reason MARKET_WINDOW_END_BEFORE_EXIT_STRICT

Different scripts for engine pnl smoke testing -

python tests/smoke_test_engine_pnl_q1_2025.py \
  --path data/curated/derivatives_clean_Q1_2025.csv \
  --coverage-mode ASOF \
  --outdir data/derived/smoke/engine_pnl_q1_2025

python tests/smoke_test_engine_pnl_q1_2025.py \
  --path data/curated/derivatives_clean_Q1_2025.csv \
  --coverage-mode STRICT \
  --outdir data/derived/smoke/engine_pnl_q1_2025_strict

## Phase 2 Backtests Runner (ASOF/STRICT)

The Phase 2 runner produces dashboard-safe artifacts by wiring a deterministic ASOF date into the engine,
persisting `skips_df`, generating realized vs unrealized positions, and writing a run manifest derived
only from saved artifacts (not logs).

### Key concepts

- `market_max_date`:
  Computed from the symbol-filtered market dataset BEFORE applying the run window.
- `as_of_date_used`:
  The date actually used for valuation:
  - If `--as-of-date` is provided: `min(as_of_date, market_max_date)`
  - Else if `--end-date` is provided: `min(end_date, market_max_date)`
  - Else: `market_max_date`

### Coverage modes

- `ASOF` (default):
  OPEN legs are emitted when the leg’s exit is beyond `as_of_date_used`.
- `STRICT`:
  Legs that require market coverage beyond the available market window are skipped and appear in `skips_df`.

### CLI usage

```bash
python -m src.run_phase2_backtests \
  --path data/curated/derivatives_clean.parquet \
  --symbol NIFTY \
  --start-date 2025-01-01 \
  --end-date 2025-03-31 \
  --tenor BOTH \
  --strategies short_straddle,short_strangle \
  --coverage-mode ASOF \
  --outdir data/output/phase2/20260107_asof_run


README snippet (add under “Phase 2 – Testing”)

Integration smoke test:

tests/test_phase2_end_to_end_asof_vs_strict_smoke.py

Runs Phase 2 twice (ASOF + STRICT) on curated Q1-2025 dataset and asserts OPEN leg handling, deterministic STRICT skips, reconciliation, Day-0 anchor, and artifact-driven manifest counts.

pytest -q tests/test_phase2_end_to_end_asof_vs_strict_smoke.py

