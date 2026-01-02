from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

PHASE2_PARAM_VERSION = "phase2_v1"

ALLOWED_TENORS_RUN: Set[str] = {"WEEKLY", "MONTHLY", "BOTH"}
ALLOWED_TENORS_SINGLE: Set[str] = {"WEEKLY", "MONTHLY"}
ALLOWED_SYMBOLS: Set[str] = {"NIFTY","BANKNIFTY"}  # Phase 2 default; can expand to BANKNIFTY later

SUPPORTED_STRATEGIES: Set[str] = {
    "short_straddle",
    "bull_call_spread",
    "short_strangle",
    "bear_put_spread",
}

LIQUIDITY_MODES: Set[str] = {"OFF", "ABSOLUTE", "PERCENTILE"}
EXIT_RULES: Set[str] = {"EXPIRY", "K_DAYS_BEFORE_EXPIRY"}  # stub for later


class ConfigError(ValueError):
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


def _normalize_upper(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return x.strip().upper()


def _parse_iso_date(s: Optional[str], field_name: str, errors: List[str]) -> Optional[date]:
    if s is None:
        return None
    try:
        # Accept YYYY-MM-DD only (explicit, deterministic)
        parts = s.split("-")
        if len(parts) != 3:
            raise ValueError("Expected YYYY-MM-DD")
        y, m, d = (int(parts[0]), int(parts[1]), int(parts[2]))
        return date(y, m, d)
    except Exception as e:
        errors.append(f"{field_name}: invalid date '{s}' ({e})")
        return None


@dataclass(frozen=True)
class Phase2RunConfig:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    symbol: str = "NIFTY"
    tenor: str = "BOTH"  # WEEKLY|MONTHLY|BOTH
    strategies: List[str] = dataclasses.field(
        default_factory=lambda: [
            "short_straddle",
            "bull_call_spread",
            "short_strangle",
            "bear_put_spread",
        ]
    )
    write_outputs: bool = True
    run_id: Optional[str] = None
    output_root: str = "data/derived/phase2"


@dataclass(frozen=True)
class Phase2StrategyCommonConfig:
    qty_lots: int = 1
    strike_band_n_weekly: int = 10
    strike_band_n_monthly: int = 15
    max_atm_search_steps: int = 4

    liquidity_mode: str = "OFF"  # OFF|ABSOLUTE|PERCENTILE
    min_contracts: int = 1
    min_open_int: int = 1
    liquidity_percentile: int = 50  # used only in PERCENTILE

    exit_rule: str = "EXPIRY"  # Phase 2 only EXPIRY (stub for later)
    exit_k_days: Optional[int] = None

    fees_bps: float = 0.0  # store; apply later
    fixed_fee_per_lot: float = 0.0  # store; apply later

    def strike_band_n(self, tenor: str) -> int:
        t = _normalize_upper(tenor)
        if t == "WEEKLY":
            return self.strike_band_n_weekly
        if t == "MONTHLY":
            return self.strike_band_n_monthly
        raise ConfigError(f"strike_band_n: tenor must be WEEKLY or MONTHLY, got '{tenor}'")


PHASE2_DEFAULTS: Dict[str, Any] = {
    "run": Phase2RunConfig(),
    "common": Phase2StrategyCommonConfig(),
    "short_straddle": {},
    "bull_call_spread": {"width_points": 200},
    "short_strangle": {"otm_distance_points_weekly": 300, "otm_distance_points_monthly": 400},
    "bear_put_spread": {"width_points": 200},
}

# These are the selection params we want strategies to stamp into trades_df.
# Keeping this list here avoids drift and makes "missing keys" fail early.
REQUIRED_SELECTION_FIELDS: Tuple[str, ...] = (
    "qty_lots",
    "strike_band_n",
    "max_atm_search_steps",
    "liquidity_mode",
    "min_contracts",
    "min_open_int",
    "liquidity_percentile",
    "exit_rule",
    "exit_k_days",
    "fees_bps",
    "fixed_fee_per_lot",
    # strategy-specific (nullable for non-applicable):
    "width_points",
    "otm_distance_points",
)


def merge_overrides(
    base_config: Union[Mapping[str, Any], Any],
    overrides: Optional[Mapping[str, Any]],
    *,
    strict: bool = True,
) -> Union[Dict[str, Any], Any]:
    """
    Merge `overrides` into `base_config`.
    - If base_config is a dataclass instance: returns a new instance (dataclasses.replace).
    - If base_config is a mapping: returns a new dict.

    strict=True => unknown keys raise ConfigError.
    """
    if not overrides:
        return base_config

    if dataclasses.is_dataclass(base_config):
        field_names = {f.name for f in dataclasses.fields(base_config)}
        unknown = sorted(set(overrides.keys()) - field_names)
        if strict and unknown:
            raise ConfigError("Unknown override keys for dataclass", errors=unknown)

        filtered = {k: v for k, v in overrides.items() if k in field_names}
        return replace(base_config, **filtered)

    # mapping / dict
    base_dict = dict(base_config)
    if strict:
        unknown = sorted(set(overrides.keys()) - set(base_dict.keys()))
        if unknown:
            raise ConfigError("Unknown override keys for dict config", errors=unknown)
    base_dict.update(dict(overrides))
    return base_dict


def get_phase2_default_run_config() -> Phase2RunConfig:
    return PHASE2_DEFAULTS["run"]


def get_phase2_default_strategy_config(strategy_name: str) -> Dict[str, Any]:
    s = strategy_name.strip().lower()
    if s not in SUPPORTED_STRATEGIES:
        raise ConfigError(f"Unsupported strategy '{strategy_name}'")
    # Return only strategy-specific defaults (common is merged elsewhere)
    return dict(PHASE2_DEFAULTS[s])


def _allowed_strategy_param_keys(strategy_name: str) -> Set[str]:
    common_keys = {f.name for f in dataclasses.fields(Phase2StrategyCommonConfig)}
    strat_keys = set(get_phase2_default_strategy_config(strategy_name).keys())
    # Strategy-specific keys that are valid even if not in defaults dict for that strategy:
    if strategy_name == "short_strangle":
        strat_keys |= {"otm_distance_points_weekly", "otm_distance_points_monthly"}
    if strategy_name in {"bull_call_spread", "bear_put_spread"}:
        strat_keys |= {"width_points"}
    return common_keys | strat_keys


def resolve_effective_strategy_params(
    strategy_name: str, tenor: str, user_overrides: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """
    Build merged params dict for one strategy + one tenor.
    - Precedence: user_overrides > defaults
    - Derived after merge: strike_band_n, otm_distance_points
    - Includes nullable strategy-specific fields for consistency.
    """
    s = strategy_name.strip().lower()
    t = _normalize_upper(tenor)

    if s not in SUPPORTED_STRATEGIES:
        raise ConfigError(f"Unsupported strategy '{strategy_name}'")

    if t not in ALLOWED_TENORS_SINGLE:
        raise ConfigError(f"resolve_effective_strategy_params requires WEEKLY or MONTHLY, got '{tenor}'")

    overrides = dict(user_overrides or {})
    allowed_keys = _allowed_strategy_param_keys(s)
    unknown = sorted(set(overrides.keys()) - allowed_keys)
    if unknown:
        raise ConfigError(f"Unknown strategy override keys for '{s}'", errors=unknown)

    common_cfg: Phase2StrategyCommonConfig = PHASE2_DEFAULTS["common"]
    params: Dict[str, Any] = dataclasses.asdict(common_cfg)

    # Merge strategy-specific defaults
    params.update(get_phase2_default_strategy_config(s))

    # Apply user overrides
    params.update(overrides)

    # Normalize enums (keeps behavior deterministic)
    params["liquidity_mode"] = _normalize_upper(str(params["liquidity_mode"]))
    params["exit_rule"] = _normalize_upper(str(params["exit_rule"]))

    # Derived: strike_band_n by tenor
    sb_weekly = int(params["strike_band_n_weekly"])
    sb_monthly = int(params["strike_band_n_monthly"])
    params["strike_band_n"] = sb_weekly if t == "WEEKLY" else sb_monthly

    # Derived: otm distance for strangle
    if s == "short_strangle":
        d_weekly = int(params["otm_distance_points_weekly"])
        d_monthly = int(params["otm_distance_points_monthly"])
        params["otm_distance_points"] = d_weekly if t == "WEEKLY" else d_monthly
    else:
        params["otm_distance_points"] = None

    # Keep width_points present & nullable for non-spreads
    if s in {"bull_call_spread", "bear_put_spread"}:
        params["width_points"] = int(params["width_points"])
    else:
        params["width_points"] = None

    # Attach for manifest/debug friendliness (safe & deterministic)
    params["strategy_name"] = s
    params["tenor"] = t
    params["param_version"] = PHASE2_PARAM_VERSION

    validate_strategy_params(s, params, t)
    return params


def validate_run_config(run_cfg: Phase2RunConfig) -> None:
    errors: List[str] = []

    tenor = _normalize_upper(run_cfg.tenor)
    if tenor not in ALLOWED_TENORS_RUN:
        errors.append(f"tenor must be one of {sorted(ALLOWED_TENORS_RUN)}, got '{run_cfg.tenor}'")

    symbol = run_cfg.symbol.strip().upper()
    if symbol not in ALLOWED_SYMBOLS:
        errors.append(f"symbol must be one of {sorted(ALLOWED_SYMBOLS)}, got '{run_cfg.symbol}'")

    strategies = [s.strip().lower() for s in run_cfg.strategies]
    unknown_strats = sorted(set(strategies) - SUPPORTED_STRATEGIES)
    if unknown_strats:
        errors.append(f"strategies contains unsupported values: {unknown_strats}")

    sd = _parse_iso_date(run_cfg.start_date, "start_date", errors)
    ed = _parse_iso_date(run_cfg.end_date, "end_date", errors)
    if sd and ed and sd > ed:
        errors.append(f"start_date must be <= end_date (got {sd} > {ed})")

    if errors:
        raise ConfigError("Invalid Phase2RunConfig", errors=errors)


def validate_strategy_params(strategy_name: str, params: Mapping[str, Any], tenor: str) -> None:
    errors: List[str] = []
    s = strategy_name.strip().lower()
    t = _normalize_upper(tenor)

    if s not in SUPPORTED_STRATEGIES:
        errors.append(f"Unsupported strategy '{strategy_name}'")

    if t not in ALLOWED_TENORS_SINGLE:
        errors.append(f"tenor must be WEEKLY or MONTHLY, got '{tenor}'")

    # Required selection fields presence (even if nullable)
    missing = [k for k in REQUIRED_SELECTION_FIELDS if k not in params]
    if missing:
        errors.append(f"Missing required selection fields: {missing}")

    # Common validations
    try:
        qty_lots = int(params.get("qty_lots"))
        if qty_lots < 1:
            errors.append("qty_lots must be >= 1")
    except Exception:
        errors.append("qty_lots must be an int")

    for k in ("strike_band_n_weekly", "strike_band_n_monthly"):
        try:
            if int(params.get(k)) < 1:
                errors.append(f"{k} must be >= 1")
        except Exception:
            errors.append(f"{k} must be an int")

    try:
        if int(params.get("max_atm_search_steps")) < 0:
            errors.append("max_atm_search_steps must be >= 0")
    except Exception:
        errors.append("max_atm_search_steps must be an int")

    lm = _normalize_upper(str(params.get("liquidity_mode")))
    if lm not in LIQUIDITY_MODES:
        errors.append(f"liquidity_mode must be one of {sorted(LIQUIDITY_MODES)}, got '{params.get('liquidity_mode')}'")

    if lm == "ABSOLUTE":
        try:
            if int(params.get("min_contracts")) < 1:
                errors.append("min_contracts must be >= 1 when liquidity_mode=ABSOLUTE")
        except Exception:
            errors.append("min_contracts must be an int when liquidity_mode=ABSOLUTE")
        try:
            if int(params.get("min_open_int")) < 1:
                errors.append("min_open_int must be >= 1 when liquidity_mode=ABSOLUTE")
        except Exception:
            errors.append("min_open_int must be an int when liquidity_mode=ABSOLUTE")

    if lm == "PERCENTILE":
        try:
            p = int(params.get("liquidity_percentile"))
            if p < 0 or p > 100:
                errors.append("liquidity_percentile must be between 0 and 100 when liquidity_mode=PERCENTILE")
        except Exception:
            errors.append("liquidity_percentile must be an int when liquidity_mode=PERCENTILE")

    er = _normalize_upper(str(params.get("exit_rule")))
    if er not in EXIT_RULES:
        errors.append(f"exit_rule must be one of {sorted(EXIT_RULES)}, got '{params.get('exit_rule')}'")
    if er == "K_DAYS_BEFORE_EXPIRY":
        k = params.get("exit_k_days")
        try:
            if k is None or int(k) <= 0:
                errors.append("exit_k_days must be positive when exit_rule=K_DAYS_BEFORE_EXPIRY")
        except Exception:
            errors.append("exit_k_days must be an int when exit_rule=K_DAYS_BEFORE_EXPIRY")

    # Strategy-specific validations
    if s in {"bull_call_spread", "bear_put_spread"}:
        wp = params.get("width_points")
        try:
            if wp is None or int(wp) <= 0:
                errors.append("width_points must be > 0 for spreads")
        except Exception:
            errors.append("width_points must be an int > 0 for spreads")

    if s == "short_strangle":
        d = params.get("otm_distance_points")
        try:
            if d is None or int(d) <= 0:
                errors.append("otm_distance_points must be > 0 for short_strangle")
        except Exception:
            errors.append("otm_distance_points must be an int > 0 for short_strangle")

    # Derived sanity
    try:
        if int(params.get("strike_band_n")) < 1:
            errors.append("strike_band_n must be >= 1 (derived)")
    except Exception:
        errors.append("strike_band_n must be an int (derived)")

    if errors:
        raise ConfigError(f"Invalid strategy params for '{s}'/{t}", errors=errors)
