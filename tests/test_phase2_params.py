import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so `import src...` works even when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.phase2_params import (  # noqa: E402
    ConfigError,
    get_phase2_default_run_config,
    resolve_effective_strategy_params,
    validate_strategy_params,
)

# Optional: support either constant name (depending on your implementation)
try:
    from src.config.phase2_params import PHASE2_PARAM_VERSION as _PARAM_VERSION  # type: ignore  # noqa: E402
except Exception:
    try:
        from src.config.phase2_params import PARAM_VERSION as _PARAM_VERSION  # type: ignore  # noqa: E402
    except Exception:
        _PARAM_VERSION = None


def test_defaults_run_config_has_tenor_both():
    cfg = get_phase2_default_run_config()
    assert cfg.tenor == "BOTH"


def test_default_strategy_params_contains_required_selection_fields():
    params = resolve_effective_strategy_params("short_straddle", "WEEKLY", user_overrides={})

    # Core selection fields should exist even if nullable
    required = [
        "qty_lots",
        "strike_band_n_weekly",
        "strike_band_n_monthly",
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
        "width_points",
        "otm_distance_points",
    ]
    for k in required:
        assert k in params

    # Only assert param_version if your implementation provides it
    if _PARAM_VERSION is not None and "param_version" in params:
        assert params["param_version"] == _PARAM_VERSION


def test_resolve_strangle_distance_weekly_monthly():
    wk = resolve_effective_strategy_params("short_strangle", "WEEKLY", user_overrides={})
    mo = resolve_effective_strategy_params("short_strangle", "MONTHLY", user_overrides={})
    assert wk["otm_distance_points"] == 300
    assert mo["otm_distance_points"] == 400


def test_resolve_strike_band_weekly_monthly():
    wk = resolve_effective_strategy_params("short_straddle", "WEEKLY", user_overrides={})
    mo = resolve_effective_strategy_params("short_straddle", "MONTHLY", user_overrides={})
    assert wk["strike_band_n"] == 10
    assert mo["strike_band_n"] == 15


def test_override_width_points_applied():
    params = resolve_effective_strategy_params(
        "bull_call_spread", "WEEKLY", user_overrides={"width_points": 250}
    )
    assert params["width_points"] == 250
    validate_strategy_params("bull_call_spread", params, "WEEKLY")


def test_override_qty_lots_applied():
    params = resolve_effective_strategy_params(
        "short_straddle", "WEEKLY", user_overrides={"qty_lots": 3}
    )
    assert params["qty_lots"] == 3
    validate_strategy_params("short_straddle", params, "WEEKLY")


def test_validate_strategy_params_rejects_bad_liquidity_mode():
    params = resolve_effective_strategy_params("short_straddle", "WEEKLY", user_overrides={})
    params["liquidity_mode"] = "NOPE"
    with pytest.raises(ConfigError) as e:
        validate_strategy_params("short_straddle", params, "WEEKLY")
    # Your ConfigError may store details in e.value.errors; also keep a loose string check
    assert "liquidity_mode" in (str(e.value) + " " + " ".join(getattr(e.value, "errors", [])))


def test_validate_strategy_params_rejects_negative_width():
    with pytest.raises(ConfigError) as e:
        resolve_effective_strategy_params(
            "bull_call_spread", "WEEKLY", user_overrides={"width_points": -1}
        )
    assert "width_points" in (str(e.value) + " " + " ".join(getattr(e.value, "errors", [])))


def test_validate_strategy_params_percentile_bounds():
    with pytest.raises(ConfigError) as e:
        resolve_effective_strategy_params(
            "short_straddle",
            "WEEKLY",
            user_overrides={"liquidity_mode": "PERCENTILE", "liquidity_percentile": 101},
        )
    assert "liquidity_percentile" in (str(e.value) + " " + " ".join(getattr(e.value, "errors", [])))


def test_effective_params_json_serializable():
    params = resolve_effective_strategy_params("bear_put_spread", "MONTHLY", user_overrides={})
    json.dumps(params)  # should not raise


if __name__ == "__main__":
    # Allows: python tests/test_phase2_params.py
    raise SystemExit(pytest.main([__file__]))
