import sys
from pathlib import Path

# Ensures imports work regardless of where pytest is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../<repo_root>
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import pytest

from src.engine.settlement_marking import (
    compute_settle_used,
    PRICE_METHOD_INTRINSIC,
    PRICE_METHOD_SETTLE,
)
from src.utils.errors import DataIntegrityError


def _base_row(**overrides):
    row = {
        "instrument": "OPTIDX",
        "option_typ": "CE",
        "strike_pr": 20000.0,
        "settle_pr": 123.0,
        "spot_close": 20100.0,
        "cal_days_to_expiry": 1,
    }
    row.update(overrides)
    return row


def test_settle_used_non_expiry_equals_settle_pr():
    df = pd.DataFrame([_base_row(cal_days_to_expiry=2, settle_pr=111.5)])
    out = compute_settle_used(df)

    assert out.loc[0, "settle_used"] == pytest.approx(111.5)
    assert out.loc[0, "price_method"] == PRICE_METHOD_SETTLE


def test_settle_used_opt_expiry_intrinsic_ce():
    df = pd.DataFrame(
        [
            _base_row(
                cal_days_to_expiry=0,
                option_typ="CE",
                spot_close=20100.0,
                strike_pr=20000.0,
                settle_pr=999.0,
            )
        ]
    )
    out = compute_settle_used(df)

    assert out.loc[0, "settle_used"] == pytest.approx(100.0)
    assert out.loc[0, "price_method"] == PRICE_METHOD_INTRINSIC
    assert out.loc[0, "settle_used"] >= 0.0


def test_settle_used_opt_expiry_intrinsic_pe():
    df = pd.DataFrame(
        [
            _base_row(
                cal_days_to_expiry=0,
                option_typ="PE",
                spot_close=19950.0,
                strike_pr=20000.0,
                settle_pr=999.0,
            )
        ]
    )
    out = compute_settle_used(df)

    assert out.loc[0, "settle_used"] == pytest.approx(50.0)
    assert out.loc[0, "price_method"] == PRICE_METHOD_INTRINSIC
    assert out.loc[0, "settle_used"] >= 0.0


def test_settle_used_futidx_unchanged():
    df = pd.DataFrame(
        [
            _base_row(
                instrument="FUTIDX",
                option_typ="XX",
                strike_pr=0.0,
                cal_days_to_expiry=0,
                settle_pr=250.0,
            )
        ]
    )
    out = compute_settle_used(df)

    assert out.loc[0, "settle_used"] == pytest.approx(250.0)
    assert out.loc[0, "price_method"] == PRICE_METHOD_SETTLE


def test_settle_used_opt_expiry_missing_spot_or_strike_raises():
    df = pd.DataFrame([_base_row(cal_days_to_expiry=0, spot_close=None)])
    with pytest.raises(DataIntegrityError):
        compute_settle_used(df)
