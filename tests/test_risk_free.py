"""Tests for the Phase 5.5 risk-free rate helper."""

from __future__ import annotations

import pytest

from var_cvar_crypto_risk.utils import annual_to_horizon_rate


def test_annual_to_horizon_rate_converts_correctly() -> None:
    # 5% annual over 1 day (365 day-count)
    expected = (1.05) ** (1 / 365) - 1.0
    assert annual_to_horizon_rate(0.05, horizon_days=1) == pytest.approx(expected)


def test_annual_to_horizon_rate_full_year_is_annual() -> None:
    assert annual_to_horizon_rate(0.05, horizon_days=365) == pytest.approx(0.05)


def test_annual_to_horizon_rate_zero_is_zero() -> None:
    assert annual_to_horizon_rate(0.0, horizon_days=30) == 0.0


def test_annual_to_horizon_rate_manual_value_compounds() -> None:
    rate = annual_to_horizon_rate(0.10, horizon_days=7, day_count=365)
    assert rate == pytest.approx((1.10) ** (7 / 365) - 1.0)
    assert rate > 0.0


def test_annual_to_horizon_rate_invalid_horizon_raises() -> None:
    with pytest.raises(ValueError):
        annual_to_horizon_rate(0.05, horizon_days=0)
