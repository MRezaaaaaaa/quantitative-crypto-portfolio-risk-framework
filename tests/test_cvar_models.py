"""Tests for CVaR models."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from var_cvar_crypto_risk.cvar_models import (
    calculate_cvar,
    gaussian_cvar,
    historical_cvar,
)
from var_cvar_crypto_risk.var_models import gaussian_var, historical_var


def test_historical_cvar_positive(long_returns: pd.Series) -> None:
    value = historical_cvar(long_returns, confidence_level=0.95)
    assert value > 0
    assert math.isfinite(value)


def test_gaussian_cvar_positive(long_returns: pd.Series) -> None:
    value = gaussian_cvar(long_returns, confidence_level=0.95)
    assert value > 0


def test_historical_cvar_geq_historical_var(long_returns: pd.Series) -> None:
    var_value = historical_var(long_returns, confidence_level=0.95)
    cvar_value = historical_cvar(long_returns, confidence_level=0.95)
    assert cvar_value >= var_value - 1e-9


def test_gaussian_cvar_geq_gaussian_var(long_returns: pd.Series) -> None:
    var_value = gaussian_var(long_returns, confidence_level=0.95)
    cvar_value = gaussian_cvar(long_returns, confidence_level=0.95)
    assert cvar_value >= var_value - 1e-9


@pytest.mark.parametrize("bad_conf", [0.0, 1.0, -0.5])
def test_invalid_confidence_raises(long_returns: pd.Series, bad_conf: float) -> None:
    with pytest.raises(ValueError):
        historical_cvar(long_returns, confidence_level=bad_conf)


def test_calculate_cvar_unknown_method_raises(long_returns: pd.Series) -> None:
    with pytest.raises(ValueError):
        calculate_cvar(long_returns, method="bogus", confidence_level=0.95)


def test_cvar_models_reject_insufficient_samples() -> None:
    with pytest.raises(ValueError, match="Historical CVaR"):
        historical_cvar(pd.Series(dtype=float))
    with pytest.raises(ValueError, match="at least 2"):
        gaussian_cvar(pd.Series([0.01]))
