"""Tests for VaR models."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from var_cvar_crypto_risk.var_models import (
    calculate_var,
    cornish_fisher_var,
    gaussian_var,
    historical_var,
    scale_var_to_horizon,
)


def test_historical_var_positive(long_returns: pd.Series) -> None:
    value = historical_var(long_returns, confidence_level=0.95)
    assert value > 0
    assert math.isfinite(value)


def test_gaussian_var_positive(long_returns: pd.Series) -> None:
    value = gaussian_var(long_returns, confidence_level=0.95)
    assert value > 0


def test_cornish_fisher_var_positive(long_returns: pd.Series) -> None:
    value = cornish_fisher_var(long_returns, confidence_level=0.95)
    assert value > 0


@pytest.mark.parametrize("method", ["historical", "gaussian", "cornish_fisher"])
def test_higher_confidence_gives_at_least_as_high_var(
    long_returns: pd.Series, method: str
) -> None:
    var_95 = calculate_var(long_returns, method=method, confidence_level=0.95)
    var_99 = calculate_var(long_returns, method=method, confidence_level=0.99)
    assert var_99 >= var_95 - 1e-9


@pytest.mark.parametrize("bad_conf", [0.0, 1.0, -0.1])
def test_invalid_confidence_raises(long_returns: pd.Series, bad_conf: float) -> None:
    with pytest.raises(ValueError):
        historical_var(long_returns, confidence_level=bad_conf)


def test_calculate_var_unknown_method_raises(long_returns: pd.Series) -> None:
    with pytest.raises(ValueError):
        calculate_var(long_returns, method="bogus", confidence_level=0.95)


def test_scale_var_to_horizon_uses_sqrt_of_time() -> None:
    scaled = scale_var_to_horizon(0.04, 10)
    assert math.isclose(scaled, 0.04 * math.sqrt(10), rel_tol=1e-12)
