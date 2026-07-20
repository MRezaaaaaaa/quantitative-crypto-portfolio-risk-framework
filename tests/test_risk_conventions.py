"""Contract tests for signed VaR/CVaR values and unit conversions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.backtesting import rolling_var_forecast
from var_cvar_crypto_risk.cvar_models import (
    gaussian_cvar,
    historical_cvar,
    return_cvar_to_money_cvar,
)
from var_cvar_crypto_risk.monte_carlo import scenario_cvar, scenario_var
from var_cvar_crypto_risk.risk_conventions import (
    loss_value_to_money,
    loss_value_to_return_threshold,
    return_threshold_to_loss_value,
)
from var_cvar_crypto_risk.var_models import (
    cornish_fisher_var,
    gaussian_var,
    historical_var,
    return_var_to_money_var,
    scale_var_to_horizon,
)


def _all_gain_returns() -> pd.Series:
    values = np.linspace(0.010, 0.012, 200)
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=200, freq="D"))


def _all_loss_returns() -> pd.Series:
    values = np.linspace(-0.012, -0.010, 200)
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=200, freq="D"))


@pytest.mark.parametrize(
    "estimator",
    [historical_var, gaussian_var, cornish_fisher_var],
)
def test_var_is_negative_when_the_left_tail_is_profitable(estimator) -> None:
    """A negative loss-space VaR represents a positive return threshold."""
    value = estimator(_all_gain_returns(), confidence_level=0.95)
    assert value < 0.0
    assert loss_value_to_return_threshold(value) > 0.0


@pytest.mark.parametrize("estimator", [historical_cvar, gaussian_cvar])
def test_cvar_is_negative_when_the_left_tail_is_profitable(estimator) -> None:
    value = estimator(_all_gain_returns(), confidence_level=0.95)
    assert value < 0.0


@pytest.mark.parametrize(
    "estimator",
    [historical_var, gaussian_var, cornish_fisher_var],
)
def test_var_is_positive_when_the_left_tail_is_loss_making(estimator) -> None:
    value = estimator(_all_loss_returns(), confidence_level=0.95)
    assert value > 0.0


@pytest.mark.parametrize("estimator", [historical_cvar, gaussian_cvar])
def test_cvar_is_positive_when_the_left_tail_is_loss_making(estimator) -> None:
    value = estimator(_all_loss_returns(), confidence_level=0.95)
    assert value > 0.0


def test_historical_var_is_negative_return_quantile() -> None:
    returns = _all_gain_returns()
    expected_threshold = float(np.quantile(returns.to_numpy(), 0.05))
    value = historical_var(returns, confidence_level=0.95)
    assert value == pytest.approx(-expected_threshold)
    assert loss_value_to_return_threshold(value) == pytest.approx(expected_threshold)


def test_scenario_var_and_cvar_follow_signed_loss_contract() -> None:
    scenarios = _all_gain_returns().rename("scenario_return")
    var = scenario_var(scenarios, confidence_level=0.95)
    cvar = scenario_cvar(scenarios, confidence_level=0.95)
    assert var < 0.0
    assert cvar < 0.0
    assert cvar >= var


@pytest.mark.parametrize("return_threshold", [-0.04, 0.0, 0.02])
def test_return_and_loss_conversions_are_exact_inverses(
    return_threshold: float,
) -> None:
    loss_value = return_threshold_to_loss_value(return_threshold)
    assert loss_value_to_return_threshold(loss_value) == pytest.approx(return_threshold)


@pytest.mark.parametrize(
    ("loss_value", "expected_money"),
    [(0.04, 4_000.0), (0.0, 0.0), (-0.02, -2_000.0)],
)
def test_money_conversion_preserves_loss_space_sign(
    loss_value: float,
    expected_money: float,
) -> None:
    assert loss_value_to_money(loss_value, 100_000.0) == pytest.approx(expected_money)


def test_legacy_var_and_cvar_money_helpers_preserve_negative_sign() -> None:
    assert return_var_to_money_var(-0.02, 100_000.0) == pytest.approx(-2_000.0)
    assert return_cvar_to_money_cvar(-0.03, 100_000.0) == pytest.approx(-3_000.0)


def test_sqrt_time_scaling_preserves_negative_sign() -> None:
    assert scale_var_to_horizon(-0.02, 4) == pytest.approx(-0.04)


def test_backtest_compares_return_to_negative_signed_var() -> None:
    returns = pd.Series(
        np.full(80, 0.01),
        index=pd.date_range("2024-01-01", periods=80, freq="D"),
    )
    result = rolling_var_forecast(
        returns,
        method="historical",
        confidence_level=0.95,
        window=30,
    )
    assert (result["var_forecast"] < 0.0).all()
    expected_breach = result["actual_return"] < -result["var_forecast"]
    pd.testing.assert_series_equal(result["breach"], expected_breach, check_names=False)


@pytest.mark.parametrize("portfolio_value", [-1.0, float("nan"), float("inf")])
def test_money_conversion_rejects_invalid_portfolio_value(
    portfolio_value: float,
) -> None:
    with pytest.raises(ValueError, match="portfolio_value"):
        loss_value_to_money(0.04, portfolio_value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_contract_conversions_reject_non_finite_risk_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        return_threshold_to_loss_value(value)
    with pytest.raises(ValueError, match="finite"):
        loss_value_to_return_threshold(value)
    with pytest.raises(ValueError, match="finite"):
        loss_value_to_money(value, 100_000.0)
