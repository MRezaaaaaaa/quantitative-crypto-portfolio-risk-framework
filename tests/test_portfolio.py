"""Tests for the portfolio module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.portfolio import (
    calculate_portfolio_returns,
    calculate_portfolio_value,
    normalize_weights,
    validate_weights,
)


def test_portfolio_returns_match_weighted_sum(
    sample_returns: pd.DataFrame, sample_weights: pd.Series
) -> None:
    portfolio = calculate_portfolio_returns(sample_returns, sample_weights)
    expected = (sample_returns * sample_weights).sum(axis=1)
    np.testing.assert_allclose(portfolio.values, expected.values, rtol=1e-12)


def test_validate_weights_pass_when_sum_is_one(sample_weights: pd.Series) -> None:
    validate_weights(
        sample_weights,
        assets=["BTC", "ETH", "SOL"],
        allow_short_selling=False,
    )


def test_validate_weights_fail_when_sum_not_one() -> None:
    bad = pd.Series({"BTC": 0.5, "ETH": 0.3, "SOL": 0.1})
    with pytest.raises(ValueError):
        validate_weights(bad, ["BTC", "ETH", "SOL"], allow_short_selling=False)


def test_validate_weights_negative_disallowed() -> None:
    bad = pd.Series({"BTC": 0.7, "ETH": 0.5, "SOL": -0.2})
    with pytest.raises(ValueError):
        validate_weights(bad, ["BTC", "ETH", "SOL"], allow_short_selling=False)


def test_validate_weights_negative_allowed_when_short_enabled() -> None:
    ok = pd.Series({"BTC": 0.7, "ETH": 0.5, "SOL": -0.2})
    validate_weights(ok, ["BTC", "ETH", "SOL"], allow_short_selling=True)


def test_normalize_weights_sums_to_one() -> None:
    raw = pd.Series({"BTC": 2.0, "ETH": 3.0, "SOL": 5.0})
    normalized = normalize_weights(raw)
    assert abs(float(normalized.sum()) - 1.0) < 1e-12


def test_portfolio_returns_index_matches_input(
    sample_returns: pd.DataFrame, sample_weights: pd.Series
) -> None:
    portfolio = calculate_portfolio_returns(sample_returns, sample_weights)
    pd.testing.assert_index_equal(portfolio.index, sample_returns.index)


def test_log_portfolio_return_is_built_from_simple_asset_gross_returns() -> None:
    simple = pd.DataFrame(
        {"A": [0.10, -0.05], "B": [0.00, 0.02]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    log_returns = np.log1p(simple)
    weights = pd.Series({"A": 0.6, "B": 0.4})
    result = calculate_portfolio_returns(
        log_returns,
        weights,
        return_method="log",
    )
    expected_simple = simple @ weights
    expected_log = np.log1p(expected_simple)
    np.testing.assert_allclose(result, expected_log, rtol=1e-12)


def test_portfolio_value_supports_log_returns_without_simple_approximation() -> None:
    simple = pd.Series(
        [0.10, -0.05, 0.02],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    expected = calculate_portfolio_value(simple, 100_000.0, return_method="simple")
    actual = calculate_portfolio_value(
        np.log1p(simple),
        100_000.0,
        return_method="log",
    )
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("method", ["arithmetic", "", "LOG"])
def test_portfolio_functions_reject_unknown_return_method(method: str) -> None:
    returns = pd.DataFrame({"A": [0.01], "B": [0.02]})
    weights = pd.Series({"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError, match="return_method"):
        calculate_portfolio_returns(returns, weights, return_method=method)
    with pytest.raises(ValueError, match="return_method"):
        calculate_portfolio_value(
            pd.Series([0.01]),
            100_000.0,
            return_method=method,
        )
