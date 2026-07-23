"""Tests for correlation and diversification analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.correlation import (
    calculate_correlation_matrix,
    calculate_rolling_average_correlation,
    calculate_stress_vs_normal_correlation,
    calculate_weighted_average_correlation,
)


def test_correlation_matrix_square_and_named(sample_returns: pd.DataFrame) -> None:
    corr = calculate_correlation_matrix(sample_returns, method="pearson")
    assert corr.shape == (sample_returns.shape[1], sample_returns.shape[1])
    assert list(corr.columns) == list(sample_returns.columns)
    assert list(corr.index) == list(sample_returns.columns)


def test_correlation_matrix_diagonal_is_one(sample_returns: pd.DataFrame) -> None:
    corr = calculate_correlation_matrix(sample_returns)
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)


def test_correlation_matrix_spearman_supported(sample_returns: pd.DataFrame) -> None:
    corr = calculate_correlation_matrix(sample_returns, method="spearman")
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)


def test_correlation_matrix_bad_method_raises(sample_returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        calculate_correlation_matrix(sample_returns, method="kendall")


def test_rolling_average_correlation_returns_series(sample_returns: pd.DataFrame) -> None:
    window = 20
    out = calculate_rolling_average_correlation(sample_returns, window=window)
    assert isinstance(out, pd.Series)
    assert len(out) == len(sample_returns.dropna()) - window + 1
    # average correlation is bounded in [-1, 1]
    assert (out.abs() <= 1.0 + 1e-9).all()


def test_rolling_average_correlation_needs_two_assets() -> None:
    single = pd.DataFrame({"BTC": np.linspace(0.0, 0.01, 50)})
    with pytest.raises(ValueError):
        calculate_rolling_average_correlation(single, window=10)


def test_correlation_matrix_rejects_wrong_type_and_no_assets() -> None:
    with pytest.raises(ValueError, match="pandas DataFrame"):
        calculate_correlation_matrix(pd.Series([0.01]))
    with pytest.raises(ValueError, match="at least one asset"):
        calculate_correlation_matrix(pd.DataFrame(index=range(3)))


def test_rolling_correlation_rejects_type_window_and_short_sample() -> None:
    with pytest.raises(ValueError, match="pandas DataFrame"):
        calculate_rolling_average_correlation(pd.Series([0.01, 0.02]), window=2)

    two_assets = pd.DataFrame({"A": [0.01, 0.02], "B": [0.02, 0.03]})
    with pytest.raises(ValueError, match="window must be"):
        calculate_rolling_average_correlation(two_assets, window=1)
    with pytest.raises(ValueError, match="Need at least window"):
        calculate_rolling_average_correlation(two_assets, window=3)


def test_weighted_average_correlation_validation_and_value() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.4], [0.4, 1.0]],
        index=["BTC", "ETH"],
        columns=["BTC", "ETH"],
    )
    weights = pd.Series({"BTC": 0.6, "ETH": 0.4})
    assert calculate_weighted_average_correlation(corr, weights) == pytest.approx(0.4)

    with pytest.raises(ValueError, match="pandas DataFrame"):
        calculate_weighted_average_correlation(np.eye(2), weights)
    with pytest.raises(ValueError, match="pandas Series"):
        calculate_weighted_average_correlation(corr, [0.6, 0.4])
    with pytest.raises(ValueError, match="Assets missing"):
        calculate_weighted_average_correlation(
            corr, pd.Series({"BTC": 0.5, "SOL": 0.5})
        )
    with pytest.raises(ValueError, match="at least 2 non-zero"):
        calculate_weighted_average_correlation(
            corr, pd.Series({"BTC": 1.0, "ETH": 0.0})
        )


def test_weighted_average_correlation_rejects_zero_pair_denominator() -> None:
    assets = ["A", "B", "C"]
    corr = pd.DataFrame(np.eye(3), index=assets, columns=assets)
    weights = pd.Series({"A": 1.0, "B": 1.0, "C": -0.5})
    with pytest.raises(ValueError, match="products sum to zero"):
        calculate_weighted_average_correlation(corr, weights)


def test_stress_correlation_rejects_invalid_inputs_and_short_overlap() -> None:
    assets = pd.DataFrame(
        {
            "BTC": np.linspace(-0.02, 0.02, 10),
            "ETH": np.linspace(-0.01, 0.03, 10),
        }
    )
    portfolio = assets.mean(axis=1)

    with pytest.raises(ValueError, match="pandas DataFrame"):
        calculate_stress_vs_normal_correlation(portfolio, portfolio)
    with pytest.raises(ValueError, match="pandas Series"):
        calculate_stress_vs_normal_correlation(assets, assets)
    with pytest.raises(ValueError, match="stress_quantile"):
        calculate_stress_vs_normal_correlation(assets, portfolio, stress_quantile=0.5)
    with pytest.raises(ValueError, match="20 overlapping"):
        calculate_stress_vs_normal_correlation(assets, portfolio)
