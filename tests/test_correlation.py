"""Tests for the Phase 5.5 correlation/diversification analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.correlation import (
    calculate_correlation_matrix,
    calculate_rolling_average_correlation,
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
