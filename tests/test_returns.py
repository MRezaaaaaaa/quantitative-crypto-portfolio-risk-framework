"""Tests for the returns module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.returns import (
    annualize_return,
    annualize_volatility,
    calculate_cumulative_returns,
    calculate_horizon_returns,
    calculate_log_returns,
    calculate_returns,
    calculate_simple_returns,
)


def test_simple_returns_first_row_correct(sample_prices: pd.DataFrame) -> None:
    rets = calculate_simple_returns(sample_prices)
    expected = sample_prices.iloc[1] / sample_prices.iloc[0] - 1
    pd.testing.assert_series_equal(
        rets.iloc[0], expected, check_names=False
    )


def test_log_returns_first_row_correct(sample_prices: pd.DataFrame) -> None:
    rets = calculate_log_returns(sample_prices)
    expected = np.log(sample_prices.iloc[1] / sample_prices.iloc[0])
    pd.testing.assert_series_equal(
        rets.iloc[0], expected, check_names=False
    )


def test_calculate_returns_no_nan(sample_prices: pd.DataFrame) -> None:
    rets = calculate_returns(sample_prices, method="simple")
    assert not rets.isna().any().any()
    assert len(rets) == len(sample_prices) - 1


def test_calculate_returns_unknown_method_raises(sample_prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        calculate_returns(sample_prices, method="bogus")


def test_annualization_returns_floats(sample_portfolio_returns: pd.Series) -> None:
    ann_ret = annualize_return(sample_portfolio_returns, periods_per_year=365)
    ann_vol = annualize_volatility(sample_portfolio_returns, periods_per_year=365)
    assert isinstance(ann_ret, float)
    assert isinstance(ann_vol, float)
    assert ann_vol > 0
    assert math.isfinite(ann_ret)
    assert math.isfinite(ann_vol)


# ── Phase 5.5: horizon returns ──────────────────────────────────────────────


def test_horizon_returns_simple_overlapping(sample_portfolio_returns: pd.Series) -> None:
    h = 5
    clean = sample_portfolio_returns.dropna()
    out = calculate_horizon_returns(clean, horizon_days=h, method="simple", overlapping=True)
    assert isinstance(out, pd.Series)
    assert len(out) == len(clean) - h + 1
    expected_first = float(np.prod(1.0 + clean.to_numpy()[:h]) - 1.0)
    assert out.iloc[0] == pytest.approx(expected_first)


def test_horizon_returns_h1_is_identity(sample_portfolio_returns: pd.Series) -> None:
    clean = sample_portfolio_returns.dropna()
    out = calculate_horizon_returns(clean, horizon_days=1)
    pd.testing.assert_series_equal(out, clean)


def test_horizon_returns_non_overlapping_block_count(sample_portfolio_returns: pd.Series) -> None:
    h = 5
    clean = sample_portfolio_returns.dropna()
    out = calculate_horizon_returns(clean, horizon_days=h, overlapping=False)
    assert len(out) == len(clean) // h


def test_horizon_returns_are_h_day_not_daily(sample_portfolio_returns: pd.Series) -> None:
    """Horizon-matched returns are larger in magnitude than daily returns."""
    clean = sample_portfolio_returns.dropna()
    daily_std = float(clean.std(ddof=1))
    h7 = calculate_horizon_returns(clean, horizon_days=7, method="simple")
    assert float(h7.std(ddof=1)) > daily_std


def test_horizon_returns_log_matches_sum(sample_portfolio_returns: pd.Series) -> None:
    h = 4
    clean = sample_portfolio_returns.dropna()
    out = calculate_horizon_returns(clean, horizon_days=h, method="log", overlapping=True)
    expected_first = float(np.sum(clean.to_numpy()[:h]))
    assert out.iloc[0] == pytest.approx(expected_first)


def test_cumulative_returns_dataframe_preserves_columns(sample_returns: pd.DataFrame) -> None:
    cum = calculate_cumulative_returns(sample_returns)
    assert list(cum.columns) == list(sample_returns.columns)
    assert cum.shape == sample_returns.shape


def test_cumulative_log_returns_match_simple_compounding() -> None:
    simple = pd.Series([0.10, -0.05, 0.02])
    expected = calculate_cumulative_returns(simple, method="simple")
    actual = calculate_cumulative_returns(np.log1p(simple), method="log")
    pd.testing.assert_series_equal(actual, expected)


def test_cumulative_returns_reject_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        calculate_cumulative_returns(pd.Series([0.01]), method="arithmetic")
