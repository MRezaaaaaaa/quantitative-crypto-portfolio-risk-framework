"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for chart smoke tests

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """50 days of synthetic, strictly positive prices for BTC, ETH, SOL."""
    rng = np.random.default_rng(seed=7)
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    base = {"BTC": 30000.0, "ETH": 2000.0, "SOL": 80.0}
    prices = {}
    for symbol, start in base.items():
        rets = rng.normal(loc=0.001, scale=0.02, size=n)
        path = start * np.cumprod(1.0 + rets)
        prices[symbol] = path
    return pd.DataFrame(prices, index=dates)


@pytest.fixture
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns from ``sample_prices``."""
    from var_cvar_crypto_risk.returns import calculate_simple_returns

    return calculate_simple_returns(sample_prices)


@pytest.fixture
def sample_weights() -> pd.Series:
    """Fixed weights ``BTC=0.5, ETH=0.3, SOL=0.2``."""
    return pd.Series({"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}, dtype=float)


@pytest.fixture
def sample_portfolio_returns(
    sample_returns: pd.DataFrame,
    sample_weights: pd.Series,
) -> pd.Series:
    """Portfolio returns using the fixed weights."""
    from var_cvar_crypto_risk.portfolio import calculate_portfolio_returns

    return calculate_portfolio_returns(sample_returns, sample_weights)


@pytest.fixture
def long_returns() -> pd.Series:
    """500-observation synthetic return series with fat tails (Student-t, df=4)."""
    rng = np.random.default_rng(seed=42)
    n = 500
    df = 4
    samples = rng.standard_t(df=df, size=n) * 0.02
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.Series(samples, index=dates, name="returns")
