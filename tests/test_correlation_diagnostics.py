"""Tests for weighted-average and stress-versus-normal correlation diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from var_cvar_crypto_risk.correlation import (  # noqa: E402
    calculate_correlation_matrix,
    calculate_stress_vs_normal_correlation,
    calculate_weighted_average_correlation,
)


@pytest.fixture
def three_asset_returns() -> pd.DataFrame:
    rng = np.random.default_rng(seed=9)
    n = 300
    common = rng.normal(0, 0.02, size=n)
    df = pd.DataFrame(
        {
            "A": common + rng.normal(0, 0.01, size=n),
            "B": common + rng.normal(0, 0.01, size=n),
            "C": rng.normal(0, 0.02, size=n),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )
    return df


def test_weighted_avg_correlation_equal_weights_matches_simple(
    three_asset_returns,
):
    corr = calculate_correlation_matrix(three_asset_returns)
    w = pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    weighted = calculate_weighted_average_correlation(corr, w)
    n = corr.shape[0]
    simple = float(
        (corr.to_numpy().sum() - n) / (n * (n - 1))
    )
    assert weighted == pytest.approx(simple)


def test_weighted_avg_correlation_tilts_toward_heavy_pair(three_asset_returns):
    corr = calculate_correlation_matrix(three_asset_returns)
    # A and B are highly correlated; overweighting them must raise the
    # weighted average above the equal-weighted average.
    w_conc = pd.Series({"A": 0.49, "B": 0.49, "C": 0.02})
    w_eq = pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    assert calculate_weighted_average_correlation(
        corr, w_conc
    ) > calculate_weighted_average_correlation(corr, w_eq)


def test_weighted_avg_correlation_errors(three_asset_returns):
    corr = calculate_correlation_matrix(three_asset_returns)
    with pytest.raises(ValueError, match="missing"):
        calculate_weighted_average_correlation(
            corr, pd.Series({"A": 0.5, "Z": 0.5})
        )
    with pytest.raises(ValueError, match="at least 2"):
        calculate_weighted_average_correlation(corr, pd.Series({"A": 1.0}))


def test_stress_vs_normal_correlation_shape(three_asset_returns):
    portfolio = three_asset_returns.mean(axis=1)
    out = calculate_stress_vs_normal_correlation(
        three_asset_returns, portfolio, stress_quantile=0.10
    )
    assert set(out) == {
        "stress_avg_corr",
        "normal_avg_corr",
        "n_stress_days",
        "n_normal_days",
        "stress_threshold",
    }
    n_total = out["n_stress_days"] + out["n_normal_days"]
    assert n_total == len(three_asset_returns)
    assert out["n_stress_days"] == pytest.approx(0.10 * n_total, rel=0.2)
    assert -1.0 <= out["stress_avg_corr"] <= 1.0
    assert -1.0 <= out["normal_avg_corr"] <= 1.0


def test_stress_vs_normal_validation(three_asset_returns):
    portfolio = three_asset_returns.mean(axis=1)
    with pytest.raises(ValueError):
        calculate_stress_vs_normal_correlation(
            three_asset_returns, portfolio, stress_quantile=0.7
        )
    with pytest.raises(ValueError):
        calculate_stress_vs_normal_correlation(
            three_asset_returns.iloc[:, :1], portfolio
        )
