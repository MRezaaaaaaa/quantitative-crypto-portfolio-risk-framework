"""Tests for the robust assumptions engine."""

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

from var_cvar_crypto_risk.assumptions import (  # noqa: E402
    AssumptionConfig,
    build_assumption_table,
    build_expected_return_candidates,
    build_volatility_table,
    estimate_covariance_robust,
    estimate_expected_returns_robust,
    estimate_volatility_robust,
    ewma_covariance,
    ewma_volatility,
    shrink_covariance,
    shrunk_mean_returns,
    trimmed_mean_returns,
    winsorize_frame,
    winsorized_mean_returns,
)


@pytest.fixture
def returns_frame() -> pd.DataFrame:
    """300 obs, 3 assets, fat-tailed with drift and one extreme outlier."""
    rng = np.random.default_rng(seed=7)
    n = 300
    data = rng.standard_t(df=4, size=(n, 3)) * 0.02
    data += np.array([0.002, 0.001, 0.0005])
    df = pd.DataFrame(
        data,
        columns=["BTC", "ETH", "SOL"],
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )
    df.iloc[10, 0] = 0.90  # extreme outlier in BTC
    return df


# ── Expected returns ─────────────────────────────────────────────────────


def test_mean_and_median_match_pandas(returns_frame):
    mu_mean = estimate_expected_returns_robust(returns_frame, "mean")
    mu_median = estimate_expected_returns_robust(returns_frame, "median")
    pd.testing.assert_series_equal(
        mu_mean, returns_frame.mean(), check_names=False
    )
    pd.testing.assert_series_equal(
        mu_median, returns_frame.median(), check_names=False
    )


def test_trimmed_mean_dampens_outlier(returns_frame):
    raw = float(returns_frame["BTC"].mean())
    trimmed = float(trimmed_mean_returns(returns_frame, 0.10)["BTC"])
    assert trimmed < raw  # the +90% outlier is cut


def test_winsorized_mean_dampens_outlier(returns_frame):
    raw = float(returns_frame["BTC"].mean())
    winsor = float(winsorized_mean_returns(returns_frame, 0.05)["BTC"])
    assert winsor < raw


def test_winsorize_frame_clips_at_quantiles(returns_frame):
    clipped = winsorize_frame(returns_frame, 0.05)
    for col in returns_frame.columns:
        lo = returns_frame[col].quantile(0.05)
        hi = returns_frame[col].quantile(0.95)
        assert clipped[col].min() >= lo - 1e-12
        assert clipped[col].max() <= hi + 1e-12


def test_winsorize_zero_proportion_is_identity(returns_frame):
    out = winsorize_frame(returns_frame, 0.0)
    pd.testing.assert_frame_equal(out, returns_frame.dropna().astype(float))


def test_shrinkage_endpoints(returns_frame):
    full = shrunk_mean_returns(returns_frame, 1.0)
    none = shrunk_mean_returns(returns_frame, 0.0)
    pd.testing.assert_series_equal(
        full, returns_frame.mean(), check_names=False
    )
    assert (none == 0.0).all()


def test_shrinkage_halves_mean(returns_frame):
    half = shrunk_mean_returns(returns_frame, 0.5)
    expected = 0.5 * returns_frame.mean()
    pd.testing.assert_series_equal(half, expected, check_names=False)


def test_zero_method(returns_frame):
    mu = estimate_expected_returns_robust(returns_frame, "zero")
    assert (mu == 0.0).all()


def test_unknown_method_raises(returns_frame):
    with pytest.raises(ValueError, match="Unsupported expected-return"):
        estimate_expected_returns_robust(returns_frame, "garch")


def test_invalid_proportions_raise(returns_frame):
    with pytest.raises(ValueError):
        trimmed_mean_returns(returns_frame, 0.6)
    with pytest.raises(ValueError):
        winsorized_mean_returns(returns_frame, -0.1)
    with pytest.raises(ValueError):
        shrunk_mean_returns(returns_frame, 1.5)


def test_candidates_table_columns(returns_frame):
    table = build_expected_return_candidates(returns_frame)
    assert list(table.columns) == [
        "mean",
        "median",
        "trimmed_mean",
        "winsorized_mean",
        "shrinkage_to_zero",
    ]
    assert list(table.index) == ["BTC", "ETH", "SOL"]
    assert table.notna().all().all()


# ── Volatility ───────────────────────────────────────────────────────────


def test_ewma_volatility_positive_and_finite(returns_frame):
    vol = ewma_volatility(returns_frame, 0.94)
    assert (vol > 0).all()
    assert np.isfinite(vol).all()


def test_ewma_volatility_weights_recent_more():
    # Calm history, violent recent past ⇒ EWMA vol > sample vol.
    rng = np.random.default_rng(seed=1)
    calm = rng.normal(0, 0.005, size=250)
    wild = rng.normal(0, 0.06, size=50)
    series = pd.DataFrame({"X": np.concatenate([calm, wild])})
    ewma = float(ewma_volatility(series, 0.94)["X"])
    sample = float(series["X"].std(ddof=1))
    assert ewma > sample


def test_ewma_invalid_lambda_raises(returns_frame):
    with pytest.raises(ValueError):
        ewma_volatility(returns_frame, 1.0)
    with pytest.raises(ValueError):
        ewma_volatility(returns_frame, 0.0)


def test_volatility_dispatch_matches_direct(returns_frame):
    sample = estimate_volatility_robust(returns_frame, "sample")
    pd.testing.assert_series_equal(
        sample, returns_frame.std(ddof=1), check_names=False
    )
    winsor = estimate_volatility_robust(returns_frame, "winsorized")
    assert float(winsor["BTC"]) < float(sample["BTC"])  # outlier clipped


def test_volatility_table_scaling(returns_frame):
    table = build_volatility_table(returns_frame, horizon_days=7)
    np.testing.assert_allclose(
        table["horizon_vol"], table["daily_vol"] * np.sqrt(7)
    )
    np.testing.assert_allclose(
        table["annualized_vol"], table["daily_vol"] * np.sqrt(365)
    )


# ── Covariance ───────────────────────────────────────────────────────────


def test_sample_covariance_matches_pandas(returns_frame):
    cov = estimate_covariance_robust(returns_frame, "sample")
    pd.testing.assert_frame_equal(cov, returns_frame.cov().astype(float))


def test_shrink_covariance_endpoints(returns_frame):
    S = returns_frame.cov()
    none = shrink_covariance(S, 0.0, "diagonal")
    pd.testing.assert_frame_equal(none, (S + S.T) / 2.0)
    full = shrink_covariance(S, 1.0, "diagonal")
    off_diag = full.to_numpy()[~np.eye(3, dtype=bool)]
    np.testing.assert_allclose(off_diag, 0.0, atol=1e-18)
    # Variances preserved by both targets.
    np.testing.assert_allclose(np.diag(full), np.diag(S))
    full_cc = shrink_covariance(S, 1.0, "constant_correlation")
    np.testing.assert_allclose(np.diag(full_cc), np.diag(S))


def test_shrink_covariance_pulls_off_diagonals(returns_frame):
    S = returns_frame.cov()
    shrunk = shrink_covariance(S, 0.5, "diagonal")
    for i in range(3):
        for j in range(3):
            if i != j:
                assert abs(shrunk.iloc[i, j]) <= abs(S.iloc[i, j]) + 1e-18


def test_shrink_covariance_invalid_inputs(returns_frame):
    S = returns_frame.cov()
    with pytest.raises(ValueError):
        shrink_covariance(S, 1.5, "diagonal")
    with pytest.raises(ValueError):
        shrink_covariance(S, 0.5, "unknown_target")
    with pytest.raises(ValueError):
        shrink_covariance(S.iloc[:2], 0.5, "diagonal")  # non-square


def test_ewma_covariance_diagonal_matches_ewma_vol(returns_frame):
    cov = ewma_covariance(returns_frame, 0.94)
    vol = ewma_volatility(returns_frame, 0.94)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), vol.to_numpy())
    # Symmetric PSD-ish
    np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T)
    eigvals = np.linalg.eigvalsh(cov.to_numpy())
    assert eigvals.min() > -1e-12


# ── Config / tables ──────────────────────────────────────────────────────


def test_config_final_expected_returns_with_views(returns_frame):
    config = AssumptionConfig(
        expected_return_method="median",
        manual_views={"BTC": 0.05},
        view_blend_weight=1.0,
    )
    final = config.final_expected_returns(returns_frame)
    assert final["BTC"] == pytest.approx(0.05)
    assert final["ETH"] == pytest.approx(float(returns_frame["ETH"].median()))


def test_config_view_blending(returns_frame):
    config = AssumptionConfig(
        expected_return_method="mean",
        manual_views={"ETH": 0.10},
        view_blend_weight=0.5,
    )
    final = config.final_expected_returns(returns_frame)
    base = float(returns_frame["ETH"].mean())
    assert final["ETH"] == pytest.approx(0.5 * 0.10 + 0.5 * base)


def test_config_ignores_views_for_unknown_assets(returns_frame):
    config = AssumptionConfig(manual_views={"DOGE": 0.42})
    final = config.final_expected_returns(returns_frame)
    pd.testing.assert_series_equal(
        final, returns_frame.mean(), check_names=False
    )


def test_assumption_table_final_column_consistency(returns_frame):
    config = AssumptionConfig(
        expected_return_method="trimmed_mean",
        trim_proportion=0.10,
        manual_views={"SOL": -0.01},
        view_blend_weight=1.0,
    )
    table = build_assumption_table(returns_frame, config)
    assert "final_expected_return" in table.columns
    assert table.loc["SOL", "final_expected_return"] == pytest.approx(-0.01)
    assert table.loc["BTC", "final_expected_return"] == pytest.approx(
        table.loc["BTC", "trimmed_mean"]
    )
    assert pd.isna(table.loc["BTC", "manual_view"])
    assert table.loc["SOL", "manual_view"] == pytest.approx(-0.01)


def test_estimators_reject_empty_frame():
    with pytest.raises(ValueError):
        estimate_expected_returns_robust(pd.DataFrame(), "mean")
