"""Tests for ``var_cvar_crypto_risk.monte_carlo`` (Phase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.monte_carlo import (
    calculate_portfolio_scenario_returns,
    compare_all_risk_methods,
    compare_monte_carlo_distributions,
    estimate_return_parameters,
    monte_carlo_risk_summary,
    scenario_cvar,
    scenario_var,
    simulate_normal_returns,
    simulate_portfolio_paths,
    simulate_student_t_returns,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def asset_returns_3a() -> pd.DataFrame:
    """500 obs × 3 assets (BTC/ETH/SOL), Student-t df=4 scale=0.02, seed=42."""
    rng = np.random.default_rng(42)
    samples = rng.standard_t(df=4, size=(500, 3)) * 0.02
    return pd.DataFrame(
        samples,
        columns=["BTC", "ETH", "SOL"],
        index=pd.date_range("2023-01-01", periods=500, freq="D"),
    )


@pytest.fixture
def weights_3a() -> pd.Series:
    return pd.Series({"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}, dtype=float)


@pytest.fixture
def portfolio_returns_simple() -> pd.Series:
    rng = np.random.default_rng(7)
    samples = rng.standard_t(df=4, size=500) * 0.02
    return pd.Series(
        samples,
        index=pd.date_range("2023-01-01", periods=500, freq="D"),
        name="portfolio_return",
    )


# ── estimate_return_parameters ──────────────────────────────────────────────


def test_estimate_return_parameters_keys_and_shapes(asset_returns_3a):
    params = estimate_return_parameters(asset_returns_3a)
    for key in (
        "mean_vector",
        "covariance_matrix",
        "correlation_matrix",
        "volatility_vector",
        "n_observations",
        "assets",
    ):
        assert key in params
    assert list(params["mean_vector"].index) == ["BTC", "ETH", "SOL"]
    assert params["covariance_matrix"].shape == (3, 3)
    assert params["correlation_matrix"].shape == (3, 3)
    assert list(params["volatility_vector"].index) == ["BTC", "ETH", "SOL"]
    assert params["n_observations"] == 500
    assert params["assets"] == ["BTC", "ETH", "SOL"]


def test_estimate_return_parameters_rejects_empty():
    with pytest.raises(ValueError):
        estimate_return_parameters(pd.DataFrame())


def test_estimate_return_parameters_annualize(asset_returns_3a):
    daily = estimate_return_parameters(asset_returns_3a, annualize=False)
    annual = estimate_return_parameters(
        asset_returns_3a, annualize=True, periods_per_year=365
    )
    assert annual["mean_vector"].iloc[0] == pytest.approx(
        daily["mean_vector"].iloc[0] * 365.0
    )


# ── simulate_normal_returns ─────────────────────────────────────────────────


def test_simulate_normal_returns_shape(asset_returns_3a):
    params = estimate_return_parameters(asset_returns_3a)
    out = simulate_normal_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        n_scenarios=2000,
        horizon_days=1,
        random_seed=1,
    )
    assert out.shape == (2000, 3)
    assert list(out.columns) == ["BTC", "ETH", "SOL"]
    assert out.index[0] == "scenario_1"


def test_simulate_normal_returns_reproducible(asset_returns_3a):
    params = estimate_return_parameters(asset_returns_3a)
    a = simulate_normal_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        n_scenarios=1000,
        random_seed=123,
    )
    b = simulate_normal_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        n_scenarios=1000,
        random_seed=123,
    )
    pd.testing.assert_frame_equal(a, b)


def test_simulate_normal_returns_horizon_scaling(asset_returns_3a):
    """Horizon=10 simulation should have larger dispersion than horizon=1."""
    params = estimate_return_parameters(asset_returns_3a)
    one_day = simulate_normal_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        n_scenarios=5000,
        horizon_days=1,
        random_seed=42,
    )
    ten_day = simulate_normal_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        n_scenarios=5000,
        horizon_days=10,
        random_seed=42,
    )
    assert float(ten_day.std(ddof=1).iloc[0]) > float(one_day.std(ddof=1).iloc[0])


# ── simulate_student_t_returns ──────────────────────────────────────────────


def test_simulate_student_t_returns_shape(asset_returns_3a):
    params = estimate_return_parameters(asset_returns_3a)
    out = simulate_student_t_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        df=5,
        n_scenarios=2000,
        random_seed=1,
    )
    assert out.shape == (2000, 3)


def test_simulate_student_t_invalid_df(asset_returns_3a):
    params = estimate_return_parameters(asset_returns_3a)
    with pytest.raises(ValueError):
        simulate_student_t_returns(
            mean_vector=params["mean_vector"],
            covariance_matrix=params["covariance_matrix"],
            df=2.0,
        )
    with pytest.raises(ValueError):
        simulate_student_t_returns(
            mean_vector=params["mean_vector"],
            covariance_matrix=params["covariance_matrix"],
            df=1.0,
        )


# ── calculate_portfolio_scenario_returns ────────────────────────────────────


def test_calculate_portfolio_scenario_returns_length(asset_returns_3a, weights_3a):
    params = estimate_return_parameters(asset_returns_3a)
    scen = simulate_normal_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        n_scenarios=1500,
        random_seed=5,
    )
    pf = calculate_portfolio_scenario_returns(scen, weights_3a)
    assert len(pf) == 1500


def test_calculate_portfolio_scenario_returns_weighted_sum():
    scen = pd.DataFrame(
        {"A": [0.10, -0.05, 0.00], "B": [0.20, 0.10, -0.30]},
        index=["s1", "s2", "s3"],
    )
    weights = pd.Series({"A": 0.4, "B": 0.6})
    pf = calculate_portfolio_scenario_returns(scen, weights)
    expected = scen["A"] * 0.4 + scen["B"] * 0.6
    pd.testing.assert_series_equal(
        pf.rename("expected").astype(float),
        expected.rename("expected").astype(float),
        check_names=False,
    )


# ── scenario_var / scenario_cvar ────────────────────────────────────────────


def test_scenario_var_positive_loss(asset_returns_3a, weights_3a):
    params = estimate_return_parameters(asset_returns_3a)
    scen = simulate_normal_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        n_scenarios=5000,
        random_seed=99,
    )
    pf = calculate_portfolio_scenario_returns(scen, weights_3a)
    var = scenario_var(pf, confidence_level=0.95)
    assert var > 0


def test_scenario_cvar_positive_and_ge_var(asset_returns_3a, weights_3a):
    params = estimate_return_parameters(asset_returns_3a)
    scen = simulate_normal_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        n_scenarios=5000,
        random_seed=99,
    )
    pf = calculate_portfolio_scenario_returns(scen, weights_3a)
    var = scenario_var(pf, confidence_level=0.95)
    cvar = scenario_cvar(pf, confidence_level=0.95)
    assert cvar > 0
    assert cvar >= var - 1e-12


# ── monte_carlo_risk_summary ────────────────────────────────────────────────


def test_monte_carlo_risk_summary_contains_var_cvar(asset_returns_3a, weights_3a):
    params = estimate_return_parameters(asset_returns_3a)
    scen = simulate_normal_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        n_scenarios=3000,
        random_seed=11,
    )
    pf = calculate_portfolio_scenario_returns(scen, weights_3a)
    summary = monte_carlo_risk_summary(
        pf, confidence_level=0.95, initial_capital=100_000, label="Normal MC"
    )
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == ["Metric", "Value", "Unit"]
    assert any("VaR" in str(v) for v in summary["Metric"])
    assert any("CVaR" in str(v) for v in summary["Metric"])
    assert any("Money VaR" in str(v) for v in summary["Metric"])


# ── simulate_portfolio_paths ────────────────────────────────────────────────


def test_simulate_portfolio_paths_initial_value():
    paths = simulate_portfolio_paths(
        portfolio_daily_mean=0.001,
        portfolio_daily_volatility=0.02,
        initial_value=100_000.0,
        n_paths=100,
        horizon_days=30,
        distribution="normal",
        random_seed=42,
    )
    assert paths.shape == (31, 100)
    assert (paths.iloc[0] == 100_000.0).all()


def test_simulate_portfolio_paths_student_t_shape():
    paths = simulate_portfolio_paths(
        portfolio_daily_mean=0.0,
        portfolio_daily_volatility=0.015,
        initial_value=50_000.0,
        n_paths=200,
        horizon_days=20,
        distribution="student_t",
        df=5,
        random_seed=0,
    )
    assert paths.shape == (21, 200)
    assert (paths.iloc[0] == 50_000.0).all()


def test_simulate_portfolio_paths_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        simulate_portfolio_paths(
            portfolio_daily_mean=0.0,
            portfolio_daily_volatility=0.01,
            initial_value=-1.0,
        )
    with pytest.raises(ValueError):
        simulate_portfolio_paths(
            portfolio_daily_mean=0.0,
            portfolio_daily_volatility=0.01,
            horizon_days=0,
        )
    with pytest.raises(ValueError):
        simulate_portfolio_paths(
            portfolio_daily_mean=0.0,
            portfolio_daily_volatility=0.01,
            distribution="bogus",
        )


# ── compare_monte_carlo_distributions ───────────────────────────────────────


def test_compare_monte_carlo_distributions_outputs(asset_returns_3a, weights_3a):
    scenarios, comparison = compare_monte_carlo_distributions(
        returns=asset_returns_3a,
        weights=weights_3a,
        confidence_level=0.95,
        n_scenarios=3000,
        horizon_days=1,
        df=5,
        random_seed=2025,
    )
    assert "normal" in scenarios and "student_t" in scenarios
    assert isinstance(comparison, pd.DataFrame)
    assert set(comparison["distribution"]) == {"normal", "student_t"}
    assert {"VaR", "CVaR", "mean_return", "volatility"}.issubset(comparison.columns)
    assert (comparison["VaR"] > 0).all()
    assert (comparison["CVaR"] >= comparison["VaR"] - 1e-12).all()


# ── compare_all_risk_methods ────────────────────────────────────────────────


def test_compare_all_risk_methods_methods_present(
    portfolio_returns_simple, asset_returns_3a, weights_3a
):
    comparison = compare_all_risk_methods(
        portfolio_returns=portfolio_returns_simple,
        asset_returns=asset_returns_3a,
        weights=weights_3a,
        confidence_level=0.95,
        horizon_days=1,
        n_scenarios=3000,
        student_t_df=5,
        random_seed=7,
    )
    methods = set(comparison["Method"].tolist())
    assert {
        "Historical",
        "Gaussian",
        "Cornish-Fisher",
        "Normal Monte Carlo",
        "Student-t Monte Carlo",
    }.issubset(methods)


def test_compare_all_risk_methods_horizon_in_output(
    portfolio_returns_simple, asset_returns_3a, weights_3a
):
    comparison = compare_all_risk_methods(
        portfolio_returns=portfolio_returns_simple,
        asset_returns=asset_returns_3a,
        weights=weights_3a,
        confidence_level=0.95,
        horizon_days=7,
        n_scenarios=2000,
        random_seed=7,
    )
    assert "Horizon Days" in comparison.columns
    assert (comparison["Horizon Days"] == 7).all()


# ── import isolation ────────────────────────────────────────────────────────


def test_monte_carlo_does_not_import_streamlit():
    """The monte_carlo module must not depend on streamlit."""
    import sys

    # Reload to make sure baseline is clean.
    sys.modules.pop("streamlit", None)
    import importlib

    import var_cvar_crypto_risk.monte_carlo as mc  # noqa: F401

    importlib.reload(mc)
    assert "streamlit" not in sys.modules


def test_backtesting_does_not_import_yfinance():
    """The backtesting module must not depend on yfinance."""
    import sys

    sys.modules.pop("yfinance", None)
    import importlib

    import var_cvar_crypto_risk.backtesting as bt  # noqa: F401

    importlib.reload(bt)
    assert "yfinance" not in sys.modules
