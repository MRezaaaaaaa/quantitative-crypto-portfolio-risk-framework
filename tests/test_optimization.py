"""Tests for the Phase-5 CVaR portfolio optimization module."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def small_scenarios() -> pd.DataFrame:
    """200-scenario, 3-asset matrix with mild positive drift and fat tails."""
    rng = np.random.default_rng(seed=11)
    n_scen = 200
    n_assets = 3
    base = rng.standard_t(df=5, size=(n_scen, n_assets)) * 0.02
    drift = np.array([0.001, 0.0008, 0.0005])
    R = base + drift
    return pd.DataFrame(R, columns=["BTC", "ETH", "SOL"])


@pytest.fixture
def asset_returns_500() -> pd.DataFrame:
    """500 days of synthetic asset returns for build_optimization_scenarios."""
    rng = np.random.default_rng(seed=42)
    n = 500
    cov = np.array(
        [
            [0.0009, 0.00045, 0.00030],
            [0.00045, 0.0012, 0.00040],
            [0.00030, 0.00040, 0.0016],
        ]
    )
    mean = np.array([0.0008, 0.0006, 0.0010])
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal(size=(n, 3))
    R = mean + Z @ L.T
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(R, index=idx, columns=["BTC", "ETH", "SOL"])


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_scenario_matrix_valid(small_scenarios):
    from var_cvar_crypto_risk.optimization import validate_scenario_matrix

    # No exception expected.
    validate_scenario_matrix(small_scenarios)


def test_validate_scenario_matrix_invalid():
    from var_cvar_crypto_risk.optimization import validate_scenario_matrix

    with pytest.raises(ValueError):
        validate_scenario_matrix(pd.DataFrame())

    bad = pd.DataFrame({"A": ["x", "y", "z"], "B": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        validate_scenario_matrix(bad)

    with pytest.raises(ValueError):
        validate_scenario_matrix(pd.DataFrame({"A": [1.0]}))


def test_validate_scenario_matrix_rejects_non_dataframe():
    from var_cvar_crypto_risk.optimization import validate_scenario_matrix

    with pytest.raises(ValueError):
        validate_scenario_matrix([[1.0, 2.0], [3.0, 4.0]])


# ─────────────────────────────────────────────────────────────────────────────
# Cash asset
# ─────────────────────────────────────────────────────────────────────────────


def test_add_cash_asset(small_scenarios):
    from var_cvar_crypto_risk.optimization import add_cash_asset

    original_cols = list(small_scenarios.columns)
    augmented = add_cash_asset(small_scenarios, cash_return=0.0001)
    assert "CASH" in augmented.columns
    assert (augmented["CASH"] == 0.0001).all()
    # Original unchanged
    assert list(small_scenarios.columns) == original_cols


def test_add_cash_asset_duplicate_raises(small_scenarios):
    from var_cvar_crypto_risk.optimization import add_cash_asset

    aug = add_cash_asset(small_scenarios, cash_return=0.0)
    with pytest.raises(ValueError):
        add_cash_asset(aug, cash_return=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Expected returns
# ─────────────────────────────────────────────────────────────────────────────


def test_estimate_expected_returns(small_scenarios):
    from var_cvar_crypto_risk.optimization import estimate_expected_returns

    er_mean = estimate_expected_returns(small_scenarios, method="mean")
    assert isinstance(er_mean, pd.Series)
    assert list(er_mean.index) == list(small_scenarios.columns)
    np.testing.assert_allclose(
        er_mean.values, small_scenarios.mean().values, atol=1e-10
    )

    er_zero = estimate_expected_returns(small_scenarios, method="zero")
    assert (er_zero == 0).all()


def test_estimate_expected_returns_bad_method(small_scenarios):
    from var_cvar_crypto_risk.optimization import estimate_expected_returns

    with pytest.raises(ValueError):
        estimate_expected_returns(small_scenarios, method="nonsense")


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio scenario metrics
# ─────────────────────────────────────────────────────────────────────────────


def test_calculate_portfolio_scenario_metrics(small_scenarios):
    from var_cvar_crypto_risk.optimization import calculate_portfolio_scenario_metrics

    weights = pd.Series({"BTC": 0.4, "ETH": 0.4, "SOL": 0.2})
    metrics = calculate_portfolio_scenario_metrics(
        small_scenarios, weights, confidence_level=0.95, initial_capital=100_000.0
    )
    for key in (
        "expected_return",
        "volatility",
        "VaR",
        "CVaR",
        "worst_return",
        "best_return",
        "money_VaR",
        "money_CVaR",
    ):
        assert key in metrics
    assert metrics["VaR"] > 0
    assert metrics["CVaR"] >= metrics["VaR"]
    assert metrics["money_VaR"] == pytest.approx(metrics["VaR"] * 100_000.0)


# ─────────────────────────────────────────────────────────────────────────────
# Minimize CVaR
# ─────────────────────────────────────────────────────────────────────────────


def test_minimize_cvar_basic(small_scenarios):
    from var_cvar_crypto_risk.optimization import minimize_cvar

    result = minimize_cvar(small_scenarios, confidence_level=0.95)
    assert "status" in result
    assert result["status"] in ("optimal", "optimal_inaccurate")
    weights = result["weights"]
    assert isinstance(weights, pd.Series)
    assert weights.sum() == pytest.approx(1.0, abs=1e-4)
    assert (weights >= -1e-7).all()  # long-only
    assert np.isfinite(result["CVaR"])
    assert result["CVaR"] > 0


def test_minimize_cvar_max_weight_constraint(small_scenarios):
    from var_cvar_crypto_risk.optimization import minimize_cvar

    result = minimize_cvar(
        small_scenarios, confidence_level=0.95, max_weight=0.5
    )
    assert result["status"] in ("optimal", "optimal_inaccurate")
    weights = result["weights"]
    assert (weights <= 0.5 + 1e-6).all()


def test_minimize_cvar_with_cash(small_scenarios):
    from var_cvar_crypto_risk.optimization import minimize_cvar

    result = minimize_cvar(
        small_scenarios, confidence_level=0.95, include_cash=True
    )
    assert result["status"] in ("optimal", "optimal_inaccurate")
    assert "CASH" in result["weights"].index


# ─────────────────────────────────────────────────────────────────────────────
# Max return under CVaR cap
# ─────────────────────────────────────────────────────────────────────────────


def test_maximize_return_with_cvar_constraint_basic(small_scenarios):
    from var_cvar_crypto_risk.optimization import (
        maximize_return_with_cvar_constraint,
    )

    # Pick a generous CVaR cap so the problem is solvable.
    result = maximize_return_with_cvar_constraint(
        small_scenarios, cvar_limit=0.20, confidence_level=0.95
    )
    assert result["status"] in ("optimal", "optimal_inaccurate")
    # CVaR shouldn't exceed cap by more than a small numerical tolerance.
    assert result["CVaR"] <= 0.20 + 1e-3
    weights = result["weights"]
    assert weights.sum() == pytest.approx(1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Min CVaR for target return
# ─────────────────────────────────────────────────────────────────────────────


def test_minimize_cvar_for_target_return_basic(small_scenarios):
    from var_cvar_crypto_risk.optimization import (
        estimate_expected_returns,
        minimize_cvar_for_target_return,
    )

    er = estimate_expected_returns(small_scenarios)
    target = float(er.min())  # easy to satisfy
    result = minimize_cvar_for_target_return(
        small_scenarios,
        expected_returns=er,
        target_return=target,
        confidence_level=0.95,
    )
    assert result["status"] in ("optimal", "optimal_inaccurate")
    assert result["expected_return"] >= target - 1e-6


def test_infeasible_target_return(small_scenarios):
    from var_cvar_crypto_risk.optimization import minimize_cvar_for_target_return

    # Target hugely above any feasible portfolio return.
    result = minimize_cvar_for_target_return(
        small_scenarios, target_return=10.0, confidence_level=0.95
    )
    assert result["status"] not in ("optimal", "optimal_inaccurate")
    assert "message" in result
    assert isinstance(result["message"], str)


# ─────────────────────────────────────────────────────────────────────────────
# Efficient frontier
# ─────────────────────────────────────────────────────────────────────────────


def test_generate_cvar_efficient_frontier(small_scenarios):
    from var_cvar_crypto_risk.optimization import generate_cvar_efficient_frontier

    frontier = generate_cvar_efficient_frontier(
        small_scenarios, n_points=8, confidence_level=0.95
    )
    assert isinstance(frontier, pd.DataFrame)
    assert "CVaR" in frontier.columns
    assert "expected_return" in frontier.columns
    weight_cols = [c for c in frontier.columns if c.startswith("weight_")]
    assert weight_cols, "expected weight_<asset> columns"
    # Must include weight column per asset.
    for asset in small_scenarios.columns:
        assert f"weight_{asset}" in frontier.columns


# ─────────────────────────────────────────────────────────────────────────────
# Current vs optimized comparison
# ─────────────────────────────────────────────────────────────────────────────


def test_compare_current_vs_optimized(small_scenarios):
    from var_cvar_crypto_risk.optimization import (
        compare_current_vs_optimized,
        minimize_cvar,
    )

    current = pd.Series({"BTC": 0.5, "ETH": 0.3, "SOL": 0.2})
    min_cvar = minimize_cvar(small_scenarios, confidence_level=0.95)
    comp = compare_current_vs_optimized(
        small_scenarios,
        current,
        {"Min CVaR": min_cvar},
        confidence_level=0.95,
        initial_capital=100_000.0,
    )
    assert isinstance(comp, pd.DataFrame)
    assert "Current" in comp["Portfolio"].tolist()
    assert "Min CVaR" in comp["Portfolio"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario builders
# ─────────────────────────────────────────────────────────────────────────────


def test_build_optimization_scenarios_historical(asset_returns_500):
    from var_cvar_crypto_risk.optimization import build_optimization_scenarios

    out_h1 = build_optimization_scenarios(asset_returns_500, source="historical")
    assert list(out_h1.columns) == list(asset_returns_500.columns)
    assert len(out_h1) == len(asset_returns_500.dropna())

    out_h7 = build_optimization_scenarios(
        asset_returns_500, source="historical", horizon_days=7
    )
    assert len(out_h7) < len(out_h1)
    assert list(out_h7.columns) == list(asset_returns_500.columns)


def test_build_optimization_scenarios_normal_mc(asset_returns_500):
    from var_cvar_crypto_risk.optimization import build_optimization_scenarios

    out = build_optimization_scenarios(
        asset_returns_500, source="normal_mc", n_scenarios=500
    )
    assert out.shape == (500, 3)
    assert list(out.columns) == list(asset_returns_500.columns)


def test_build_optimization_scenarios_student_t_mc(asset_returns_500):
    from var_cvar_crypto_risk.optimization import build_optimization_scenarios

    out = build_optimization_scenarios(
        asset_returns_500, source="student_t_mc", n_scenarios=400, student_t_df=5
    )
    assert out.shape == (400, 3)


def test_build_optimization_scenarios_bad_source(asset_returns_500):
    from var_cvar_crypto_risk.optimization import build_optimization_scenarios

    with pytest.raises(ValueError):
        build_optimization_scenarios(asset_returns_500, source="bogus")


# ─────────────────────────────────────────────────────────────────────────────
# Format weights table
# ─────────────────────────────────────────────────────────────────────────────


def test_format_weights_table():
    from var_cvar_crypto_risk.optimization import format_weights_table

    weights = pd.Series({"BTC": 0.2, "ETH": 0.5, "SOL": 0.3})
    table = format_weights_table(weights)
    assert list(table.columns) == ["Asset", "Weight"]
    assert table["Asset"].iloc[0] == "ETH"  # sorted descending


# ─────────────────────────────────────────────────────────────────────────────
# Import isolation
# ─────────────────────────────────────────────────────────────────────────────


def test_optimization_import_without_streamlit():
    """``optimization.py`` must not pull in Streamlit."""
    # Force a fresh import.
    for mod_name in list(sys.modules):
        if mod_name.startswith("var_cvar_crypto_risk.optimization"):
            del sys.modules[mod_name]
    streamlit_was_loaded = "streamlit" in sys.modules
    importlib.import_module("var_cvar_crypto_risk.optimization")
    if not streamlit_was_loaded:
        assert "streamlit" not in sys.modules, (
            "optimization module must not import Streamlit"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5.5: shrinkage estimator + max-Sharpe portfolio
# ─────────────────────────────────────────────────────────────────────────────


def test_estimate_expected_returns_shrinkage(small_scenarios):
    from var_cvar_crypto_risk.optimization import estimate_expected_returns

    mean = estimate_expected_returns(small_scenarios, method="mean")
    shrunk = estimate_expected_returns(
        small_scenarios, method="shrinkage_to_zero", shrinkage_weight=0.5
    )
    pd.testing.assert_series_equal(shrunk, (0.5 * mean).rename("expected_return"))


def test_estimate_expected_returns_shrinkage_bad_weight(small_scenarios):
    from var_cvar_crypto_risk.optimization import estimate_expected_returns

    with pytest.raises(ValueError):
        estimate_expected_returns(
            small_scenarios, method="shrinkage_to_zero", shrinkage_weight=1.5
        )


def test_maximize_sharpe_weights_sum_to_one(small_scenarios):
    from var_cvar_crypto_risk.optimization import maximize_sharpe_ratio

    result = maximize_sharpe_ratio(
        small_scenarios, risk_free_rate=0.0, long_only=True, max_weight=1.0
    )
    assert result["status"] == "optimal"
    assert float(result["weights"].sum()) == pytest.approx(1.0, abs=1e-4)


def test_maximize_sharpe_output_has_sharpe_ratio(small_scenarios):
    from var_cvar_crypto_risk.optimization import maximize_sharpe_ratio

    result = maximize_sharpe_ratio(small_scenarios)
    assert "sharpe_ratio" in result
    assert np.isfinite(result["sharpe_ratio"])
    # Sharpe = (E[r] - rf) / vol consistency
    expected = result["expected_return"] / result["volatility"]
    assert result["sharpe_ratio"] == pytest.approx(expected, rel=1e-6)


def test_scenario_metrics_includes_sharpe(small_scenarios):
    from var_cvar_crypto_risk.optimization import (
        calculate_portfolio_scenario_metrics,
    )

    weights = pd.Series({"BTC": 0.4, "ETH": 0.3, "SOL": 0.3})
    m = calculate_portfolio_scenario_metrics(
        small_scenarios, weights, confidence_level=0.95, risk_free_rate=0.0
    )
    assert "sharpe_ratio" in m
    assert m["sharpe_ratio"] == pytest.approx(
        m["expected_return"] / m["volatility"], rel=1e-9
    )


def test_compare_current_vs_optimized_has_sharpe_column(small_scenarios):
    from var_cvar_crypto_risk.optimization import (
        compare_current_vs_optimized,
        minimize_cvar,
    )

    current = pd.Series({"BTC": 0.5, "ETH": 0.3, "SOL": 0.2})
    res = minimize_cvar(small_scenarios)
    table = compare_current_vs_optimized(
        small_scenarios, current, {"Min CVaR": res}, risk_free_rate=0.0
    )
    assert "Sharpe" in table.columns
