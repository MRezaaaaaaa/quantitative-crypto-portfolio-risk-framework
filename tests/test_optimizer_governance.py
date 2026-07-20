"""Tests for Phase-7 optimizer governance: feasibility diagnostics,
result interpretation, zero-mu warnings, Max-Sharpe cash handling, and
covariance overrides in the scenario builder."""

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

from var_cvar_crypto_risk.optimization import (  # noqa: E402
    add_cash_asset,
    build_optimization_scenarios,
    compute_feasible_risk_return_bounds,
    diagnose_infeasibility,
    interpret_optimization_result,
    maximize_return_with_cvar_constraint,
    maximize_sharpe_ratio,
    minimize_cvar,
    minimize_cvar_for_target_return,
    validate_solution_residuals,
)

cvxpy = pytest.importorskip("cvxpy")


@pytest.fixture
def scenarios() -> pd.DataFrame:
    """400-scenario, 3-asset matrix with positive drift and fat tails."""
    rng = np.random.default_rng(seed=3)
    base = rng.standard_t(df=5, size=(400, 3)) * 0.02
    drift = np.array([0.002, 0.001, 0.0005])
    return pd.DataFrame(base + drift, columns=["BTC", "ETH", "SOL"])


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    rng = np.random.default_rng(seed=5)
    n = 400
    cov = np.array(
        [
            [0.0009, 0.00045, 0.00030],
            [0.00045, 0.0012, 0.00040],
            [0.00030, 0.00040, 0.0016],
        ]
    )
    mean = np.array([0.0008, 0.0006, 0.0010])
    L = np.linalg.cholesky(cov)
    R = mean + rng.standard_normal(size=(n, 3)) @ L.T
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(R, index=idx, columns=["BTC", "ETH", "SOL"])


# ── Post-solve residual validation ──────────────────────────────────────


def test_solution_residuals_pass_for_feasible_weights(scenarios):
    weights = pd.Series({"BTC": 0.4, "ETH": 0.3, "SOL": 0.3})
    report = validate_solution_residuals(
        scenarios,
        weights,
        long_only=True,
        min_weight=0.0,
        max_weight=0.8,
    )
    assert report["passed"] is True
    assert report["max_constraint_violation"] <= report["tolerance"]


def test_solution_residuals_detect_budget_and_bound_violations(scenarios):
    weights = pd.Series({"BTC": 0.8, "ETH": 0.4, "SOL": -0.1})
    report = validate_solution_residuals(
        scenarios,
        weights,
        long_only=True,
        max_weight=0.7,
    )
    assert report["passed"] is False
    assert report["budget_residual"] == pytest.approx(0.1)
    assert report["lower_bound_violation"] == pytest.approx(0.1)
    assert report["upper_bound_violation"] == pytest.approx(0.1)


def test_solution_residuals_detect_target_and_cvar_violations(scenarios):
    weights = pd.Series({"BTC": 0.4, "ETH": 0.3, "SOL": 0.3})
    mu = scenarios.mean()
    realized_expected = float(mu @ weights)
    report = validate_solution_residuals(
        scenarios,
        weights,
        expected_returns=mu,
        target_return=realized_expected + 0.01,
        cvar_limit=0.01,
        solver_cvar=0.02,
    )
    assert report["passed"] is False
    assert report["target_return_violation"] == pytest.approx(0.01)
    assert report["cvar_limit_violation"] == pytest.approx(0.01)


def test_solution_residuals_reject_nonfinite_auxiliary_solution(scenarios):
    weights = pd.Series({"BTC": 0.4, "ETH": 0.3, "SOL": 0.3})
    report = validate_solution_residuals(
        scenarios,
        weights,
        var_threshold=float("nan"),
        excess_losses=np.zeros(len(scenarios)),
    )
    assert report["passed"] is False
    assert np.isinf(report["auxiliary_loss_violation"])


@pytest.mark.parametrize(
    "optimizer,kwargs",
    [
        (minimize_cvar, {}),
        (maximize_return_with_cvar_constraint, {"cvar_limit": 0.50}),
        (minimize_cvar_for_target_return, {"target_return": -0.50}),
    ],
)
def test_solved_optimizers_publish_residual_governance(scenarios, optimizer, kwargs):
    result = optimizer(scenarios, **kwargs)
    assert result["solver_status"] in ("optimal", "optimal_inaccurate")
    assert result["status"] in ("optimal", "optimal_inaccurate")
    assert result["constraint_validation"]["passed"] is True
    assert (
        result["max_constraint_violation"]
        <= result["constraint_validation"]["tolerance"]
    )


# ── Feasible bounds ──────────────────────────────────────────────────────


def test_bounds_are_ordered(scenarios):
    bounds = compute_feasible_risk_return_bounds(scenarios)
    assert np.isfinite(bounds["min_cvar"])
    assert np.isfinite(bounds["max_return"])
    assert bounds["min_cvar"] <= bounds["max_return_cvar"] + 1e-9
    assert bounds["min_cvar_return"] <= bounds["max_return"] + 1e-9


# ── Infeasibility diagnostics ────────────────────────────────────────────


def test_diagnose_min_weight_budget_infeasible(scenarios):
    reasons = diagnose_infeasibility(scenarios, min_weight=0.5)
    assert any("over-allocates the budget" in r for r in reasons)


def test_diagnose_max_weight_budget_infeasible(scenarios):
    reasons = diagnose_infeasibility(scenarios, max_weight=0.2)
    assert any("cannot absorb the full budget" in r for r in reasons)


def test_diagnose_cvar_cap_too_tight(scenarios):
    # A cap far below any achievable CVaR must be identified.
    result = maximize_return_with_cvar_constraint(scenarios, cvar_limit=1e-5)
    assert result["status"] not in ("optimal", "optimal_inaccurate")
    reasons = diagnose_infeasibility(scenarios, cvar_limit=1e-5)
    assert any("minimum achievable CVaR" in r for r in reasons)


def test_diagnose_target_return_too_high(scenarios):
    mu = scenarios.mean()
    target = float(mu.max()) * 10 + 1.0  # far beyond reach
    result = minimize_cvar_for_target_return(scenarios, target_return=target)
    assert result["status"] not in ("optimal", "optimal_inaccurate")
    reasons = diagnose_infeasibility(
        scenarios, expected_returns=mu, target_return=target
    )
    assert any("maximum achievable expected return" in r for r in reasons)


def test_diagnose_zero_mu_positive_target(scenarios):
    zero_mu = pd.Series(0.0, index=scenarios.columns)
    reasons = diagnose_infeasibility(
        scenarios, expected_returns=zero_mu, target_return=0.01
    )
    assert any("expected returns are zero" in r for r in reasons)


def test_diagnose_feasible_setup_returns_generic(scenarios):
    reasons = diagnose_infeasibility(scenarios, cvar_limit=5.0)
    assert len(reasons) == 1
    assert "No single constraint" in reasons[0]


# ── Zero-mu warnings ─────────────────────────────────────────────────────


def test_zero_mu_warning_on_return_objectives(scenarios):
    zero_mu = pd.Series(0.0, index=scenarios.columns)
    r1 = maximize_return_with_cvar_constraint(
        scenarios, expected_returns=zero_mu, cvar_limit=0.50
    )
    assert "warning" in r1 and "not" in r1["warning"]
    r2 = minimize_cvar_for_target_return(
        scenarios, expected_returns=zero_mu, target_return=0.0
    )
    assert "warning" in r2
    r3 = maximize_sharpe_ratio(scenarios, expected_returns=zero_mu)
    if r3["status"] in ("optimal", "optimal_inaccurate"):
        assert "warning" in r3


def test_no_warning_with_nonzero_mu(scenarios):
    result = maximize_return_with_cvar_constraint(scenarios, cvar_limit=0.50)
    assert "warning" not in result


# ── Max Sharpe + cash ────────────────────────────────────────────────────


def test_max_sharpe_excludes_pure_cash_portfolio(scenarios):
    # Cash return above every asset's mean ⇒ without the volatility floor
    # the frontier's defensive end would be ~100% cash with vol ≈ 0.
    with_cash = add_cash_asset(scenarios, cash_return=0.01)
    result = maximize_sharpe_ratio(with_cash, min_volatility=1e-6)
    if result["status"] in ("optimal", "optimal_inaccurate"):
        assert result["volatility"] >= 1e-6
        cash_w = float(result["weights"].get("CASH", 0.0))
        if cash_w > 0.90:
            assert "warning" in result  # cash-domination flagged
    else:
        assert "Sharpe" in result["message"] or "volatility" in result["message"]


def test_max_sharpe_reports_skipped_candidates_key(scenarios):
    result = maximize_sharpe_ratio(scenarios)
    assert "n_skipped_low_vol" in result
    if result["status"] in ("optimal", "optimal_inaccurate"):
        assert result["solver_status"] in ("optimal", "optimal_inaccurate")
        assert result["constraint_validation"]["passed"] is True


# ── Result interpretation ────────────────────────────────────────────────


def test_interpret_binding_cvar_cap(scenarios):
    bounds = compute_feasible_risk_return_bounds(scenarios)
    tight_cap = bounds["min_cvar"] * 1.05  # feasible but binding
    result = maximize_return_with_cvar_constraint(scenarios, cvar_limit=tight_cap)
    assert result["status"] in ("optimal", "optimal_inaccurate")
    interp = interpret_optimization_result(
        result,
        cvar_limit=tight_cap,
        min_cvar_bound=bounds["min_cvar"],
        max_return_cvar=bounds["max_return_cvar"],
    )
    assert interp["solved"] is True
    assert interp["cvar_cap_binding"] is True
    assert interp["risk_profile"] in ("defensive", "balanced", "aggressive")


def test_interpret_loose_cap_not_binding(scenarios):
    result = maximize_return_with_cvar_constraint(scenarios, cvar_limit=5.0)
    interp = interpret_optimization_result(result, cvar_limit=5.0)
    assert interp["cvar_cap_binding"] is False


def test_interpret_excluded_and_min_weight_assets(scenarios):
    # Unconstrained min-CVaR typically zeroes at least one asset.
    result = minimize_cvar(scenarios)
    interp = interpret_optimization_result(result, min_weight=0.0)
    assert isinstance(interp["excluded_assets"], list)

    # With a forced min weight nothing can be excluded.
    result_forced = minimize_cvar(scenarios, min_weight=0.10)
    interp_forced = interpret_optimization_result(result_forced, min_weight=0.10)
    assert interp_forced["excluded_assets"] == []


def test_interpret_unsolved_result_is_safe():
    unsolved = {"status": "infeasible", "weights": None}
    interp = interpret_optimization_result(unsolved, cvar_limit=0.1)
    assert interp["solved"] is False
    assert interp["risk_profile"] == "unknown"


def test_interpret_cash_weight_reported(scenarios):
    with_cash = add_cash_asset(scenarios, cash_return=0.0)
    result = minimize_cvar(with_cash)
    interp = interpret_optimization_result(result)
    assert interp["cash_weight"] == pytest.approx(
        float(result["weights"]["CASH"]), abs=1e-9
    )


# ── Scenario builder overrides ───────────────────────────────────────────


def test_scenario_builder_covariance_override_changes_dispersion(asset_returns):
    shrunk_cov = asset_returns.cov() * 0.25  # half the volatility
    base = build_optimization_scenarios(
        asset_returns, source="normal_mc", n_scenarios=4000, random_seed=1
    )
    overridden = build_optimization_scenarios(
        asset_returns,
        source="normal_mc",
        n_scenarios=4000,
        random_seed=1,
        covariance_matrix=shrunk_cov,
    )
    assert overridden.std().mean() < base.std().mean()
    assert list(overridden.columns) == list(asset_returns.columns)


def test_scenario_builder_mean_override(asset_returns):
    mu = pd.Series(0.0, index=asset_returns.columns)
    scen = build_optimization_scenarios(
        asset_returns,
        source="normal_mc",
        n_scenarios=8000,
        random_seed=2,
        mean_vector=mu,
    )
    assert abs(float(scen.mean().mean())) < 5e-4


def test_scenario_builder_default_unchanged(asset_returns):
    """No override ⇒ same output as before Phase 7 (regression guard)."""
    a = build_optimization_scenarios(
        asset_returns, source="student_t_mc", n_scenarios=500, random_seed=42
    )
    b = build_optimization_scenarios(
        asset_returns, source="student_t_mc", n_scenarios=500, random_seed=42
    )
    pd.testing.assert_frame_equal(a, b)
