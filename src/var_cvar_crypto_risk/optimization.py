"""Scenario-based CVaR portfolio optimization.

This module is pure analytics: no plotting, no Streamlit, no I/O.

It implements the **Rockafellar-Uryasev** linear-programming formulation
of CVaR optimization on top of a scenario return matrix
``R ∈ R^{n_scenarios x n_assets}``. The scenario matrix can come from
historical returns or from the Monte Carlo scenario engine.

Conventions
-----------
* ``scenario_returns`` is a ``pandas.DataFrame`` with one **scenario per
  row** and one **asset per column**.
* Portfolio scenario return is ``R @ w``; scenario loss is ``-R @ w``.
* VaR and CVaR are reported as signed decimal loss values: positive = loss,
  zero = break-even, and negative = gain at the measured tail threshold.
* For confidence level ``β``, CVaR is

  .. math::

     \\text{CVaR}_β = t + \\frac{1}{(1-β)\\,N}\\,\\sum_i u_i,

  with auxiliaries ``u_i ≥ loss_i − t``, ``u_i ≥ 0``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .monte_carlo import (
    estimate_return_parameters,
    scenario_cvar,
    scenario_var,
    simulate_normal_returns,
    simulate_student_t_returns,
)
from .risk_conventions import loss_value_to_money


# Solver preference: keep open-source. ECOS is preferred when present (faster
# for small LPs) but it's no longer part of the default CVXPY install on
# modern wheels, so we fall back to CLARABEL → SCS.
_SOLVER_PREFERENCE: tuple[str, ...] = ("ECOS", "CLARABEL", "SCS")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _import_cvxpy():
    """Lazy import of CVXPY so that ``import optimization`` is cheap.

    Importing CVXPY at module load time can pull in heavy compiled
    dependencies. Optimization functions need it; validation/builder
    helpers do not.
    """
    try:
        import cvxpy as cp  # type: ignore
    except ImportError as exc:  # pragma: no cover - import-error path
        raise ImportError(
            "CVXPY is required for portfolio optimization. "
            "Install it with `pip install cvxpy`."
        ) from exc
    return cp


def _pick_solver(preferred: str | None = None) -> str:
    cp = _import_cvxpy()
    available = set(cp.installed_solvers())
    if preferred is not None:
        if preferred in available:
            return preferred
        # caller asked for something we don't have — fall through to default
    for name in _SOLVER_PREFERENCE:
        if name in available:
            return name
    # Last resort: let CVXPY pick.
    return next(iter(available)) if available else "SCS"


def _build_weight_constraints(
    w,
    n_assets: int,
    long_only: bool,
    min_weight: float | None,
    max_weight: float | None,
) -> list:
    """Build the standard portfolio weight constraints."""
    cp = _import_cvxpy()
    constraints = [cp.sum(w) == 1.0]

    if long_only:
        # Resolve effective lower bound. If the user passed a negative
        # min_weight together with long_only=True, force it to 0 so the
        # long-only invariant holds.
        lb = 0.0 if min_weight is None else max(0.0, float(min_weight))
        constraints.append(w >= lb)
    elif min_weight is not None:
        constraints.append(w >= float(min_weight))

    if max_weight is not None:
        constraints.append(w <= float(max_weight))

    # If max_weight is unbounded but a min_weight cap is provided that
    # makes the budget infeasible, CVXPY will surface that as infeasible
    # — we don't try to be clever here.
    _ = n_assets
    return constraints


def _solve(problem, solver: str | None, fallback: bool = True) -> str:
    """Solve a CVXPY problem with a preferred solver and graceful fallback.

    Returns the actual solver used (after fallback).
    """
    cp = _import_cvxpy()
    chosen = _pick_solver(solver)
    try:
        problem.solve(solver=chosen, verbose=False)
        return chosen
    except (cp.SolverError, Exception):  # noqa: BLE001
        if not fallback:
            raise
        for name in _SOLVER_PREFERENCE:
            if name == chosen:
                continue
            if name not in cp.installed_solvers():
                continue
            try:
                problem.solve(solver=name, verbose=False)
                return name
            except Exception:  # noqa: BLE001
                continue
        # Final attempt with default
        problem.solve(verbose=False)
        return "default"


def _status_message(status: str) -> str:
    return {
        "optimal": "Solved to optimality.",
        "optimal_inaccurate": "Solved (inaccurate).",
        "infeasible": "Problem is infeasible under the given constraints.",
        "infeasible_inaccurate": "Problem appears infeasible.",
        "unbounded": "Problem is unbounded.",
        "unbounded_inaccurate": "Problem appears unbounded.",
        "solver_error": "Solver error.",
        "user_limit": "Solver hit user limit before completion.",
    }.get(status, f"Solver returned: {status}.")


def _is_solved(status: str) -> bool:
    return status in ("optimal", "optimal_inaccurate")


def _zero_mu_warning(mu: np.ndarray) -> str | None:
    """Warning text when a return-based objective gets an all-zero mu."""
    if np.allclose(np.asarray(mu, dtype=float), 0.0, atol=1e-12):
        return (
            "All expected returns are zero — a return-based objective is not "
            "meaningful under this assumption. The solver only resolves "
            "constraint feasibility; the reported weights should not be "
            "interpreted as return-seeking."
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public: validation / scenario plumbing
# ─────────────────────────────────────────────────────────────────────────────


def validate_scenario_matrix(scenario_returns: pd.DataFrame) -> None:
    """Validate a scenario return matrix.

    Parameters
    ----------
    scenario_returns : pd.DataFrame
        Rows = scenarios, columns = assets. Values are simple returns
        (e.g. ``-0.05`` for a −5 % scenario).

    Raises
    ------
    ValueError
        If the input is not a non-empty numeric DataFrame, has fewer than
        two scenarios, zero assets, contains an all-NaN column, or any
        non-finite values.
    """
    if not isinstance(scenario_returns, pd.DataFrame):
        raise ValueError("scenario_returns must be a pandas DataFrame.")
    if scenario_returns.empty:
        raise ValueError("scenario_returns is empty.")
    if scenario_returns.shape[0] < 2:
        raise ValueError(f"Need at least 2 scenarios, got {scenario_returns.shape[0]}.")
    if scenario_returns.shape[1] < 1:
        raise ValueError("scenario_returns must contain at least one asset.")
    non_numeric = [
        col
        for col in scenario_returns.columns
        if not pd.api.types.is_numeric_dtype(scenario_returns[col])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric scenario columns: {non_numeric}")
    all_nan = scenario_returns.columns[scenario_returns.isna().all(axis=0)].tolist()
    if all_nan:
        raise ValueError(f"All-NaN scenario column(s): {all_nan}")
    if scenario_returns.isna().any().any():
        raise ValueError(
            "scenario_returns contains NaN values. Clean them with "
            "`scenario_returns.dropna()` before optimization."
        )
    if not np.isfinite(scenario_returns.to_numpy(dtype=float)).all():
        raise ValueError("scenario_returns contains non-finite values (inf/-inf).")


def validate_solution_residuals(
    scenario_returns: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float = 0.95,
    long_only: bool = True,
    min_weight: float | None = None,
    max_weight: float | None = 1.0,
    expected_returns: pd.Series | None = None,
    target_return: float | None = None,
    cvar_limit: float | None = None,
    solver_cvar: float | None = None,
    var_threshold: float | None = None,
    excess_losses: np.ndarray | None = None,
    tolerance: float = 1e-5,
) -> dict:
    """Recompute primal feasibility residuals after an optimizer solve.

    Solver status is only a numerical claim. This function independently
    checks the budget, effective weight box, optional target return and CVaR
    cap, plus the Rockafellar-Uryasev auxiliary constraints when their solved
    values are supplied.
    """
    validate_scenario_matrix(scenario_returns)
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")
    if not isinstance(weights, pd.Series):
        raise ValueError("weights must be a pandas Series.")
    aligned = weights.reindex(scenario_returns.columns)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index.tolist()
        raise ValueError(f"weights missing or non-finite for assets: {missing}")

    w = aligned.to_numpy(dtype=float)
    finite_weights = bool(np.isfinite(w).all())
    budget_residual = abs(float(w.sum()) - 1.0) if finite_weights else float("inf")

    effective_lower_bound: float | None
    if long_only:
        effective_lower_bound = max(
            0.0,
            float(min_weight) if min_weight is not None else 0.0,
        )
    else:
        effective_lower_bound = float(min_weight) if min_weight is not None else None
    lower_bound_violation = (
        max(float(effective_lower_bound - np.min(w)), 0.0)
        if finite_weights and effective_lower_bound is not None
        else (0.0 if finite_weights else float("inf"))
    )
    upper_bound_violation = (
        max(float(np.max(w) - float(max_weight)), 0.0)
        if finite_weights and max_weight is not None
        else (0.0 if finite_weights else float("inf"))
    )

    target_return_violation = 0.0
    achieved_expected_return = float("nan")
    if target_return is not None:
        mu = (
            scenario_returns.mean()
            if expected_returns is None
            else expected_returns.reindex(scenario_returns.columns)
        )
        if mu.isna().any() or not np.isfinite(mu.to_numpy(dtype=float)).all():
            raise ValueError("expected_returns must be finite and cover every asset.")
        achieved_expected_return = float(mu.to_numpy(dtype=float) @ w)
        target_return_violation = max(
            float(target_return) - achieved_expected_return,
            0.0,
        )

    evaluated_cvar = float("nan")
    cvar_limit_violation = 0.0
    if cvar_limit is not None:
        if solver_cvar is None:
            portfolio_returns = pd.Series(scenario_returns.to_numpy(dtype=float) @ w)
            evaluated_cvar = float(scenario_cvar(portfolio_returns, confidence_level))
        else:
            evaluated_cvar = float(solver_cvar)
        cvar_limit_violation = (
            max(evaluated_cvar - float(cvar_limit), 0.0)
            if np.isfinite(evaluated_cvar)
            else float("inf")
        )

    auxiliary_nonnegative_violation = 0.0
    auxiliary_loss_violation = 0.0
    if (var_threshold is None) != (excess_losses is None):
        raise ValueError("var_threshold and excess_losses must be supplied together.")
    if var_threshold is not None and excess_losses is not None:
        u = np.asarray(excess_losses, dtype=float).reshape(-1)
        if (
            not np.isfinite(float(var_threshold))
            or u.size != len(scenario_returns)
            or not np.isfinite(u).all()
        ):
            auxiliary_nonnegative_violation = float("inf")
            auxiliary_loss_violation = float("inf")
        else:
            losses = -(scenario_returns.to_numpy(dtype=float) @ w)
            auxiliary_nonnegative_violation = max(float(-np.min(u)), 0.0)
            auxiliary_loss_violation = max(
                float(np.max(losses - float(var_threshold) - u)),
                0.0,
            )

    residual_values = [
        budget_residual,
        lower_bound_violation,
        upper_bound_violation,
        target_return_violation,
        cvar_limit_violation,
        auxiliary_nonnegative_violation,
        auxiliary_loss_violation,
    ]
    max_violation = float(max(residual_values))
    passed = bool(finite_weights and max_violation <= float(tolerance))

    return {
        "passed": passed,
        "status": "passed" if passed else "failed",
        "tolerance": float(tolerance),
        "finite_weights": finite_weights,
        "budget_residual": budget_residual,
        "effective_lower_bound": effective_lower_bound,
        "effective_upper_bound": (
            float(max_weight) if max_weight is not None else None
        ),
        "lower_bound_violation": lower_bound_violation,
        "upper_bound_violation": upper_bound_violation,
        "achieved_expected_return": achieved_expected_return,
        "target_return_violation": target_return_violation,
        "evaluated_cvar": evaluated_cvar,
        "cvar_limit_violation": cvar_limit_violation,
        "auxiliary_nonnegative_violation": auxiliary_nonnegative_violation,
        "auxiliary_loss_violation": auxiliary_loss_violation,
        "max_constraint_violation": max_violation,
    }


def add_cash_asset(
    scenario_returns: pd.DataFrame,
    cash_return: float = 0.0,
    cash_name: str = "CASH",
) -> pd.DataFrame:
    """Append a constant-return cash column to a scenario matrix.

    Parameters
    ----------
    scenario_returns : pd.DataFrame
        Will not be mutated; a copy is returned.
    cash_return : float
        Return earned by cash in every scenario (after horizon scaling, if
        any — this function does no horizon scaling itself).
    cash_name : str
        Column name. Defaults to ``"CASH"``.

    Returns
    -------
    pd.DataFrame
        Copy with one extra column appended on the right.

    Raises
    ------
    ValueError
        If ``cash_name`` already exists in ``scenario_returns``.
    """
    if not isinstance(scenario_returns, pd.DataFrame):
        raise ValueError("scenario_returns must be a pandas DataFrame.")
    if cash_name in scenario_returns.columns:
        raise ValueError(
            f"Column '{cash_name}' already exists in scenario_returns. "
            "Pass a different `cash_name` or drop the existing column."
        )
    out = scenario_returns.copy()
    out[cash_name] = float(cash_return)
    return out


def estimate_expected_returns(
    scenario_returns: pd.DataFrame,
    method: str = "mean",
    shrinkage_weight: float = 0.5,
) -> pd.Series:
    """Estimate an expected-return vector from a scenario matrix.

    Parameters
    ----------
    scenario_returns : pd.DataFrame
    method : {"mean", "median", "zero", "shrinkage_to_zero"}
        * ``"mean"`` (default): arithmetic mean of each column.
        * ``"median"``: column median (more robust to outliers).
        * ``"zero"``: zero vector — useful for pure tail-risk optimization
          where the user does not want expected return to bias the solution.
        * ``"shrinkage_to_zero"``: ``shrinkage_weight * mean`` — shrinks the
          sample mean toward zero to dampen the selection bias from assets
          with exceptional historical growth.
    shrinkage_weight : float, in ``[0, 1]``
        Weight on the historical mean for ``"shrinkage_to_zero"`` (the
        complement is implicitly placed on a zero prior). ``1.0`` reproduces
        the plain mean; ``0.0`` reproduces the zero vector.

    Returns
    -------
    pd.Series
        Indexed by asset name.
    """
    validate_scenario_matrix(scenario_returns)
    method_lc = method.lower()
    if method_lc == "mean":
        out = scenario_returns.mean(axis=0)
    elif method_lc == "median":
        out = scenario_returns.median(axis=0)
    elif method_lc == "zero":
        out = pd.Series(0.0, index=scenario_returns.columns)
    elif method_lc in ("shrinkage_to_zero", "shrinkage"):
        if not (0.0 <= shrinkage_weight <= 1.0):
            raise ValueError(
                f"shrinkage_weight must be in [0, 1], got {shrinkage_weight}."
            )
        out = float(shrinkage_weight) * scenario_returns.mean(axis=0)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. "
            "Choose from 'mean', 'median', 'zero', 'shrinkage_to_zero'."
        )
    out.name = "expected_return"
    return out.astype(float)


def calculate_portfolio_scenario_metrics(
    scenario_returns: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float = 0.95,
    initial_capital: float | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute portfolio metrics (return / vol / VaR / CVaR / extremes)
    on a scenario matrix.

    Parameters
    ----------
    scenario_returns : pd.DataFrame
        Validated upstream.
    weights : pd.Series
        Indexed by asset; must cover every column of ``scenario_returns``.
    confidence_level : float, in (0, 1)
    initial_capital : float, optional
        If supplied, adds ``money_VaR`` and ``money_CVaR``.
    risk_free_rate : float, optional
        Per-horizon risk-free rate used for the Sharpe ratio
        ``(E[r] - rf) / vol``. Defaults to ``0.0``.

    Returns
    -------
    dict
        Keys: ``expected_return``, ``volatility``, ``VaR``, ``CVaR``,
        ``worst_return``, ``best_return``, ``sharpe_ratio``, and optionally
        ``money_VaR`` / ``money_CVaR``.
    """
    validate_scenario_matrix(scenario_returns)
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")
    if not isinstance(weights, pd.Series):
        raise ValueError("weights must be a pd.Series.")

    aligned = weights.reindex(scenario_returns.columns)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index.tolist()
        raise ValueError(f"weights missing for scenario columns: {missing}")

    portfolio = pd.Series(
        scenario_returns.to_numpy(dtype=float) @ aligned.to_numpy(dtype=float),
        index=scenario_returns.index,
        name="portfolio_scenario_return",
    )

    var = scenario_var(portfolio, confidence_level)
    cvar = scenario_cvar(portfolio, confidence_level)
    expected_return = float(portfolio.mean())
    volatility = float(portfolio.std(ddof=1))
    sharpe_ratio = (
        (expected_return - float(risk_free_rate)) / volatility
        if volatility > 0
        else 0.0
    )

    metrics: dict = {
        "expected_return": expected_return,
        "volatility": volatility,
        "VaR": float(var),
        "CVaR": float(cvar),
        "worst_return": float(portfolio.min()),
        "best_return": float(portfolio.max()),
        "sharpe_ratio": float(sharpe_ratio),
        "confidence_level": float(confidence_level),
        "n_scenarios": int(portfolio.size),
    }
    if initial_capital is not None:
        metrics["money_VaR"] = loss_value_to_money(var, initial_capital)
        metrics["money_CVaR"] = loss_value_to_money(cvar, initial_capital)
    return metrics


def format_weights_table(weights: pd.Series) -> pd.DataFrame:
    """Pretty-print weights as a sorted ``Asset / Weight`` table."""
    if not isinstance(weights, pd.Series):
        raise ValueError("weights must be a pd.Series.")
    df = pd.DataFrame(
        {"Asset": weights.index.astype(str), "Weight": weights.values.astype(float)}
    )
    df = df.sort_values("Weight", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public: scenario builders
# ─────────────────────────────────────────────────────────────────────────────


def _aggregate_horizon_simple(returns: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Rolling h-day simple aggregated returns: ``prod(1+r)-1``."""
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}.")
    if horizon_days == 1:
        return returns.dropna()

    cleaned = returns.dropna()
    if len(cleaned) < horizon_days:
        raise ValueError(
            f"Not enough observations ({len(cleaned)}) for horizon {horizon_days}."
        )

    rolling = (
        (1.0 + cleaned)
        .rolling(window=int(horizon_days))
        .apply(lambda x: float(np.prod(x) - 1.0), raw=True)
    )
    return rolling.dropna()


def build_optimization_scenarios(
    asset_returns: pd.DataFrame,
    source: str = "historical",
    n_scenarios: int = 5000,
    horizon_days: int = 1,
    student_t_df: float = 5,
    random_seed: int | None = 42,
    mean_vector: pd.Series | None = None,
    covariance_matrix: pd.DataFrame | None = None,
    return_method: str = "simple",
    covariance_policy: str = "repair",
) -> pd.DataFrame:
    """Build a scenario matrix for optimization.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Historical asset returns (rows = dates, columns = assets).
    source : {"historical", "normal_mc", "student_t_mc"}
    n_scenarios : int
        Only used for Monte Carlo sources.
    horizon_days : int
        For historical source, returns are aggregated into rolling
        h-day simple returns when ``> 1``. For Monte Carlo sources, the
        i.i.d. horizon scaling of mean / covariance is used.
    student_t_df : float
        Degrees of freedom for the Student-t source (must be ``> 2``).
    random_seed : int or None
    mean_vector, covariance_matrix : optional
        DAILY mean / covariance overrides for the Monte Carlo sources
        (e.g. robust or shrinkage estimates from the assumptions engine).
        When omitted, sample estimates from ``asset_returns`` are used.
        Ignored for the historical source.
    return_method : {"simple"}
        Optimization scenarios must be expressed as simple returns. Log
        returns must be converted before calling this function.
    covariance_policy : {"repair", "strict"}
        Governance policy for Monte Carlo covariance inputs. Ignored for
        historical scenarios.

    Returns
    -------
    pd.DataFrame
        Rows = scenarios, columns = asset names matching ``asset_returns``.
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise ValueError("asset_returns must be a pandas DataFrame.")
    if asset_returns.empty:
        raise ValueError("asset_returns is empty.")
    if return_method != "simple":
        raise ValueError(
            "build_optimization_scenarios requires simple-return inputs; "
            "convert log returns before calling."
        )
    src = source.lower()

    if src == "historical":
        scenarios = _aggregate_horizon_simple(asset_returns, int(horizon_days))
        scenarios = scenarios.reset_index(drop=True)
        scenarios.index = pd.Index([f"hist_{i + 1}" for i in range(len(scenarios))])
        scenarios.columns = asset_returns.columns
        return scenarios

    if src in ("normal_mc", "student_t_mc"):
        if mean_vector is None or covariance_matrix is None:
            params = estimate_return_parameters(asset_returns)
            if mean_vector is None:
                mean_vector = params["mean_vector"]
            if covariance_matrix is None:
                covariance_matrix = params["covariance_matrix"]

    if src == "normal_mc":
        return simulate_normal_returns(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            n_scenarios=int(n_scenarios),
            horizon_days=int(horizon_days),
            random_seed=random_seed,
            covariance_policy=covariance_policy,
        )

    if src == "student_t_mc":
        return simulate_student_t_returns(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            df=float(student_t_df),
            n_scenarios=int(n_scenarios),
            horizon_days=int(horizon_days),
            random_seed=random_seed,
            covariance_policy=covariance_policy,
        )

    raise ValueError(
        f"Unsupported scenario source '{source}'. "
        "Use 'historical', 'normal_mc', or 'student_t_mc'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public: optimizers
# ─────────────────────────────────────────────────────────────────────────────


def _empty_result(
    assets: list[str],
    status: str,
    message: str,
    solver: str,
    extras: Optional[dict] = None,
) -> dict:
    base: dict = {
        "status": status,
        "solver_status": status,
        "objective_value": float("nan"),
        "weights": pd.Series([float("nan")] * len(assets), index=assets, name="weight"),
        "expected_return": float("nan"),
        "VaR": float("nan"),
        "CVaR": float("nan"),
        "volatility": float("nan"),
        "solver": solver,
        "message": message,
        "constraint_validation": {
            "passed": False,
            "status": "not_run",
            "max_constraint_violation": float("nan"),
        },
        "max_constraint_violation": float("nan"),
    }
    if extras:
        base.update(extras)
    return base


def _enrich_with_metrics(
    result: dict,
    scenario_returns: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float,
    initial_capital: float | None,
) -> dict:
    """Recompute / overwrite metrics from the scenario matrix using the
    optimal weights. This guarantees that what we report is consistent
    with what the user sees in the comparison table."""
    metrics = calculate_portfolio_scenario_metrics(
        scenario_returns,
        weights,
        confidence_level=confidence_level,
        initial_capital=initial_capital,
    )
    result["expected_return"] = metrics["expected_return"]
    result["volatility"] = metrics["volatility"]
    result["VaR"] = metrics["VaR"]
    result["CVaR"] = metrics["CVaR"]
    if initial_capital is not None:
        result["money_VaR"] = metrics.get("money_VaR")
        result["money_CVaR"] = metrics.get("money_CVaR")
    return result


def _apply_solution_governance(
    result: dict,
    scenario_returns: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float,
    long_only: bool,
    min_weight: float | None,
    max_weight: float | None,
    expected_returns: pd.Series | None = None,
    target_return: float | None = None,
    cvar_limit: float | None = None,
    solver_cvar: float | None = None,
    var_threshold: float | None = None,
    excess_losses: np.ndarray | None = None,
    tolerance: float = 1e-5,
) -> dict:
    """Attach independent residual checks and reject false solver success."""
    solver_status = str(result.get("status", "unknown"))
    validation = validate_solution_residuals(
        scenario_returns,
        weights,
        confidence_level=confidence_level,
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
        expected_returns=expected_returns,
        target_return=target_return,
        cvar_limit=cvar_limit,
        solver_cvar=solver_cvar,
        var_threshold=var_threshold,
        excess_losses=excess_losses,
        tolerance=tolerance,
    )
    result["solver_status"] = solver_status
    result["constraint_validation"] = validation
    result["max_constraint_violation"] = validation["max_constraint_violation"]
    if not validation["passed"]:
        result["status"] = "validation_failed"
        result["message"] = (
            f"Solver reported {solver_status}, but independent constraint "
            "validation failed: max violation "
            f"{validation['max_constraint_violation']:.3e} exceeds tolerance "
            f"{validation['tolerance']:.3e}."
        )
    else:
        result["message"] = (
            f"{_status_message(solver_status)} Independent residual validation passed."
        )
    return result


def minimize_cvar(
    scenario_returns: pd.DataFrame,
    confidence_level: float = 0.95,
    long_only: bool = True,
    min_weight: float | None = None,
    max_weight: float | None = 1.0,
    include_cash: bool = False,
    cash_return: float = 0.0,
    solver: str | None = None,
) -> dict:
    """Minimize CVaR via the Rockafellar-Uryasev LP.

    Decision variables
    ------------------
    ``w`` (n_assets), ``t`` (VaR threshold), ``u`` (n_scenarios excess
    losses).

    Returns
    -------
    dict
        Keys: ``status``, ``objective_value``, ``weights``,
        ``expected_return``, ``VaR``, ``CVaR``, ``volatility``,
        ``solver``, ``message``.
    """
    validate_scenario_matrix(scenario_returns)
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")

    if include_cash:
        scenario_returns = add_cash_asset(scenario_returns, cash_return=cash_return)

    assets = list(scenario_returns.columns)
    R = scenario_returns.to_numpy(dtype=float)
    n_scenarios, n_assets = R.shape
    beta = float(confidence_level)

    cp = _import_cvxpy()
    w = cp.Variable(n_assets, name="w")
    t = cp.Variable(name="t")
    u = cp.Variable(n_scenarios, nonneg=True, name="u")

    losses = -R @ w  # shape (n_scenarios,)
    cvar_expr = t + (1.0 / ((1.0 - beta) * n_scenarios)) * cp.sum(u)

    constraints = _build_weight_constraints(
        w, n_assets, long_only, min_weight, max_weight
    )
    constraints.append(u >= losses - t)

    problem = cp.Problem(cp.Minimize(cvar_expr), constraints)

    try:
        solver_used = _solve(problem, solver)
    except Exception as exc:  # noqa: BLE001
        return _empty_result(
            assets,
            status="solver_error",
            message=f"Solver failed: {exc}",
            solver=str(solver or "auto"),
        )

    status = problem.status
    if not _is_solved(status) or w.value is None:
        return _empty_result(
            assets,
            status=status,
            message=_status_message(status),
            solver=solver_used,
        )

    weights = pd.Series(
        np.asarray(w.value, dtype=float).flatten(),
        index=assets,
        name="weight",
    )
    t_value = float(t.value) if t.value is not None else float("nan")
    u_values = (
        np.asarray(u.value, dtype=float).reshape(-1)
        if u.value is not None
        else np.full(n_scenarios, float("nan"))
    )
    solver_cvar = t_value + (1.0 / ((1.0 - beta) * n_scenarios)) * float(
        np.sum(u_values)
    )

    result = {
        "status": status,
        "solver_status": status,
        "objective_value": float(problem.value),
        "weights": weights,
        "expected_return": float("nan"),
        "VaR": float(t.value) if t.value is not None else float("nan"),
        "CVaR": float(problem.value),
        "volatility": float("nan"),
        "solver": solver_used,
        "message": _status_message(status),
        "confidence_level": beta,
    }
    result = _enrich_with_metrics(
        result, scenario_returns, weights, beta, initial_capital=None
    )
    result = _apply_solution_governance(
        result,
        scenario_returns,
        weights,
        confidence_level=beta,
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
        solver_cvar=solver_cvar,
        var_threshold=t_value,
        excess_losses=u_values,
    )
    return result


def maximize_return_with_cvar_constraint(
    scenario_returns: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    cvar_limit: float = 0.10,
    confidence_level: float = 0.95,
    long_only: bool = True,
    min_weight: float | None = None,
    max_weight: float | None = 1.0,
    include_cash: bool = False,
    cash_return: float = 0.0,
    solver: str | None = None,
) -> dict:
    """Maximize expected return subject to ``CVaR(w) <= cvar_limit``.

    ``cvar_limit`` is intentionally restricted to a positive loss budget
    (e.g. ``0.10`` means at most a 10% CVaR loss), although reported CVaR
    values follow the signed loss-space output contract.
    """
    validate_scenario_matrix(scenario_returns)
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")
    if cvar_limit <= 0:
        raise ValueError(f"cvar_limit must be > 0, got {cvar_limit}.")

    if include_cash:
        scenario_returns = add_cash_asset(scenario_returns, cash_return=cash_return)

    if expected_returns is None:
        expected_returns = estimate_expected_returns(scenario_returns, "mean")
    else:
        expected_returns = expected_returns.reindex(scenario_returns.columns)
        if expected_returns.isna().any():
            missing = expected_returns[expected_returns.isna()].index.tolist()
            raise ValueError(f"expected_returns missing for columns: {missing}")

    assets = list(scenario_returns.columns)
    R = scenario_returns.to_numpy(dtype=float)
    mu = expected_returns.to_numpy(dtype=float)
    n_scenarios, n_assets = R.shape
    beta = float(confidence_level)

    cp = _import_cvxpy()
    w = cp.Variable(n_assets, name="w")
    t = cp.Variable(name="t")
    u = cp.Variable(n_scenarios, nonneg=True, name="u")

    losses = -R @ w
    cvar_expr = t + (1.0 / ((1.0 - beta) * n_scenarios)) * cp.sum(u)

    constraints = _build_weight_constraints(
        w, n_assets, long_only, min_weight, max_weight
    )
    constraints.append(u >= losses - t)
    constraints.append(cvar_expr <= float(cvar_limit))

    problem = cp.Problem(cp.Maximize(mu @ w), constraints)

    try:
        solver_used = _solve(problem, solver)
    except Exception as exc:  # noqa: BLE001
        return _empty_result(
            assets,
            status="solver_error",
            message=f"Solver failed: {exc}",
            solver=str(solver or "auto"),
            extras={"cvar_limit": float(cvar_limit)},
        )

    status = problem.status
    if not _is_solved(status) or w.value is None:
        return _empty_result(
            assets,
            status=status,
            message=_status_message(status),
            solver=solver_used,
            extras={"cvar_limit": float(cvar_limit)},
        )

    weights = pd.Series(
        np.asarray(w.value, dtype=float).flatten(),
        index=assets,
        name="weight",
    )
    t_value = float(t.value) if t.value is not None else float("nan")
    u_values = (
        np.asarray(u.value, dtype=float).reshape(-1)
        if u.value is not None
        else np.full(n_scenarios, float("nan"))
    )
    solver_cvar = t_value + (1.0 / ((1.0 - beta) * n_scenarios)) * float(
        np.sum(u_values)
    )

    result = {
        "status": status,
        "solver_status": status,
        "objective_value": float(problem.value),
        "weights": weights,
        "expected_return": float(problem.value),
        "VaR": float(t.value) if t.value is not None else float("nan"),
        "CVaR": float("nan"),
        "volatility": float("nan"),
        "cvar_limit": float(cvar_limit),
        "solver": solver_used,
        "message": _status_message(status),
        "confidence_level": beta,
    }
    warning = _zero_mu_warning(mu)
    if warning:
        result["warning"] = warning
    result = _enrich_with_metrics(
        result, scenario_returns, weights, beta, initial_capital=None
    )
    result = _apply_solution_governance(
        result,
        scenario_returns,
        weights,
        confidence_level=beta,
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
        expected_returns=expected_returns,
        cvar_limit=float(cvar_limit),
        solver_cvar=solver_cvar,
        var_threshold=t_value,
        excess_losses=u_values,
    )
    return result


def minimize_cvar_for_target_return(
    scenario_returns: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    target_return: float = 0.0,
    confidence_level: float = 0.95,
    long_only: bool = True,
    min_weight: float | None = None,
    max_weight: float | None = 1.0,
    include_cash: bool = False,
    cash_return: float = 0.0,
    solver: str | None = None,
) -> dict:
    """Minimize CVaR subject to ``E[r] @ w >= target_return``."""
    validate_scenario_matrix(scenario_returns)
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")

    if include_cash:
        scenario_returns = add_cash_asset(scenario_returns, cash_return=cash_return)

    if expected_returns is None:
        expected_returns = estimate_expected_returns(scenario_returns, "mean")
    else:
        expected_returns = expected_returns.reindex(scenario_returns.columns)
        if expected_returns.isna().any():
            missing = expected_returns[expected_returns.isna()].index.tolist()
            raise ValueError(f"expected_returns missing for columns: {missing}")

    assets = list(scenario_returns.columns)
    R = scenario_returns.to_numpy(dtype=float)
    mu = expected_returns.to_numpy(dtype=float)
    n_scenarios, n_assets = R.shape
    beta = float(confidence_level)

    cp = _import_cvxpy()
    w = cp.Variable(n_assets, name="w")
    t = cp.Variable(name="t")
    u = cp.Variable(n_scenarios, nonneg=True, name="u")

    losses = -R @ w
    cvar_expr = t + (1.0 / ((1.0 - beta) * n_scenarios)) * cp.sum(u)

    constraints = _build_weight_constraints(
        w, n_assets, long_only, min_weight, max_weight
    )
    constraints.append(u >= losses - t)
    constraints.append(mu @ w >= float(target_return))

    problem = cp.Problem(cp.Minimize(cvar_expr), constraints)

    try:
        solver_used = _solve(problem, solver)
    except Exception as exc:  # noqa: BLE001
        return _empty_result(
            assets,
            status="solver_error",
            message=f"Solver failed: {exc}",
            solver=str(solver or "auto"),
            extras={"target_return": float(target_return)},
        )

    status = problem.status
    if not _is_solved(status) or w.value is None:
        return _empty_result(
            assets,
            status=status,
            message=_status_message(status),
            solver=solver_used,
            extras={"target_return": float(target_return)},
        )

    weights = pd.Series(
        np.asarray(w.value, dtype=float).flatten(),
        index=assets,
        name="weight",
    )
    t_value = float(t.value) if t.value is not None else float("nan")
    u_values = (
        np.asarray(u.value, dtype=float).reshape(-1)
        if u.value is not None
        else np.full(n_scenarios, float("nan"))
    )
    solver_cvar = t_value + (1.0 / ((1.0 - beta) * n_scenarios)) * float(
        np.sum(u_values)
    )

    result = {
        "status": status,
        "solver_status": status,
        "objective_value": float(problem.value),
        "weights": weights,
        "expected_return": float("nan"),
        "VaR": float(t.value) if t.value is not None else float("nan"),
        "CVaR": float(problem.value),
        "volatility": float("nan"),
        "target_return": float(target_return),
        "solver": solver_used,
        "message": _status_message(status),
        "confidence_level": beta,
    }
    warning = _zero_mu_warning(mu)
    if warning:
        result["warning"] = warning
    result = _enrich_with_metrics(
        result, scenario_returns, weights, beta, initial_capital=None
    )
    result = _apply_solution_governance(
        result,
        scenario_returns,
        weights,
        confidence_level=beta,
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
        expected_returns=expected_returns,
        target_return=float(target_return),
        solver_cvar=solver_cvar,
        var_threshold=t_value,
        excess_losses=u_values,
    )
    return result


def generate_cvar_efficient_frontier(
    scenario_returns: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    confidence_level: float = 0.95,
    n_points: int = 20,
    long_only: bool = True,
    min_weight: float | None = None,
    max_weight: float | None = 1.0,
    include_cash: bool = False,
    cash_return: float = 0.0,
    solver: str | None = None,
) -> pd.DataFrame:
    """Sweep target returns from low (min-CVaR portfolio) to high (max
    feasible expected return) and assemble a CVaR efficient frontier.

    Returns
    -------
    pd.DataFrame
        Columns: ``target_return``, ``expected_return``, ``volatility``,
        ``VaR``, ``CVaR``, ``status``, and one column per asset named
        ``weight_<ASSET>``. Infeasible target returns are skipped.
    """
    validate_scenario_matrix(scenario_returns)
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}.")

    if include_cash:
        scenario_returns = add_cash_asset(scenario_returns, cash_return=cash_return)

    if expected_returns is None:
        expected_returns = estimate_expected_returns(scenario_returns, "mean")
    else:
        expected_returns = expected_returns.reindex(scenario_returns.columns)
        if expected_returns.isna().any():
            missing = expected_returns[expected_returns.isna()].index.tolist()
            raise ValueError(f"expected_returns missing for columns: {missing}")

    assets = list(scenario_returns.columns)

    # Lower bound: expected return of the unconstrained min-CVaR portfolio.
    min_cvar = minimize_cvar(
        scenario_returns,
        confidence_level=confidence_level,
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
        include_cash=False,  # already added above
        solver=solver,
    )

    # Upper bound: maximize expected return subject ONLY to weight
    # constraints (no CVaR cap).
    cp = _import_cvxpy()
    R = scenario_returns.to_numpy(dtype=float)
    n_assets = R.shape[1]
    mu = expected_returns.to_numpy(dtype=float)

    w = cp.Variable(n_assets)
    constraints = _build_weight_constraints(
        w, n_assets, long_only, min_weight, max_weight
    )
    upper_problem = cp.Problem(cp.Maximize(mu @ w), constraints)
    try:
        _solve(upper_problem, solver)
    except Exception:  # noqa: BLE001
        upper_problem._status = "solver_error"  # type: ignore[attr-defined]

    if _is_solved(upper_problem.status) and w.value is not None:
        upper_weights = pd.Series(
            np.asarray(w.value, dtype=float).flatten(),
            index=assets,
        )
        upper_validation = validate_solution_residuals(
            scenario_returns,
            upper_weights,
            confidence_level=confidence_level,
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        upper_return = (
            float(np.dot(mu, upper_weights.to_numpy()))
            if upper_validation["passed"]
            else float(np.max(mu))
        )
    else:
        upper_return = float(np.max(mu))

    if _is_solved(min_cvar["status"]):
        lower_return = float(min_cvar["expected_return"])
    else:
        lower_return = float(np.min(mu))

    if upper_return <= lower_return:
        # Degenerate; produce a single point at the min-CVaR portfolio.
        targets = np.array([lower_return])
    else:
        targets = np.linspace(lower_return, upper_return, int(n_points))

    rows: list[dict] = []
    weight_cols = [f"weight_{a}" for a in assets]
    for target in targets:
        result = minimize_cvar_for_target_return(
            scenario_returns,
            expected_returns=expected_returns,
            target_return=float(target),
            confidence_level=confidence_level,
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
            include_cash=False,  # already added
            solver=solver,
        )
        row = {
            "target_return": float(target),
            "expected_return": float(result.get("expected_return", float("nan"))),
            "volatility": float(result.get("volatility", float("nan"))),
            "VaR": float(result.get("VaR", float("nan"))),
            "CVaR": float(result.get("CVaR", float("nan"))),
            "status": str(result.get("status", "unknown")),
            "solver_status": str(result.get("solver_status", "unknown")),
            "max_constraint_violation": float(
                result.get("max_constraint_violation", float("nan"))
            ),
        }
        weights = result.get("weights")
        if isinstance(weights, pd.Series):
            for asset in assets:
                row[f"weight_{asset}"] = float(weights.get(asset, float("nan")))
        else:
            for col in weight_cols:
                row[col] = float("nan")
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "target_return",
                "expected_return",
                "volatility",
                "VaR",
                "CVaR",
                "status",
                *weight_cols,
            ]
        )

    frontier = pd.DataFrame(rows)
    # Keep only points that solved — but record an unsolved-points count
    # in an attribute so downstream code can warn if it wants.
    feasible = frontier[
        frontier["status"].isin(["optimal", "optimal_inaccurate"])
    ].reset_index(drop=True)
    if feasible.empty:
        return frontier  # all infeasible — return as-is
    feasible.attrs["n_infeasible"] = int(len(frontier) - len(feasible))
    return feasible


def maximize_sharpe_ratio(
    scenario_returns: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95,
    long_only: bool = True,
    min_weight: float | None = None,
    max_weight: float | None = 1.0,
    include_cash: bool = False,
    cash_return: float = 0.0,
    solver: str | None = None,
    n_grid: int = 25,
    min_volatility: float = 1e-6,
    cash_name: str = "CASH",
) -> dict:
    """Maximum-Sharpe portfolio via constrained candidate selection.

    Direct Sharpe maximization is a non-convex fractional programme. Instead
    of a fragile non-convex solve, we generate a set of **constraint-feasible**
    candidate portfolios along the CVaR efficient frontier (each produced by an
    LP, so every box / long-only constraint is honoured) and return the one
    with the highest Sharpe ratio ``(E[r] - rf) / vol``. This is always
    feasible and needs only the open-source LP solvers already in use.

    Parameters
    ----------
    min_volatility : float
        Candidates with volatility below this floor are excluded — a
        (near-)zero-volatility portfolio (e.g. 100 % cash) makes the Sharpe
        ratio numerically explosive and economically meaningless. If every
        candidate is below the floor the result is marked infeasible with
        an explanatory message.
    cash_name : str
        Column treated as cash for the cash-domination warning.

    Returns
    -------
    dict
        The uniform optimiser result dict plus ``sharpe_ratio`` and
        ``method`` (``"candidate-grid"``); may include a ``warning`` key
        when cash dominates the selected portfolio or expected returns
        are all zero.
    """
    validate_scenario_matrix(scenario_returns)
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")

    if include_cash:
        scenario_returns = add_cash_asset(scenario_returns, cash_return=cash_return)

    assets = list(scenario_returns.columns)
    if expected_returns is None:
        expected_returns = estimate_expected_returns(scenario_returns, "mean")
    else:
        expected_returns = expected_returns.reindex(scenario_returns.columns)
        if expected_returns.isna().any():
            missing = expected_returns[expected_returns.isna()].index.tolist()
            raise ValueError(f"expected_returns missing for columns: {missing}")

    frontier = generate_cvar_efficient_frontier(
        scenario_returns,
        expected_returns=expected_returns,
        confidence_level=confidence_level,
        n_points=int(n_grid),
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
        include_cash=False,  # already handled above
        solver=solver,
    )

    vol_floor = max(float(min_volatility), 0.0)
    best_weights: pd.Series | None = None
    best_solver_status = "unknown"
    best_sharpe = -np.inf
    n_skipped_low_vol = 0
    for _, row in frontier.iterrows():
        vol = float(row.get("volatility", float("nan")))
        er = float(row.get("expected_return", float("nan")))
        if not (np.isfinite(vol) and vol > 0 and np.isfinite(er)):
            continue
        if vol < vol_floor:
            n_skipped_low_vol += 1
            continue
        sharpe = (er - float(risk_free_rate)) / vol
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = pd.Series(
                {a: float(row.get(f"weight_{a}", float("nan"))) for a in assets},
                name="weight",
            )
            best_solver_status = str(row.get("solver_status", "unknown"))

    if best_weights is None or best_weights.isna().any():
        message = "No feasible candidate portfolio produced a finite Sharpe ratio."
        if n_skipped_low_vol > 0:
            message += (
                f" {n_skipped_low_vol} candidate(s) were excluded for "
                f"near-zero volatility (< {vol_floor:g}) — e.g. a ~100 % cash "
                "portfolio, whose Sharpe ratio is not meaningful."
            )
        return _empty_result(
            assets,
            status="infeasible",
            message=message,
            solver=str(solver or "auto"),
            extras={
                "sharpe_ratio": float("nan"),
                "method": "candidate-grid",
                "n_skipped_low_vol": n_skipped_low_vol,
            },
        )

    metrics = calculate_portfolio_scenario_metrics(
        scenario_returns,
        best_weights,
        confidence_level=confidence_level,
        risk_free_rate=risk_free_rate,
    )
    result = {
        "status": "optimal",
        "solver_status": best_solver_status,
        "objective_value": float(metrics["sharpe_ratio"]),
        "weights": best_weights,
        "expected_return": float(metrics["expected_return"]),
        "VaR": float(metrics["VaR"]),
        "CVaR": float(metrics["CVaR"]),
        "volatility": float(metrics["volatility"]),
        "sharpe_ratio": float(metrics["sharpe_ratio"]),
        "risk_free_rate": float(risk_free_rate),
        "confidence_level": float(confidence_level),
        "solver": str(solver or "auto"),
        "method": "candidate-grid",
        "n_skipped_low_vol": n_skipped_low_vol,
        "message": "Selected max-Sharpe portfolio from the CVaR frontier candidates.",
    }
    result = _apply_solution_governance(
        result,
        scenario_returns,
        best_weights,
        confidence_level=confidence_level,
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
        expected_returns=expected_returns,
    )

    warnings: list[str] = []
    mu_warning = _zero_mu_warning(expected_returns.to_numpy(dtype=float))
    if mu_warning:
        warnings.append(mu_warning)
    cash_weight = float(best_weights.get(cash_name, 0.0))
    if cash_weight > 0.90:
        warnings.append(
            f"The max-Sharpe portfolio is {cash_weight * 100:.0f}% cash. "
            "Cash is an *absolute* defensive asset (its excess return over "
            "the risk-free rate is ~0 and its volatility is ~0), so a "
            "cash-dominated Sharpe portfolio mostly reflects the cash return "
            "vs risk-free spread, not a genuine risk/return trade-off."
        )
    if n_skipped_low_vol > 0:
        warnings.append(
            f"{n_skipped_low_vol} near-zero-volatility candidate(s) "
            "(e.g. ~100 % cash) were excluded from the Sharpe comparison."
        )
    if warnings:
        result["warning"] = " ".join(warnings)
    return result


def compare_current_vs_optimized(
    scenario_returns: pd.DataFrame,
    current_weights: pd.Series,
    optimized_results: dict[str, dict],
    confidence_level: float = 0.95,
    initial_capital: float | None = None,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Tabulate current portfolio vs each optimized portfolio.

    Parameters
    ----------
    scenario_returns : pd.DataFrame
        Scenario matrix used for **every** comparison. If the optimization
        was run with ``include_cash=True``, the cash column must also be
        present here (and ``current_weights`` should include CASH=0).
    current_weights : pd.Series
        Weights of the user's existing portfolio.
    optimized_results : dict
        ``{"Min CVaR": result_dict, ...}``. ``"Current"`` is added
        automatically if not already present.
    confidence_level : float
    initial_capital : float, optional
        If supplied, money VaR / CVaR columns are populated.

    Returns
    -------
    pd.DataFrame
        Columns: ``Portfolio``, ``Expected Return``, ``Volatility``,
        ``VaR``, ``CVaR``, ``Money VaR``, ``Money CVaR``, ``Status``.
    """
    validate_scenario_matrix(scenario_returns)

    rows: list[dict] = []
    cap = float(initial_capital) if initial_capital is not None else None

    # Current portfolio metrics — guaranteed-feasible, computed directly.
    try:
        current_metrics = calculate_portfolio_scenario_metrics(
            scenario_returns,
            current_weights,
            confidence_level=confidence_level,
            initial_capital=cap,
            risk_free_rate=risk_free_rate,
        )
        rows.append(
            {
                "Portfolio": "Current",
                "Expected Return": current_metrics["expected_return"],
                "Volatility": current_metrics["volatility"],
                "Sharpe": current_metrics["sharpe_ratio"],
                "VaR": current_metrics["VaR"],
                "CVaR": current_metrics["CVaR"],
                "Money VaR": current_metrics.get("money_VaR", float("nan")),
                "Money CVaR": current_metrics.get("money_CVaR", float("nan")),
                "Status": "current",
            }
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            {
                "Portfolio": "Current",
                "Expected Return": float("nan"),
                "Volatility": float("nan"),
                "Sharpe": float("nan"),
                "VaR": float("nan"),
                "CVaR": float("nan"),
                "Money VaR": float("nan"),
                "Money CVaR": float("nan"),
                "Status": f"error: {exc}",
            }
        )

    for label, result in optimized_results.items():
        if label.lower() == "current":
            # avoid duplicating
            continue
        status = str(result.get("status", "unknown"))
        weights = result.get("weights")
        if (
            _is_solved(status)
            and isinstance(weights, pd.Series)
            and not weights.isna().all()
        ):
            try:
                m = calculate_portfolio_scenario_metrics(
                    scenario_returns,
                    weights,
                    confidence_level=confidence_level,
                    initial_capital=cap,
                    risk_free_rate=risk_free_rate,
                )
                rows.append(
                    {
                        "Portfolio": label,
                        "Expected Return": m["expected_return"],
                        "Volatility": m["volatility"],
                        "Sharpe": m["sharpe_ratio"],
                        "VaR": m["VaR"],
                        "CVaR": m["CVaR"],
                        "Money VaR": m.get("money_VaR", float("nan")),
                        "Money CVaR": m.get("money_CVaR", float("nan")),
                        "Status": status,
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001
                status = f"metric_error: {exc}"
        rows.append(
            {
                "Portfolio": label,
                "Expected Return": float("nan"),
                "Volatility": float("nan"),
                "Sharpe": float("nan"),
                "VaR": float("nan"),
                "CVaR": float("nan"),
                "Money VaR": float("nan"),
                "Money CVaR": float("nan"),
                "Status": status,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Portfolio",
            "Expected Return",
            "Volatility",
            "Sharpe",
            "VaR",
            "CVaR",
            "Money VaR",
            "Money CVaR",
            "Status",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public: governance — feasibility diagnostics and result interpretation
# ─────────────────────────────────────────────────────────────────────────────


def compute_feasible_risk_return_bounds(
    scenario_returns: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    confidence_level: float = 0.95,
    long_only: bool = True,
    min_weight: float | None = None,
    max_weight: float | None = 1.0,
    solver: str | None = None,
) -> dict:
    """Feasible bounds under the weight constraints only.

    Returns
    -------
    dict
        ``min_cvar`` (lowest achievable CVaR), ``min_cvar_return`` (its
        expected return), ``max_return`` (highest achievable expected
        return), ``max_return_cvar`` (its CVaR). Any value can be NaN if
        the corresponding LP failed.
    """
    validate_scenario_matrix(scenario_returns)
    if expected_returns is None:
        expected_returns = estimate_expected_returns(scenario_returns, "mean")
    else:
        expected_returns = expected_returns.reindex(scenario_returns.columns)

    bounds = {
        "min_cvar": float("nan"),
        "min_cvar_return": float("nan"),
        "max_return": float("nan"),
        "max_return_cvar": float("nan"),
    }

    min_cvar_result = minimize_cvar(
        scenario_returns,
        confidence_level=confidence_level,
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
        solver=solver,
    )
    if _is_solved(min_cvar_result["status"]):
        bounds["min_cvar"] = float(min_cvar_result["CVaR"])
        bounds["min_cvar_return"] = float(min_cvar_result["expected_return"])

    # Max expected return subject ONLY to the weight constraints.
    cp = _import_cvxpy()
    mu = expected_returns.to_numpy(dtype=float)
    n_assets = scenario_returns.shape[1]
    w = cp.Variable(n_assets)
    constraints = _build_weight_constraints(
        w, n_assets, long_only, min_weight, max_weight
    )
    problem = cp.Problem(cp.Maximize(mu @ w), constraints)
    try:
        _solve(problem, solver)
    except Exception:  # noqa: BLE001
        return bounds
    if _is_solved(problem.status) and w.value is not None:
        weights = pd.Series(
            np.asarray(w.value, dtype=float).flatten(),
            index=list(scenario_returns.columns),
        )
        validation = validate_solution_residuals(
            scenario_returns,
            weights,
            confidence_level=confidence_level,
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        if not validation["passed"]:
            return bounds
        bounds["max_return"] = float(np.dot(mu, weights.to_numpy()))
        try:
            m = calculate_portfolio_scenario_metrics(
                scenario_returns, weights, confidence_level=confidence_level
            )
            bounds["max_return_cvar"] = float(m["CVaR"])
        except Exception:  # noqa: BLE001
            pass
    return bounds


def diagnose_infeasibility(
    scenario_returns: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    confidence_level: float = 0.95,
    long_only: bool = True,
    min_weight: float | None = None,
    max_weight: float | None = 1.0,
    cvar_limit: float | None = None,
    target_return: float | None = None,
    cash_enabled: bool = False,
    solver: str | None = None,
) -> list[str]:
    """Best-effort explanations for an infeasible optimization.

    Checks, in order: budget feasibility of the weight box, an unreachable
    CVaR cap, an unreachable target return, and all-zero expected returns.
    Returns a list of human-readable diagnostic strings (possibly with a
    single generic entry when no specific cause can be pinned down).
    """
    reasons: list[str] = []
    n_assets = int(scenario_returns.shape[1])

    lb = float(min_weight) if min_weight is not None else (0.0 if long_only else None)
    if long_only and lb is not None:
        lb = max(0.0, lb)
    ub = float(max_weight) if max_weight is not None else None

    if lb is not None and lb * n_assets > 1.0 + 1e-9:
        reasons.append(
            f"Min weight {lb:.2f} × {n_assets} assets = {lb * n_assets:.2f} > 1 — "
            "the minimum-weight constraint alone over-allocates the budget. "
            "Lower the min weight or reduce the number of assets."
        )
    if ub is not None and ub * n_assets < 1.0 - 1e-9:
        reasons.append(
            f"Max weight {ub:.2f} × {n_assets} assets = {ub * n_assets:.2f} < 1 — "
            "the maximum-weight constraint cannot absorb the full budget. "
            "Raise the max weight or add assets"
            + ("" if cash_enabled else " (or enable the cash asset)")
            + "."
        )
    if lb is not None and ub is not None and lb > ub:
        reasons.append(
            f"Min weight {lb:.2f} exceeds max weight {ub:.2f} — no weight "
            "vector can satisfy both."
        )
    if reasons:
        return reasons  # budget-infeasible; deeper LPs would also fail

    bounds = compute_feasible_risk_return_bounds(
        scenario_returns,
        expected_returns=expected_returns,
        confidence_level=confidence_level,
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
        solver=solver,
    )

    if cvar_limit is not None and np.isfinite(bounds["min_cvar"]):
        if bounds["min_cvar"] > float(cvar_limit) + 1e-9:
            reasons.append(
                f"CVaR cap {cvar_limit * 100:.2f}% is below the minimum "
                f"achievable CVaR {bounds['min_cvar'] * 100:.2f}% under the "
                "current constraints — the risk budget is too tight. Raise "
                "the cap, relax the weight constraints"
                + ("" if cash_enabled else ", or enable the cash asset")
                + "."
            )

    if target_return is not None:
        mu = (
            expected_returns.reindex(scenario_returns.columns)
            if expected_returns is not None
            else estimate_expected_returns(scenario_returns, "mean")
        )
        mu_arr = mu.to_numpy(dtype=float)
        if np.allclose(mu_arr, 0.0, atol=1e-12) and float(target_return) > 0:
            reasons.append(
                "All expected returns are zero, so any positive target "
                "return is unreachable by construction. Pick a non-zero "
                "expected-return estimator or lower the target to 0."
            )
        elif np.isfinite(bounds["max_return"]) and (
            float(target_return) > bounds["max_return"] + 1e-12
        ):
            reasons.append(
                f"Target return {target_return * 100:.3f}% exceeds the "
                f"maximum achievable expected return "
                f"{bounds['max_return'] * 100:.3f}% under the current "
                "expected-return assumptions and weight constraints. Lower "
                "the target, relax the constraints, or use a less "
                "conservative expected-return estimator."
            )

    if not reasons:
        reasons.append(
            "No single constraint could be identified as the cause. Likely "
            "candidates: the combination of CVaR cap / target return with "
            "the weight box, or a numerically difficult scenario matrix. "
            "Try relaxing one constraint at a time to isolate it."
        )
    return reasons


def interpret_optimization_result(
    result: dict,
    cvar_limit: float | None = None,
    target_return: float | None = None,
    min_weight: float | None = None,
    min_cvar_bound: float | None = None,
    max_return_cvar: float | None = None,
    cash_name: str = "CASH",
    binding_tolerance: float = 1e-3,
    weight_epsilon: float = 1e-4,
) -> dict:
    """Interpret an optimizer result dict for display.

    Parameters
    ----------
    result : dict
        A uniform optimizer result (from any optimizer in this module).
    cvar_limit, target_return, min_weight : optional
        The constraint values used, for binding checks.
    min_cvar_bound, max_return_cvar : optional
        Feasible CVaR bounds (from
        :func:`compute_feasible_risk_return_bounds`) used to classify the
        portfolio as defensive / balanced / aggressive by where its CVaR
        sits inside the feasible risk range.

    Returns
    -------
    dict
        Keys: ``solved`` (bool), ``cvar_cap_binding``,
        ``target_return_binding`` (bool | None when not applicable),
        ``cash_weight`` (float), ``excluded_assets`` (list),
        ``at_min_weight_assets`` (list), ``risk_profile``
        (``"defensive" | "balanced" | "aggressive" | "unknown"``),
        ``notes`` (list[str]).
    """
    notes: list[str] = []
    solved = _is_solved(str(result.get("status", "")))
    weights = result.get("weights")
    has_weights = isinstance(weights, pd.Series) and not weights.isna().all()

    interpretation: dict = {
        "solved": solved,
        "cvar_cap_binding": None,
        "target_return_binding": None,
        "cash_weight": 0.0,
        "excluded_assets": [],
        "at_min_weight_assets": [],
        "risk_profile": "unknown",
        "notes": notes,
    }
    if not (solved and has_weights):
        return interpretation

    cvar = float(result.get("CVaR", float("nan")))
    er = float(result.get("expected_return", float("nan")))

    if cvar_limit is not None and np.isfinite(cvar):
        binding = abs(cvar - float(cvar_limit)) <= binding_tolerance
        interpretation["cvar_cap_binding"] = bool(binding)
        notes.append(
            f"CVaR cap {cvar_limit * 100:.2f}% is "
            + (
                "**binding** — the optimizer used the full risk budget."
                if binding
                else f"not binding (portfolio CVaR {cvar * 100:.2f}%)."
            )
        )
    if target_return is not None and np.isfinite(er):
        binding = abs(er - float(target_return)) <= binding_tolerance
        interpretation["target_return_binding"] = bool(binding)
        notes.append(
            f"Target return {target_return * 100:.3f}% is "
            + (
                "**binding** — expected return sits exactly on the target."
                if binding
                else f"not binding (expected return {er * 100:.3f}%)."
            )
        )

    cash_weight = float(weights.get(cash_name, 0.0))
    interpretation["cash_weight"] = cash_weight
    if cash_weight > weight_epsilon:
        notes.append(
            f"Cash allocation: {cash_weight * 100:.1f}%. Cash is an "
            "*absolute* defensive asset (constant return, ~zero volatility); "
            "a 'defensive' crypto asset like BTC is only defensive *relative "
            "to other crypto assets*."
        )

    lb = float(min_weight) if min_weight is not None else 0.0
    for asset, w_val in weights.items():
        if asset == cash_name:
            continue
        w_f = float(w_val)
        if w_f <= weight_epsilon and lb <= weight_epsilon:
            interpretation["excluded_assets"].append(str(asset))
        elif lb > weight_epsilon and abs(w_f - lb) <= weight_epsilon:
            interpretation["at_min_weight_assets"].append(str(asset))
    if interpretation["excluded_assets"]:
        notes.append(
            "Excluded (zero-weight) assets: "
            + ", ".join(interpretation["excluded_assets"])
            + "."
        )
    if interpretation["at_min_weight_assets"]:
        notes.append(
            "Held only at the minimum weight (forced diversification): "
            + ", ".join(interpretation["at_min_weight_assets"])
            + ". Forced diversification lowers concentration risk but can "
            "reduce expected return / Sharpe or raise CVaR."
        )

    if (
        min_cvar_bound is not None
        and max_return_cvar is not None
        and np.isfinite(min_cvar_bound)
        and np.isfinite(max_return_cvar)
        and np.isfinite(cvar)
        and max_return_cvar > min_cvar_bound + 1e-12
    ):
        position = (cvar - float(min_cvar_bound)) / (
            float(max_return_cvar) - float(min_cvar_bound)
        )
        position = min(max(position, 0.0), 1.0)
        profile = (
            "defensive"
            if position < 1.0 / 3.0
            else ("balanced" if position < 2.0 / 3.0 else "aggressive")
        )
        interpretation["risk_profile"] = profile
        notes.append(
            f"Risk profile: **{profile}** — portfolio CVaR sits at "
            f"{position * 100:.0f}% of the feasible risk range "
            f"({min_cvar_bound * 100:.2f}% → {max_return_cvar * 100:.2f}%)."
        )
    elif cash_weight > 0.5:
        interpretation["risk_profile"] = "defensive"
        notes.append("Risk profile: **defensive** (cash-dominated).")

    return interpretation


# Re-export math name so callers can do ``optimization.math.sqrt`` if they
# really want; keeps the module surface minimal otherwise.
__all__ = [
    "validate_scenario_matrix",
    "validate_solution_residuals",
    "add_cash_asset",
    "estimate_expected_returns",
    "calculate_portfolio_scenario_metrics",
    "format_weights_table",
    "build_optimization_scenarios",
    "minimize_cvar",
    "maximize_return_with_cvar_constraint",
    "minimize_cvar_for_target_return",
    "maximize_sharpe_ratio",
    "generate_cvar_efficient_frontier",
    "compare_current_vs_optimized",
    "compute_feasible_risk_return_bounds",
    "diagnose_infeasibility",
    "interpret_optimization_result",
]

_ = math  # keep math import for future analytical extensions
