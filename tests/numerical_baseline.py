"""Deterministic numerical-baseline computation shared by tests and tooling."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from var_cvar_crypto_risk.assumptions import (
    estimate_covariance_robust,
    estimate_expected_returns_robust,
)
from var_cvar_crypto_risk.backtesting import (
    christoffersen_cc_test,
    christoffersen_independence_test,
    kupiec_pof_test,
    rolling_var_forecast,
)
from var_cvar_crypto_risk.cvar_models import gaussian_cvar, historical_cvar
from var_cvar_crypto_risk.monte_carlo import (
    calculate_portfolio_scenario_returns,
    estimate_return_parameters,
    scenario_cvar,
    scenario_var,
    simulate_normal_returns,
)
from var_cvar_crypto_risk.portfolio import calculate_portfolio_returns
from var_cvar_crypto_risk.returns import calculate_simple_returns
from var_cvar_crypto_risk.var_models import (
    cornish_fisher_var,
    gaussian_var,
    historical_var,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "synthetic_daily_prices.csv"
LOCK_PATH = PROJECT_ROOT / "uv.lock"
GOLDEN_PATH = PROJECT_ROOT / "tests" / "fixtures" / "golden_baseline.json"

CONFIDENCE_LEVEL = 0.95
WEIGHTS = pd.Series({"BTC": 0.50, "ETH": 0.30, "SOL": 0.20}, dtype=float)
EWMA_LAMBDA = 0.92
SHRINKAGE_DELTA = 0.20
TRIM_PROPORTION = 0.10
WINSOR_PROPORTION = 0.10
SHRINKAGE_WEIGHT = 0.50
MC_HORIZON_DAYS = 7
MC_SCENARIOS = 4096
MC_SEED = 20260720
BACKTEST_WINDOW = 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _series_dict(series: pd.Series) -> dict[str, float]:
    return {str(key): float(value) for key, value in series.items()}


def _frame_dict(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(row): {str(col): float(frame.loc[row, col]) for col in frame.columns}
        for row in frame.index
    }


def compute_numerical_baseline() -> dict:
    """Compute a deterministic cross-module regression snapshot.

    The input is synthetic and contains no vendor or private data. This helper
    intentionally calls public project functions; it does not duplicate the
    formulas under test.
    """
    prices = pd.read_csv(FIXTURE_PATH, parse_dates=["Date"], index_col="Date")
    returns = calculate_simple_returns(prices)
    portfolio = calculate_portfolio_returns(returns, WEIGHTS)

    expected_return_methods = (
        "mean",
        "median",
        "trimmed_mean",
        "winsorized_mean",
        "shrinkage_to_zero",
    )
    expected_returns = {
        method: _series_dict(
            estimate_expected_returns_robust(
                returns,
                method=method,
                trim_proportion=TRIM_PROPORTION,
                winsor_proportion=WINSOR_PROPORTION,
                shrinkage_weight=SHRINKAGE_WEIGHT,
            )
        )
        for method in expected_return_methods
    }

    sample_covariance = estimate_covariance_robust(returns, method="sample")
    ewma_covariance = estimate_covariance_robust(
        returns,
        method="ewma",
        decay_lambda=EWMA_LAMBDA,
    )
    shrunk_covariance = estimate_covariance_robust(
        returns,
        method="shrinkage",
        shrinkage_delta=SHRINKAGE_DELTA,
        shrinkage_target="constant_correlation",
    )

    parameters = estimate_return_parameters(returns)
    scenarios = simulate_normal_returns(
        mean_vector=parameters["mean_vector"],
        covariance_matrix=sample_covariance,
        n_scenarios=MC_SCENARIOS,
        horizon_days=MC_HORIZON_DAYS,
        random_seed=MC_SEED,
    )
    scenario_portfolio = calculate_portfolio_scenario_returns(scenarios, WEIGHTS)

    backtest = rolling_var_forecast(
        portfolio,
        method="gaussian",
        confidence_level=CONFIDENCE_LEVEL,
        window=BACKTEST_WINDOW,
        horizon_days=1,
        return_method="simple",
        backtest_mode="non_overlapping",
    )
    kupiec = kupiec_pof_test(backtest["breach"], CONFIDENCE_LEVEL)
    independence = christoffersen_independence_test(backtest["breach"])
    conditional_coverage = christoffersen_cc_test(
        backtest["breach"], CONFIDENCE_LEVEL
    )

    return {
        "schema_version": 1,
        "input": {
            "fixture": str(FIXTURE_PATH.relative_to(PROJECT_ROOT)),
            "fixture_sha256": _sha256(FIXTURE_PATH),
            "lock_sha256": _sha256(LOCK_PATH),
            "price_observations": int(len(prices)),
            "return_observations": int(len(returns)),
            "assets": list(returns.columns),
        },
        "parameters": {
            "confidence_level": CONFIDENCE_LEVEL,
            "weights": _series_dict(WEIGHTS),
            "ewma_lambda": EWMA_LAMBDA,
            "shrinkage_delta": SHRINKAGE_DELTA,
            "shrinkage_target": "constant_correlation",
            "trim_proportion_each_tail": TRIM_PROPORTION,
            "winsor_proportion_each_tail": WINSOR_PROPORTION,
            "mean_shrinkage_weight": SHRINKAGE_WEIGHT,
            "mc_horizon_days": MC_HORIZON_DAYS,
            "mc_scenarios": MC_SCENARIOS,
            "mc_seed": MC_SEED,
            "backtest_window": BACKTEST_WINDOW,
        },
        "portfolio": {
            "mean": float(portfolio.mean()),
            "volatility": float(portfolio.std(ddof=1)),
            "minimum": float(portfolio.min()),
            "maximum": float(portfolio.max()),
        },
        "risk": {
            "historical_var": historical_var(portfolio, CONFIDENCE_LEVEL),
            "gaussian_var": gaussian_var(portfolio, CONFIDENCE_LEVEL),
            "cornish_fisher_var": cornish_fisher_var(
                portfolio, CONFIDENCE_LEVEL
            ),
            "historical_cvar": historical_cvar(portfolio, CONFIDENCE_LEVEL),
            "gaussian_cvar": gaussian_cvar(portfolio, CONFIDENCE_LEVEL),
        },
        "expected_returns": expected_returns,
        "covariance": {
            "sample": _frame_dict(sample_covariance),
            "ewma": _frame_dict(ewma_covariance),
            "shrinkage_constant_correlation": _frame_dict(shrunk_covariance),
        },
        "monte_carlo": {
            "portfolio_mean": float(scenario_portfolio.mean()),
            "portfolio_volatility": float(scenario_portfolio.std(ddof=1)),
            "var": scenario_var(scenario_portfolio, CONFIDENCE_LEVEL),
            "cvar": scenario_cvar(scenario_portfolio, CONFIDENCE_LEVEL),
            "minimum": float(scenario_portfolio.min()),
            "maximum": float(scenario_portfolio.max()),
        },
        "backtesting": {
            "forecast_count": int(len(backtest)),
            "breach_count": int(backtest["breach"].sum()),
            "mean_var_forecast": float(backtest["var_forecast"].mean()),
            "kupiec_lr_statistic": float(kupiec["lr_statistic"]),
            "kupiec_p_value": float(kupiec["p_value"]),
            "christoffersen_lr_statistic": float(
                independence["lr_statistic"]
            ),
            "christoffersen_p_value": float(independence["p_value"]),
            "conditional_coverage_lr_statistic": float(
                conditional_coverage["lr_cc"]
            ),
            "conditional_coverage_p_value": float(
                conditional_coverage["p_value"]
            ),
        },
    }
