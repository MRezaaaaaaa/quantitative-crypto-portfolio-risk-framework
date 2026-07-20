"""Monte Carlo scenario engine for multi-asset crypto portfolios.

This module is pure analytics: no plotting, no Streamlit, no I/O.

Capabilities
------------
* Estimate mean / covariance / volatility from historical asset returns.
* Simulate multivariate scenario returns under Normal or Student-t laws.
* Aggregate scenario asset returns into portfolio scenario returns.
* Compute scenario-based VaR and CVaR in signed loss space.
* Simulate portfolio value paths over a forward horizon.
* Compare Normal vs Student-t Monte Carlo in one call.
* Compare every available risk method (Historical / Gaussian /
  Cornish-Fisher / Normal MC / Student-t MC) in one DataFrame.

Conventions
-----------
* VaR / CVaR values are decimal signed losses: positive = loss, zero =
  break-even, and negative = gain at the measured tail threshold.
* For ``horizon_days > 1`` the parametric drift and dispersion of the
  multivariate simulators are scaled as ``mean * horizon`` and
  ``cov * horizon`` (i.i.d. approximation).
* Random seeds make every simulation reproducible.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .cvar_models import calculate_cvar
from .risk_conventions import (
    loss_value_to_money,
    return_threshold_to_loss_value,
)
from .var_models import calculate_var


_SUPPORTED_DISTRIBUTIONS: tuple[str, ...] = ("normal", "student_t")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter estimation
# ─────────────────────────────────────────────────────────────────────────────


def estimate_return_parameters(
    returns: pd.DataFrame,
    annualize: bool = False,
    periods_per_year: int = 365,
) -> dict:
    """Estimate mean vector, covariance, correlation, and volatility.

    Parameters
    ----------
    returns : pd.DataFrame
        Columns = assets, rows = observations.
    annualize : bool, optional
        If ``True``, scale mean by ``periods_per_year`` and covariance by
        ``periods_per_year`` (so volatility scales by ``sqrt``).
    periods_per_year : int, optional
        Defaults to ``365`` for crypto.

    Returns
    -------
    dict
        Keys ``mean_vector`` (pd.Series), ``covariance_matrix`` (pd.DataFrame),
        ``correlation_matrix`` (pd.DataFrame), ``volatility_vector``
        (pd.Series), ``n_observations`` (int), ``assets`` (list[str]).

    Raises
    ------
    ValueError
        On empty input, fewer than 2 observations, non-numeric columns,
        or all-NaN columns.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a pd.DataFrame.")
    if returns.empty:
        raise ValueError("returns is empty.")
    if returns.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 observations, got {returns.shape[0]}."
        )
    if not all(pd.api.types.is_numeric_dtype(returns[col]) for col in returns.columns):
        raise ValueError("All columns of returns must be numeric.")
    if returns.isna().all(axis=0).any():
        bad = returns.columns[returns.isna().all(axis=0)].tolist()
        raise ValueError(f"All-NaN column(s) in returns: {bad}.")

    clean = returns.dropna()
    if len(clean) < 2:
        raise ValueError(
            "After dropping NaNs, fewer than 2 observations remain."
        )

    mean_vector = clean.mean()
    cov = clean.cov()
    vol = clean.std(ddof=1)
    corr = clean.corr()

    if annualize:
        mean_vector = mean_vector * float(periods_per_year)
        cov = cov * float(periods_per_year)
        vol = vol * math.sqrt(float(periods_per_year))

    return {
        "mean_vector": mean_vector.astype(float),
        "covariance_matrix": cov.astype(float),
        "correlation_matrix": corr.astype(float),
        "volatility_vector": vol.astype(float),
        "n_observations": int(len(clean)),
        "assets": list(clean.columns),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulation primitives
# ─────────────────────────────────────────────────────────────────────────────


def _validate_simulation_inputs(
    mean_vector: pd.Series,
    covariance_matrix: pd.DataFrame,
    n_scenarios: int,
    horizon_days: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validate shapes and return aligned numpy arrays + asset list."""
    if n_scenarios <= 0:
        raise ValueError(f"n_scenarios must be > 0, got {n_scenarios}.")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}.")

    assets = list(mean_vector.index)
    cov_assets = list(covariance_matrix.index)
    if cov_assets != assets or list(covariance_matrix.columns) != assets:
        cov_aligned = covariance_matrix.reindex(index=assets, columns=assets)
        if cov_aligned.isna().any().any():
            raise ValueError(
                "covariance_matrix indices do not align with mean_vector."
            )
        covariance_matrix = cov_aligned

    mu = mean_vector.to_numpy(dtype=float)
    cov = covariance_matrix.to_numpy(dtype=float)

    if cov.shape != (mu.size, mu.size):
        raise ValueError(
            f"covariance_matrix shape {cov.shape} does not match "
            f"mean_vector size {mu.size}."
        )

    # Symmetrize defensively.
    cov = 0.5 * (cov + cov.T)
    return mu, cov, assets


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Cholesky factor of ``cov``, with diagonal jitter on failure."""
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * np.eye(cov.shape[0])
        for k in range(10):
            try:
                return np.linalg.cholesky(cov + (10**k) * jitter)
            except np.linalg.LinAlgError:
                continue
        raise


def simulate_normal_returns(
    mean_vector: pd.Series,
    covariance_matrix: pd.DataFrame,
    n_scenarios: int = 5000,
    horizon_days: int = 1,
    random_seed: int | None = 42,
) -> pd.DataFrame:
    """Generate multivariate Normal return scenarios.

    For ``horizon_days > 1``, mean and covariance are scaled linearly with
    the horizon (i.i.d. assumption).

    Returns
    -------
    pd.DataFrame
        Shape ``n_scenarios × n_assets``. Index = ``scenario_i`` strings.
    """
    mu, cov, assets = _validate_simulation_inputs(
        mean_vector, covariance_matrix, n_scenarios, horizon_days
    )

    mu_h = mu * float(horizon_days)
    cov_h = cov * float(horizon_days)

    rng = np.random.default_rng(random_seed)
    L = _safe_cholesky(cov_h)
    z = rng.standard_normal(size=(int(n_scenarios), mu.size))
    samples = mu_h + z @ L.T

    return pd.DataFrame(
        samples,
        columns=assets,
        index=[f"scenario_{i + 1}" for i in range(int(n_scenarios))],
    )


def simulate_student_t_returns(
    mean_vector: pd.Series,
    covariance_matrix: pd.DataFrame,
    df: float = 5,
    n_scenarios: int = 5000,
    horizon_days: int = 1,
    random_seed: int | None = 42,
) -> pd.DataFrame:
    """Generate multivariate Student-t return scenarios with fat tails.

    The samples are scaled so that, in the limit, their covariance matches
    the supplied ``covariance_matrix`` (modulo the i.i.d. ``horizon_days``
    scaling).

    Parameters
    ----------
    df : float
        Degrees of freedom. Must be ``> 2`` (so variance exists).

    Returns
    -------
    pd.DataFrame
        Shape ``n_scenarios × n_assets``. Index = ``scenario_i`` strings.
    """
    if df <= 2:
        raise ValueError(f"df must be > 2 for finite variance, got {df}.")

    mu, cov, assets = _validate_simulation_inputs(
        mean_vector, covariance_matrix, n_scenarios, horizon_days
    )

    mu_h = mu * float(horizon_days)
    cov_h = cov * float(horizon_days)
    scale_cov = cov_h * (df - 2.0) / df  # so that resulting cov ≈ cov_h

    rng = np.random.default_rng(random_seed)
    L = _safe_cholesky(scale_cov)
    z = rng.standard_normal(size=(int(n_scenarios), mu.size))
    chi2 = rng.chisquare(df=df, size=int(n_scenarios)) / df
    inv_scale = 1.0 / np.sqrt(chi2)[:, None]
    samples = mu_h + inv_scale * (z @ L.T)

    return pd.DataFrame(
        samples,
        columns=assets,
        index=[f"scenario_{i + 1}" for i in range(int(n_scenarios))],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio aggregation and scenario risk
# ─────────────────────────────────────────────────────────────────────────────


def calculate_portfolio_scenario_returns(
    scenario_returns: pd.DataFrame,
    weights: pd.Series,
    return_method: str = "simple",
) -> pd.Series:
    """Aggregate asset scenario returns into portfolio scenario returns.

    Parameters
    ----------
    scenario_returns : pd.DataFrame
        Scenarios as rows, assets as columns.
    weights : pd.Series
        Indexed by asset symbol. Must cover every column.
    return_method : {"simple"}
        Scenario aggregation is arithmetic and therefore requires simple
        asset returns.

    Returns
    -------
    pd.Series
        Portfolio scenario returns, indexed by ``scenario_returns.index``.
    """
    if not isinstance(scenario_returns, pd.DataFrame):
        raise ValueError("scenario_returns must be a pd.DataFrame.")
    if scenario_returns.empty:
        raise ValueError("scenario_returns is empty.")
    if not isinstance(weights, pd.Series):
        raise ValueError("weights must be a pd.Series.")
    if return_method != "simple":
        raise ValueError(
            "calculate_portfolio_scenario_returns requires simple-return "
            "inputs; convert log returns before calling."
        )

    aligned = weights.reindex(scenario_returns.columns)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index.tolist()
        raise ValueError(f"weights missing for scenario columns: {missing}")

    portfolio = scenario_returns.to_numpy(dtype=float) @ aligned.to_numpy(dtype=float)
    return pd.Series(
        portfolio,
        index=scenario_returns.index,
        name="portfolio_scenario_return",
    )


def scenario_var(
    scenario_returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """Scenario-based VaR as a signed decimal loss value."""
    if not isinstance(scenario_returns, pd.Series):
        raise ValueError("scenario_returns must be a pd.Series.")
    if scenario_returns.empty:
        raise ValueError("scenario_returns is empty.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}."
        )
    alpha = 1.0 - float(confidence_level)
    threshold = float(np.quantile(scenario_returns.to_numpy(dtype=float), alpha))
    return return_threshold_to_loss_value(threshold)


def scenario_cvar(
    scenario_returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """Scenario-based CVaR in signed loss space; ``CVaR >= VaR``."""
    if not isinstance(scenario_returns, pd.Series):
        raise ValueError("scenario_returns must be a pd.Series.")
    if scenario_returns.empty:
        raise ValueError("scenario_returns is empty.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}."
        )
    alpha = 1.0 - float(confidence_level)
    values = scenario_returns.to_numpy(dtype=float)
    threshold = float(np.quantile(values, alpha))
    tail = values[values <= threshold]
    if tail.size == 0:
        raise ValueError(
            "No tail observations found for scenario CVaR. "
            "Increase n_scenarios or lower confidence_level."
        )
    return return_threshold_to_loss_value(float(tail.mean()))


# ─────────────────────────────────────────────────────────────────────────────
# Summaries
# ─────────────────────────────────────────────────────────────────────────────


def monte_carlo_risk_summary(
    scenario_returns: pd.Series,
    confidence_level: float = 0.95,
    initial_capital: float | None = None,
    label: str = "Monte Carlo",
) -> pd.DataFrame:
    """Build a one-method risk summary table from scenario returns.

    Returns
    -------
    pd.DataFrame
        Columns ``["Metric", "Value", "Unit"]``.
    """
    if scenario_returns.empty:
        raise ValueError("scenario_returns is empty.")

    var = scenario_var(scenario_returns, confidence_level)
    cvar = scenario_cvar(scenario_returns, confidence_level)
    confidence_pct = float(confidence_level) * 100.0

    rows: list[dict] = [
        {"Metric": "Method", "Value": label, "Unit": ""},
        {
            "Metric": "Confidence Level",
            "Value": confidence_pct,
            "Unit": "%",
        },
        {
            "Metric": "Number of Scenarios",
            "Value": int(scenario_returns.size),
            "Unit": "",
        },
        {
            "Metric": "Mean Scenario Return",
            "Value": float(scenario_returns.mean()) * 100.0,
            "Unit": "%",
        },
        {
            "Metric": "Volatility",
            "Value": float(scenario_returns.std(ddof=1)) * 100.0,
            "Unit": "%",
        },
        {
            "Metric": f"{label} VaR {confidence_pct:.0f}%",
            "Value": var * 100.0,
            "Unit": "%",
        },
        {
            "Metric": f"{label} CVaR {confidence_pct:.0f}%",
            "Value": cvar * 100.0,
            "Unit": "%",
        },
        {
            "Metric": "Worst Scenario Return",
            "Value": float(scenario_returns.min()) * 100.0,
            "Unit": "%",
        },
        {
            "Metric": "Best Scenario Return",
            "Value": float(scenario_returns.max()) * 100.0,
            "Unit": "%",
        },
    ]

    if initial_capital is not None:
        rows.append(
            {
                "Metric": f"{label} Money VaR {confidence_pct:.0f}%",
                "Value": loss_value_to_money(var, initial_capital),
                "Unit": "USD",
            }
        )
        rows.append(
            {
                "Metric": f"{label} Money CVaR {confidence_pct:.0f}%",
                "Value": loss_value_to_money(cvar, initial_capital),
                "Unit": "USD",
            }
        )

    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio value paths
# ─────────────────────────────────────────────────────────────────────────────


def simulate_portfolio_paths(
    portfolio_daily_mean: float,
    portfolio_daily_volatility: float,
    initial_value: float = 100_000.0,
    n_paths: int = 500,
    horizon_days: int = 30,
    distribution: str = "normal",
    df: float = 5,
    random_seed: int | None = 42,
    return_method: str = "simple",
) -> pd.DataFrame:
    """Simulate portfolio value paths over a forward horizon.

    The first row equals ``initial_value`` for every path.

    Parameters
    ----------
    portfolio_daily_mean : float
        Daily mean return of the portfolio.
    portfolio_daily_volatility : float
        Daily standard deviation (must be ``> 0``).
    initial_value : float, optional
    n_paths : int, optional
    horizon_days : int, optional
    distribution : {"normal", "student_t"}, optional
    df : float, optional
        Student-t degrees of freedom. Used only when
        ``distribution == "student_t"``. Must be ``> 2``.
    random_seed : int or None, optional
    return_method : {"simple"}
        Simulated innovations are compounded arithmetically into wealth paths.
    """
    if initial_value <= 0:
        raise ValueError(f"initial_value must be > 0, got {initial_value}.")
    if return_method != "simple":
        raise ValueError(
            "simulate_portfolio_paths requires simple-return inputs and "
            "innovations; log-return compounding is not accepted here."
        )
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}.")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}.")
    if distribution not in _SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. "
            f"Use one of: {list(_SUPPORTED_DISTRIBUTIONS)}."
        )
    if portfolio_daily_volatility < 0:
        raise ValueError(
            f"portfolio_daily_volatility must be >= 0, got {portfolio_daily_volatility}."
        )

    rng = np.random.default_rng(random_seed)
    mu = float(portfolio_daily_mean)
    sigma = float(portfolio_daily_volatility)
    h = int(horizon_days)
    n = int(n_paths)

    if distribution == "normal":
        shocks = rng.standard_normal(size=(h, n))
        daily_returns = mu + sigma * shocks
    else:
        if df <= 2:
            raise ValueError(f"df must be > 2, got {df}.")
        z = rng.standard_normal(size=(h, n))
        chi2 = rng.chisquare(df=df, size=(h, n)) / df
        t_shock = z / np.sqrt(chi2)
        scale = math.sqrt((df - 2.0) / df)
        daily_returns = mu + sigma * scale * t_shock

    growth = np.cumprod(1.0 + daily_returns, axis=0)
    paths = np.vstack([np.ones((1, n)), growth]) * float(initial_value)

    return pd.DataFrame(
        paths,
        index=pd.RangeIndex(start=0, stop=h + 1, name="day"),
        columns=[f"path_{i + 1}" for i in range(n)],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Normal vs Student-t comparison
# ─────────────────────────────────────────────────────────────────────────────


def compare_monte_carlo_distributions(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float = 0.95,
    n_scenarios: int = 5000,
    horizon_days: int = 1,
    df: float = 5,
    random_seed: int | None = 42,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Compare Normal and Student-t Monte Carlo on the same portfolio.

    Returns
    -------
    tuple
        ``(scenario_returns_by_distribution, comparison_df)``.
    """
    params = estimate_return_parameters(returns)
    mean_vector = params["mean_vector"]
    cov = params["covariance_matrix"]

    normal_scenarios = simulate_normal_returns(
        mean_vector=mean_vector,
        covariance_matrix=cov,
        n_scenarios=n_scenarios,
        horizon_days=horizon_days,
        random_seed=random_seed,
    )
    student_scenarios = simulate_student_t_returns(
        mean_vector=mean_vector,
        covariance_matrix=cov,
        df=df,
        n_scenarios=n_scenarios,
        horizon_days=horizon_days,
        random_seed=random_seed,
    )

    normal_pf = calculate_portfolio_scenario_returns(normal_scenarios, weights)
    student_pf = calculate_portfolio_scenario_returns(student_scenarios, weights)

    rows = []
    for name, series in (("normal", normal_pf), ("student_t", student_pf)):
        rows.append(
            {
                "distribution": name,
                "confidence_level": float(confidence_level),
                "horizon_days": int(horizon_days),
                "n_scenarios": int(n_scenarios),
                "VaR": scenario_var(series, confidence_level),
                "CVaR": scenario_cvar(series, confidence_level),
                "mean_return": float(series.mean()),
                "volatility": float(series.std(ddof=1)),
                "worst_return": float(series.min()),
                "best_return": float(series.max()),
            }
        )

    return {"normal": normal_pf, "student_t": student_pf}, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-method comparison
# ─────────────────────────────────────────────────────────────────────────────


def _horizon_aggregate_simple(values: np.ndarray, h: int) -> np.ndarray:
    if h <= 1:
        return values
    n = len(values)
    if n < h:
        return values
    out = np.empty(n - h + 1, dtype=float)
    for j in range(n - h + 1):
        out[j] = float(np.prod(1.0 + values[j : j + h]) - 1.0)
    return out


def compare_all_risk_methods(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    n_scenarios: int = 5000,
    student_t_df: float = 5,
    random_seed: int | None = 42,
    return_method: str = "simple",
) -> pd.DataFrame:
    """Compare Historical / Gaussian / Cornish-Fisher / Normal MC / Student-t MC.

    For ``horizon_days > 1``:

    * Historical / Gaussian / Cornish-Fisher are computed on rolling h-day
      aggregated portfolio returns (no sqrt-time scaling).
    * Normal MC and Student-t MC use ``horizon_days`` directly in scenario
      generation.

    Returns
    -------
    pd.DataFrame
        Columns ``Method``, ``VaR``, ``CVaR``, ``Confidence Level``,
        ``Horizon Days``, ``Notes``. ``VaR`` and ``CVaR`` are signed decimal
        loss values (e.g. ``0.082`` = 8.2% loss; ``-0.02`` = 2% gain).
    """
    if return_method != "simple":
        raise ValueError(
            "compare_all_risk_methods requires simple-return inputs; convert "
            "log returns before calling."
        )
    h = int(horizon_days)
    if h < 1:
        raise ValueError(f"horizon_days must be >= 1, got {h}.")

    values = portfolio_returns.dropna().to_numpy(dtype=float)
    if h == 1:
        agg = values
        agg_note_suffix = ""
    else:
        agg = _horizon_aggregate_simple(values, h)
        agg_note_suffix = f" on rolling {h}-day returns"

    agg_series = pd.Series(agg, name="aggregated_returns")

    rows: list[dict] = []

    def _safe(fn):
        try:
            return float(fn())
        except Exception:  # noqa: BLE001
            return float("nan")

    rows.append(
        {
            "Method": "Historical",
            "VaR": _safe(lambda: calculate_var(agg_series, "historical", confidence_level)),
            "CVaR": _safe(lambda: calculate_cvar(agg_series, "historical", confidence_level)),
            "Confidence Level": float(confidence_level),
            "Horizon Days": h,
            "Notes": f"Empirical left-tail percentile{agg_note_suffix}.",
        }
    )
    rows.append(
        {
            "Method": "Gaussian",
            "VaR": _safe(lambda: calculate_var(agg_series, "gaussian", confidence_level)),
            "CVaR": _safe(lambda: calculate_cvar(agg_series, "gaussian", confidence_level)),
            "Confidence Level": float(confidence_level),
            "Horizon Days": h,
            "Notes": f"Parametric Normal{agg_note_suffix}.",
        }
    )
    rows.append(
        {
            "Method": "Cornish-Fisher",
            "VaR": _safe(
                lambda: calculate_var(agg_series, "cornish_fisher", confidence_level)
            ),
            "CVaR": float("nan"),
            "Confidence Level": float(confidence_level),
            "Horizon Days": h,
            "Notes": (
                f"Cornish-Fisher expansion{agg_note_suffix}. CVaR not implemented."
            ),
        }
    )

    try:
        params = estimate_return_parameters(asset_returns)
        normal_scen = simulate_normal_returns(
            mean_vector=params["mean_vector"],
            covariance_matrix=params["covariance_matrix"],
            n_scenarios=n_scenarios,
            horizon_days=h,
            random_seed=random_seed,
        )
        normal_pf = calculate_portfolio_scenario_returns(normal_scen, weights)
        student_scen = simulate_student_t_returns(
            mean_vector=params["mean_vector"],
            covariance_matrix=params["covariance_matrix"],
            df=student_t_df,
            n_scenarios=n_scenarios,
            horizon_days=h,
            random_seed=random_seed,
        )
        student_pf = calculate_portfolio_scenario_returns(student_scen, weights)
        rows.append(
            {
                "Method": "Normal Monte Carlo",
                "VaR": scenario_var(normal_pf, confidence_level),
                "CVaR": scenario_cvar(normal_pf, confidence_level),
                "Confidence Level": float(confidence_level),
                "Horizon Days": h,
                "Notes": f"Multivariate Normal, {n_scenarios:,} scenarios, h={h}.",
            }
        )
        rows.append(
            {
                "Method": "Student-t Monte Carlo",
                "VaR": scenario_var(student_pf, confidence_level),
                "CVaR": scenario_cvar(student_pf, confidence_level),
                "Confidence Level": float(confidence_level),
                "Horizon Days": h,
                "Notes": (
                    f"Multivariate Student-t (df={student_t_df}), "
                    f"{n_scenarios:,} scenarios, h={h}."
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            {
                "Method": "Normal Monte Carlo",
                "VaR": float("nan"),
                "CVaR": float("nan"),
                "Confidence Level": float(confidence_level),
                "Horizon Days": h,
                "Notes": f"Simulation failed: {exc}",
            }
        )
        rows.append(
            {
                "Method": "Student-t Monte Carlo",
                "VaR": float("nan"),
                "CVaR": float("nan"),
                "Confidence Level": float(confidence_level),
                "Horizon Days": h,
                "Notes": f"Simulation failed: {exc}",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Method",
            "VaR",
            "CVaR",
            "Confidence Level",
            "Horizon Days",
            "Notes",
        ],
    )
