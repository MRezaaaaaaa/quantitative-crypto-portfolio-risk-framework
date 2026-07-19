"""Robust Assumptions Engine (Phase 7).

Pure analytics: no plotting, no Streamlit, no I/O.

This module builds **transparent, robust optimizer inputs**:

* Expected-return estimators — mean, median, trimmed mean, winsorized
  mean, shrinkage-to-zero, zero — plus manual-view blending.
* Robust volatility estimators — sample, winsorized, EWMA (RiskMetrics).
* Robust covariance estimators — sample, EWMA, linear shrinkage toward a
  diagonal or constant-correlation target.

Every estimator works on a generic ``(n_observations × n_assets)`` return
or scenario DataFrame, so the same recipe can be applied to historical
returns, horizon-aggregated returns, or Monte Carlo scenario matrices.
The :class:`AssumptionConfig` dataclass captures a full recipe so the
Streamlit app can store it once and re-apply it to any scenario matrix
(keeping the Robust Assumptions tab and the Optimizer tab consistent).

Design note: the config is a plain dataclass with string-keyed methods so
future estimators (Black-Litterman, Entropy Pooling, regime-conditional
means, scenario reweighting) can be added as new ``method`` values without
breaking existing callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .views import AssetReturnView, apply_manual_expected_return_views

EXPECTED_RETURN_METHODS: tuple[str, ...] = (
    "mean",
    "median",
    "trimmed_mean",
    "winsorized_mean",
    "shrinkage_to_zero",
    "zero",
)

VOLATILITY_METHODS: tuple[str, ...] = ("sample", "winsorized", "ewma")

COVARIANCE_METHODS: tuple[str, ...] = ("sample", "ewma", "shrinkage")

SHRINKAGE_TARGETS: tuple[str, ...] = ("diagonal", "constant_correlation")


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _validate_returns_frame(returns: pd.DataFrame, min_obs: int = 2) -> pd.DataFrame:
    """Validate and clean a returns/scenario DataFrame. Returns the clean copy."""
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns is empty.")
    non_numeric = [
        col
        for col in returns.columns
        if not pd.api.types.is_numeric_dtype(returns[col])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric columns: {non_numeric}")
    clean = returns.dropna()
    if len(clean) < min_obs:
        raise ValueError(
            f"Need at least {min_obs} complete observations, got {len(clean)}."
        )
    return clean.astype(float)


def _validate_proportion(value: float, name: str, upper: float = 0.5) -> float:
    value = float(value)
    if not (0.0 <= value < upper):
        raise ValueError(f"{name} must be in [0, {upper}), got {value}.")
    return value


def _validate_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return value


def _ewma_weights(n: int, decay_lambda: float) -> np.ndarray:
    """Normalized EWMA weights, most recent observation heaviest."""
    lam = float(decay_lambda)
    if not (0.0 < lam < 1.0):
        raise ValueError(f"decay_lambda must be in (0, 1), got {lam}.")
    # index 0 = oldest, index n-1 = most recent
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)
    return w / w.sum()


def winsorize_frame(
    returns: pd.DataFrame, winsor_proportion: float = 0.05
) -> pd.DataFrame:
    """Clip each column at its empirical [p, 1-p] quantiles."""
    p = _validate_proportion(winsor_proportion, "winsor_proportion")
    clean = _validate_returns_frame(returns)
    if p == 0.0:
        return clean
    lower = clean.quantile(p)
    upper = clean.quantile(1.0 - p)
    return clean.clip(lower=lower, upper=upper, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Expected returns
# ─────────────────────────────────────────────────────────────────────────────


def trimmed_mean_returns(
    returns: pd.DataFrame, trim_proportion: float = 0.10
) -> pd.Series:
    """Column-wise trimmed mean (``trim_proportion`` cut from *each* tail)."""
    p = _validate_proportion(trim_proportion, "trim_proportion")
    clean = _validate_returns_frame(returns)
    out = pd.Series(
        {
            col: float(stats.trim_mean(clean[col].to_numpy(), p))
            for col in clean.columns
        },
        name="trimmed_mean",
    )
    return out


def winsorized_mean_returns(
    returns: pd.DataFrame, winsor_proportion: float = 0.05
) -> pd.Series:
    """Column-wise mean after winsorizing at the [p, 1-p] quantiles."""
    out = winsorize_frame(returns, winsor_proportion).mean(axis=0)
    out.name = "winsorized_mean"
    return out.astype(float)


def shrunk_mean_returns(
    returns: pd.DataFrame, shrinkage_weight: float = 0.5
) -> pd.Series:
    """``shrinkage_weight × sample mean`` (complement shrinks to a zero prior)."""
    w = _validate_unit_interval(shrinkage_weight, "shrinkage_weight")
    clean = _validate_returns_frame(returns)
    out = w * clean.mean(axis=0)
    out.name = "shrinkage_to_zero"
    return out.astype(float)


def estimate_expected_returns_robust(
    returns: pd.DataFrame,
    method: str = "mean",
    trim_proportion: float = 0.10,
    winsor_proportion: float = 0.05,
    shrinkage_weight: float = 0.5,
) -> pd.Series:
    """Dispatch a single expected-return estimator over a returns frame.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns or a scenario matrix (rows = observations,
        columns = assets). The estimate is *per observation period*: if
        rows are 7-day returns/scenarios, the result is per 7-day horizon.
    method : one of :data:`EXPECTED_RETURN_METHODS`
    trim_proportion, winsor_proportion, shrinkage_weight : float
        Estimator-specific parameters (ignored by methods that don't use
        them).
    """
    method_lc = str(method).lower()
    clean = _validate_returns_frame(returns)
    if method_lc == "mean":
        out = clean.mean(axis=0)
    elif method_lc == "median":
        out = clean.median(axis=0)
    elif method_lc == "trimmed_mean":
        out = trimmed_mean_returns(clean, trim_proportion)
    elif method_lc == "winsorized_mean":
        out = winsorized_mean_returns(clean, winsor_proportion)
    elif method_lc in ("shrinkage_to_zero", "shrinkage"):
        out = shrunk_mean_returns(clean, shrinkage_weight)
    elif method_lc == "zero":
        out = pd.Series(0.0, index=clean.columns)
    else:
        raise ValueError(
            f"Unsupported expected-return method '{method}'. "
            f"Choose from {list(EXPECTED_RETURN_METHODS)}."
        )
    out.name = "expected_return"
    return out.astype(float)


def build_expected_return_candidates(
    returns: pd.DataFrame,
    trim_proportion: float = 0.10,
    winsor_proportion: float = 0.05,
    shrinkage_weight: float = 0.5,
) -> pd.DataFrame:
    """All candidate expected-return estimates side by side.

    Returns
    -------
    pd.DataFrame
        Indexed by asset. Columns: ``mean``, ``median``, ``trimmed_mean``,
        ``winsorized_mean``, ``shrinkage_to_zero``. Values are per
        observation-period returns (horizon labelling is the caller's job).
    """
    clean = _validate_returns_frame(returns)
    return pd.DataFrame(
        {
            "mean": clean.mean(axis=0),
            "median": clean.median(axis=0),
            "trimmed_mean": trimmed_mean_returns(clean, trim_proportion),
            "winsorized_mean": winsorized_mean_returns(clean, winsor_proportion),
            "shrinkage_to_zero": shrunk_mean_returns(clean, shrinkage_weight),
        }
    ).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Volatility
# ─────────────────────────────────────────────────────────────────────────────


def ewma_volatility(
    returns: pd.DataFrame, decay_lambda: float = 0.94
) -> pd.Series:
    """RiskMetrics-style EWMA volatility per asset.

    Uses the zero-mean squared-return convention:
    ``sigma^2 = sum_i w_i * r_i^2`` with normalized weights
    ``w_i ∝ lambda^(age_i)`` (most recent observation heaviest).
    """
    clean = _validate_returns_frame(returns)
    w = _ewma_weights(len(clean), decay_lambda)
    var = pd.Series(
        w @ (clean.to_numpy(dtype=float) ** 2),
        index=clean.columns,
        name="ewma_volatility",
    )
    return np.sqrt(var).astype(float)


def estimate_volatility_robust(
    returns: pd.DataFrame,
    method: str = "sample",
    winsor_proportion: float = 0.05,
    decay_lambda: float = 0.94,
) -> pd.Series:
    """Dispatch a single per-asset volatility estimator.

    The result is per observation period (daily input ⇒ daily volatility).
    """
    method_lc = str(method).lower()
    clean = _validate_returns_frame(returns)
    if method_lc == "sample":
        out = clean.std(ddof=1)
    elif method_lc == "winsorized":
        out = winsorize_frame(clean, winsor_proportion).std(ddof=1)
    elif method_lc == "ewma":
        out = ewma_volatility(clean, decay_lambda)
    else:
        raise ValueError(
            f"Unsupported volatility method '{method}'. "
            f"Choose from {list(VOLATILITY_METHODS)}."
        )
    out.name = "volatility"
    return out.astype(float)


def build_volatility_table(
    daily_returns: pd.DataFrame,
    horizon_days: int = 1,
    winsor_proportion: float = 0.05,
    decay_lambda: float = 0.94,
) -> pd.DataFrame:
    """Side-by-side per-asset volatility estimates with explicit horizons.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        DAILY returns (this function owns the horizon scaling).
    horizon_days : int
        Horizon for the √t-scaled columns.

    Returns
    -------
    pd.DataFrame
        Indexed by asset. Columns: ``daily_vol``, ``winsorized_daily_vol``,
        ``ewma_daily_vol``, ``horizon_vol``, ``ewma_horizon_vol``,
        ``annualized_vol``. Horizon columns use i.i.d. √t scaling.
    """
    if int(horizon_days) < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}.")
    clean = _validate_returns_frame(daily_returns)
    scale_h = float(np.sqrt(int(horizon_days)))
    scale_y = float(np.sqrt(365.0))

    daily = clean.std(ddof=1)
    winsor = winsorize_frame(clean, winsor_proportion).std(ddof=1)
    ewma = ewma_volatility(clean, decay_lambda)

    return pd.DataFrame(
        {
            "daily_vol": daily,
            "winsorized_daily_vol": winsor,
            "ewma_daily_vol": ewma,
            "horizon_vol": daily * scale_h,
            "ewma_horizon_vol": ewma * scale_h,
            "annualized_vol": daily * scale_y,
        }
    ).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Covariance
# ─────────────────────────────────────────────────────────────────────────────


def ewma_covariance(
    returns: pd.DataFrame, decay_lambda: float = 0.94
) -> pd.DataFrame:
    """EWMA covariance matrix (zero-mean convention, RiskMetrics weights)."""
    clean = _validate_returns_frame(returns)
    x = clean.to_numpy(dtype=float)
    w = _ewma_weights(len(clean), decay_lambda)
    cov = (x * w[:, None]).T @ x
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)


def shrink_covariance(
    sample_cov: pd.DataFrame,
    shrinkage_delta: float = 0.2,
    target: str = "constant_correlation",
) -> pd.DataFrame:
    """Linear shrinkage ``(1 - δ)·S + δ·T`` toward a structured target.

    Parameters
    ----------
    sample_cov : pd.DataFrame
        Square sample covariance matrix.
    shrinkage_delta : float, in [0, 1]
        0 ⇒ pure sample covariance; 1 ⇒ pure target.
    target : {"diagonal", "constant_correlation"}
        * ``diagonal`` — keeps sample variances, zeroes all covariances.
        * ``constant_correlation`` — keeps sample variances, replaces every
          pairwise correlation with the average off-diagonal correlation
          (the Ledoit-Wolf constant-correlation target).
    """
    if not isinstance(sample_cov, pd.DataFrame):
        raise ValueError("sample_cov must be a pandas DataFrame.")
    if sample_cov.shape[0] != sample_cov.shape[1]:
        raise ValueError(f"sample_cov must be square, got {sample_cov.shape}.")
    delta = _validate_unit_interval(shrinkage_delta, "shrinkage_delta")
    target_lc = str(target).lower()

    S = sample_cov.to_numpy(dtype=float)
    S = 0.5 * (S + S.T)
    sd = np.sqrt(np.diag(S))

    if target_lc == "diagonal":
        T = np.diag(np.diag(S))
    elif target_lc == "constant_correlation":
        n = S.shape[0]
        if n < 2 or np.any(sd <= 0):
            T = np.diag(np.diag(S))
        else:
            corr = S / np.outer(sd, sd)
            mean_corr = (corr.sum() - n) / (n * (n - 1))
            T = mean_corr * np.outer(sd, sd)
            np.fill_diagonal(T, np.diag(S))
    else:
        raise ValueError(
            f"Unsupported shrinkage target '{target}'. "
            f"Choose from {list(SHRINKAGE_TARGETS)}."
        )

    shrunk = (1.0 - delta) * S + delta * T
    shrunk = 0.5 * (shrunk + shrunk.T)
    return pd.DataFrame(shrunk, index=sample_cov.index, columns=sample_cov.columns)


def estimate_covariance_robust(
    returns: pd.DataFrame,
    method: str = "sample",
    shrinkage_delta: float = 0.2,
    shrinkage_target: str = "constant_correlation",
    decay_lambda: float = 0.94,
) -> pd.DataFrame:
    """Dispatch a covariance estimator over a returns frame.

    The result covariance is per observation period (daily input ⇒ daily
    covariance); Monte Carlo horizon scaling happens downstream.
    """
    method_lc = str(method).lower()
    clean = _validate_returns_frame(returns)
    if method_lc == "sample":
        return clean.cov().astype(float)
    if method_lc == "ewma":
        return ewma_covariance(clean, decay_lambda)
    if method_lc == "shrinkage":
        return shrink_covariance(
            clean.cov(), shrinkage_delta=shrinkage_delta, target=shrinkage_target
        )
    raise ValueError(
        f"Unsupported covariance method '{method}'. "
        f"Choose from {list(COVARIANCE_METHODS)}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Assumption recipe (config) — the seam the app and optimizer share
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AssumptionConfig:
    """A full, reusable optimizer-assumption recipe.

    Stores *how* to build the expected-return vector (and optionally a
    robust covariance) so the same recipe can be re-applied to whatever
    scenario matrix the optimizer ends up using. Manual views are stored
    as per-horizon expected returns.
    """

    expected_return_method: str = "mean"
    trim_proportion: float = 0.10
    winsor_proportion: float = 0.05
    shrinkage_weight: float = 0.5
    manual_views: dict[str, float] = field(default_factory=dict)
    view_blend_weight: float = 1.0
    covariance_method: str = "sample"
    shrinkage_delta: float = 0.2
    shrinkage_target: str = "constant_correlation"
    decay_lambda: float = 0.94

    def final_expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        """Base estimator → manual-view blend → final per-asset E[r]."""
        base = estimate_expected_returns_robust(
            returns,
            method=self.expected_return_method,
            trim_proportion=self.trim_proportion,
            winsor_proportion=self.winsor_proportion,
            shrinkage_weight=self.shrinkage_weight,
        )
        views = [
            AssetReturnView(asset=a, expected_return=float(v))
            for a, v in self.manual_views.items()
            if a in base.index
        ]
        if views:
            base = apply_manual_expected_return_views(
                base, views, blend_weight=self.view_blend_weight
            )
        return base

    def covariance(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        return estimate_covariance_robust(
            daily_returns,
            method=self.covariance_method,
            shrinkage_delta=self.shrinkage_delta,
            shrinkage_target=self.shrinkage_target,
            decay_lambda=self.decay_lambda,
        )


def build_assumption_table(
    returns: pd.DataFrame,
    config: AssumptionConfig,
) -> pd.DataFrame:
    """The per-asset transparency table for the Robust Assumptions tab.

    Returns
    -------
    pd.DataFrame
        Indexed by asset. Columns: ``mean``, ``median``, ``trimmed_mean``,
        ``winsorized_mean``, ``shrinkage_to_zero``, ``manual_view``
        (NaN where absent), ``final_expected_return``. All values are per
        observation period of ``returns``.
    """
    candidates = build_expected_return_candidates(
        returns,
        trim_proportion=config.trim_proportion,
        winsor_proportion=config.winsor_proportion,
        shrinkage_weight=config.shrinkage_weight,
    )
    table = candidates.copy()
    table["manual_view"] = pd.Series(
        {a: float(v) for a, v in config.manual_views.items()},
        dtype=float,
    ).reindex(table.index)
    table["final_expected_return"] = config.final_expected_returns(returns)
    return table


__all__ = [
    "EXPECTED_RETURN_METHODS",
    "VOLATILITY_METHODS",
    "COVARIANCE_METHODS",
    "SHRINKAGE_TARGETS",
    "AssumptionConfig",
    "winsorize_frame",
    "trimmed_mean_returns",
    "winsorized_mean_returns",
    "shrunk_mean_returns",
    "estimate_expected_returns_robust",
    "build_expected_return_candidates",
    "ewma_volatility",
    "estimate_volatility_robust",
    "build_volatility_table",
    "ewma_covariance",
    "shrink_covariance",
    "estimate_covariance_robust",
    "build_assumption_table",
]
