"""VaR backtesting and model validation.

This module implements rolling one-step-ahead VaR forecasting with no
look-ahead bias and a suite of statistical coverage tests aligned with the
Basel III market risk framework:

* Kupiec Proportion of Failures (POF) test
* Christoffersen Independence test
* Christoffersen Conditional Coverage (CC) test

A two-tier traffic light system (Basel III absolute counts and a generalised
ratio-based variant) summarises model adequacy.

All VaR values consumed and produced by this module are positive loss numbers
(e.g. ``0.042`` means a 4.2% loss threshold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .var_models import calculate_var


_SUPPORTED_METHODS: tuple[str, ...] = ("historical", "gaussian", "cornish_fisher")
_SUPPORTED_TRAFFIC_LIGHT_MODES: tuple[str, ...] = ("auto", "basel3", "rate_based")
_SUPPORTED_RETURN_METHODS: tuple[str, ...] = ("simple", "log")
_SUPPORTED_BACKTEST_MODES: tuple[str, ...] = ("overlapping", "non_overlapping")
_DEFAULT_SIGNIFICANCE: float = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Horizon-aware realised returns
# ─────────────────────────────────────────────────────────────────────────────


def calculate_realized_horizon_returns(
    returns: pd.Series,
    horizon_days: int = 1,
    method: str = "simple",
) -> pd.Series:
    """Compute realised forward h-day portfolio returns indexed by forecast date.

    For each index ``t`` in the input series, the realised forward h-day return
    is computed from the next ``h`` observations strictly **after** ``t``:

    - ``simple`` : ``prod(1 + r[t+1..t+h]) - 1``
    - ``log``    : ``sum(r[t+1..t+h])``

    The last ``h`` observations of the input do not have a full forward
    window and are dropped.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns with a DatetimeIndex. NaN values are not
        allowed.
    horizon_days : int, optional
        Forward horizon. ``1`` returns the raw next-day return at each ``t``.
    method : {"simple", "log"}, optional
        Compounding convention.

    Returns
    -------
    pd.Series
        Index = ``returns.index[:-horizon_days]`` (forecast start dates).
        Values = realised forward h-day return.

    Raises
    ------
    ValueError
        If inputs are invalid or there are fewer than ``horizon_days + 1``
        observations.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pd.Series.")
    if returns.empty:
        raise ValueError("returns is empty.")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}.")
    if method not in _SUPPORTED_RETURN_METHODS:
        raise ValueError(
            f"Unsupported return method '{method}'. "
            f"Use one of: {list(_SUPPORTED_RETURN_METHODS)}."
        )
    if returns.isna().any():
        raise ValueError("returns contains NaN values; clean before calling.")
    if len(returns) <= horizon_days:
        raise ValueError(
            f"Need at least horizon_days+1={horizon_days + 1} observations, "
            f"got {len(returns)}."
        )

    values = returns.to_numpy(dtype=float)
    n = len(values)
    h = int(horizon_days)
    out = np.empty(n - h, dtype=float)

    if method == "simple":
        for t in range(n - h):
            out[t] = float(np.prod(1.0 + values[t + 1 : t + 1 + h]) - 1.0)
    else:  # log
        for t in range(n - h):
            out[t] = float(np.sum(values[t + 1 : t + 1 + h]))

    return pd.Series(
        out,
        index=returns.index[: n - h],
        name=f"realized_{h}d_return",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rolling VaR forecasting
# ─────────────────────────────────────────────────────────────────────────────


def _aggregate_horizon_returns(
    values: np.ndarray, horizon_days: int, return_method: str
) -> np.ndarray:
    """Rolling h-day aggregated returns inside a 1-D return array.

    Returns ``len(values) - horizon_days + 1`` aggregated returns.
    """
    h = int(horizon_days)
    n = len(values)
    if h <= 1:
        return values.copy()
    out = np.empty(n - h + 1, dtype=float)
    if return_method == "simple":
        for j in range(n - h + 1):
            out[j] = float(np.prod(1.0 + values[j : j + h]) - 1.0)
    else:
        for j in range(n - h + 1):
            out[j] = float(np.sum(values[j : j + h]))
    return out


def rolling_var_forecast(
    returns: pd.Series,
    method: str = "historical",
    confidence_level: float = 0.95,
    window: int = 252,
    horizon_days: int = 1,
    return_method: str = "simple",
    backtest_mode: str = "overlapping",
) -> pd.DataFrame:
    """Generate rolling h-day VaR forecasts with no look-ahead bias.

    For each date ``t`` after the first ``window`` observations the forecast
    uses ``returns[t-window : t]`` (strictly excluding ``t``) to compute the
    h-day VaR estimate. For ``horizon_days > 1`` the lookback window is
    aggregated into rolling h-day returns first, then VaR is computed on
    those h-day returns — no scaling is applied. The realised h-day return
    is the realised portfolio return over ``returns[t : t + horizon_days]``
    so that for ``horizon_days == 1`` the function reproduces the original
    one-step-ahead behaviour.

    ``backtest_mode`` controls how forecast dates are stepped:

    - ``"overlapping"`` (default): step size 1 — every date is a forecast
      date and, for ``horizon_days > 1``, realised h-day returns overlap.
    - ``"non_overlapping"``: step size ``horizon_days`` — forecast dates are
      spaced one full horizon apart so realised h-day returns do not overlap.
      This yields roughly ``1 / horizon_days`` as many observations and is
      more appropriate for the independence-based Christoffersen tests. For
      ``horizon_days == 1`` both modes are identical.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns with a DatetimeIndex. Must be clean (no NaN values).
    method : {"historical", "gaussian", "cornish_fisher"}, optional
    confidence_level : float, optional
        Must be strictly in ``(0, 1)``.
    window : int, optional
        Number of past observations used per forecast. Must be ``>= 30``.
    horizon_days : int, optional
        Forecast horizon in days. ``1`` reproduces the one-step-ahead behaviour.
    return_method : {"simple", "log"}, optional
        Compounding convention for h-day return aggregation.

    Returns
    -------
    pd.DataFrame
        Index is the DatetimeIndex of forecast dates. Columns:

        - ``actual_return`` : float — realised h-day return starting at the date.
        - ``var_forecast``  : float — positive h-day VaR estimate.
        - ``breach``        : bool  — ``actual_return < -var_forecast``.
        - ``horizon_days``  : int   — the configured horizon.
        - ``method``        : str   — the VaR method used.

    Raises
    ------
    ValueError
        If inputs are invalid or there are too few observations.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pd.Series.")
    if returns.empty:
        raise ValueError("returns is empty.")
    if window < 30:
        raise ValueError(f"window must be >= 30, got {window}.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}."
        )
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Use one of: {list(_SUPPORTED_METHODS)}."
        )
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}.")
    if return_method not in _SUPPORTED_RETURN_METHODS:
        raise ValueError(
            f"Unsupported return_method '{return_method}'. "
            f"Use one of: {list(_SUPPORTED_RETURN_METHODS)}."
        )
    if backtest_mode not in _SUPPORTED_BACKTEST_MODES:
        raise ValueError(
            f"Unsupported backtest_mode '{backtest_mode}'. "
            f"Use one of: {list(_SUPPORTED_BACKTEST_MODES)}."
        )
    if window < horizon_days:
        raise ValueError(
            f"window={window} must be >= horizon_days={horizon_days}."
        )

    h = int(horizon_days)
    min_required = window + h
    if len(returns) < min_required:
        raise ValueError(
            f"returns has {len(returns)} observations but "
            f"window+horizon_days={min_required} are required for at least "
            "one forecast."
        )

    values = returns.to_numpy(dtype=float)
    index = returns.index
    n = len(values)
    step_size = 1 if backtest_mode == "overlapping" else h

    forecast_positions = list(range(window, n - h + 1, step_size))
    n_forecasts = len(forecast_positions)
    actual = np.empty(n_forecasts, dtype=float)
    var_fc = np.empty(n_forecasts, dtype=float)

    for i, t in enumerate(forecast_positions):
        lookback = values[t - window : t]
        if h == 1:
            estimator_input = lookback
        else:
            estimator_input = _aggregate_horizon_returns(
                lookback, horizon_days=h, return_method=return_method
            )
        var_fc[i] = float(
            calculate_var(pd.Series(estimator_input), method, confidence_level)
        )

        future = values[t : t + h]
        if h == 1:
            actual[i] = float(future[0])
        elif return_method == "simple":
            actual[i] = float(np.prod(1.0 + future) - 1.0)
        else:
            actual[i] = float(np.sum(future))

    breach = actual < -var_fc

    return pd.DataFrame(
        {
            "actual_return": actual,
            "var_forecast": var_fc,
            "breach": breach,
            "horizon_days": np.full(n_forecasts, h, dtype=int),
            "method": [method] * n_forecasts,
            "backtest_mode": [backtest_mode] * n_forecasts,
            "step_size": np.full(n_forecasts, step_size, dtype=int),
        },
        index=index[forecast_positions],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Breach analytics
# ─────────────────────────────────────────────────────────────────────────────


def calculate_var_breaches(
    backtest_df: pd.DataFrame,
    confidence_level: float,
) -> dict:
    """Summarise breach statistics from a :func:`rolling_var_forecast` output.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Must contain columns ``actual_return``, ``var_forecast``, ``breach``.
    confidence_level : float

    Returns
    -------
    dict
        Keys: ``observations``, ``actual_breaches``, ``expected_breaches``,
        ``expected_breach_rate``, ``actual_breach_rate``, ``breach_ratio``,
        ``confidence_level``.

    Raises
    ------
    ValueError
        If ``observations`` is zero.
    """
    required = {"actual_return", "var_forecast", "breach"}
    missing = required - set(backtest_df.columns)
    if missing:
        raise ValueError(f"backtest_df is missing required columns: {missing}.")

    observations = int(len(backtest_df))
    if observations == 0:
        raise ValueError("backtest_df has zero observations.")

    actual_breaches = int(backtest_df["breach"].astype(bool).sum())
    expected_breach_rate = float(1.0 - confidence_level)
    expected_breaches = float(observations) * expected_breach_rate
    actual_breach_rate = actual_breaches / observations
    breach_ratio = (
        actual_breaches / expected_breaches
        if expected_breaches > 0
        else float("nan")
    )

    return {
        "observations": observations,
        "actual_breaches": actual_breaches,
        "expected_breaches": expected_breaches,
        "expected_breach_rate": expected_breach_rate,
        "actual_breach_rate": actual_breach_rate,
        "breach_ratio": breach_ratio,
        "confidence_level": float(confidence_level),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────


def _xlogx(count: float, prob: float) -> float:
    """Return ``count * log(prob)`` with the convention ``0 * log(0) = 0``."""
    if count == 0:
        return 0.0
    if prob <= 0.0:
        return -np.inf
    return float(count) * float(np.log(prob))


def _to_breach_int_array(breaches: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(breaches, pd.Series):
        arr = breaches.to_numpy()
    else:
        arr = np.asarray(breaches)
    if arr.dtype == bool:
        return arr.astype(np.int8)
    return (arr != 0).astype(np.int8)


def kupiec_pof_test(
    breaches: pd.Series | np.ndarray,
    confidence_level: float,
) -> dict:
    """Kupiec Proportion of Failures (POF) test.

    Tests whether the observed breach frequency is statistically consistent
    with the model's stated confidence level.

    Likelihood ratio statistic::

        LR_pof = -2 * ln(
            [(1 - p)^(n - x) * p^x] / [(1 - x/n)^(n - x) * (x/n)^x]
        )

    with ``p = 1 - confidence_level`` and ``p_value = 1 - chi2.cdf(LR_pof, df=1)``.

    Parameters
    ----------
    breaches : pd.Series or np.ndarray
        Boolean (or 0/1 integer) array. ``True`` denotes a breach.
    confidence_level : float

    Returns
    -------
    dict
        Keys: ``test_name``, ``observations``, ``breaches``,
        ``expected_breach_probability``, ``observed_breach_probability``,
        ``lr_statistic``, ``p_value``, ``pass_test``, ``interpretation``.
    """
    arr = _to_breach_int_array(breaches)
    n = int(arr.size)
    x = int(arr.sum())
    p = float(1.0 - confidence_level)

    if n == 0:
        return {
            "test_name": "Kupiec POF",
            "observations": 0,
            "breaches": 0,
            "expected_breach_probability": p,
            "observed_breach_probability": float("nan"),
            "lr_statistic": float("nan"),
            "p_value": float("nan"),
            "pass_test": None,
            "interpretation": "Insufficient data for Kupiec POF test.",
        }

    p_hat = x / n

    log_num = _xlogx(n - x, 1.0 - p) + _xlogx(x, p)
    log_den = _xlogx(n - x, 1.0 - p_hat) + _xlogx(x, p_hat)
    lr_stat = float(-2.0 * (log_num - log_den))
    if not np.isfinite(lr_stat):
        lr_stat = float("nan")
        p_value = float("nan")
    else:
        if lr_stat < 0.0:
            lr_stat = 0.0
        p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

    pass_test = (
        bool(p_value >= _DEFAULT_SIGNIFICANCE) if np.isfinite(p_value) else None
    )

    if pass_test is None:
        interpretation = "Kupiec POF test inconclusive (degenerate sample)."
    elif pass_test:
        interpretation = (
            f"Observed breach rate {p_hat:.4f} is statistically consistent with "
            f"the expected rate {p:.4f} (p={p_value:.4f}); the model is not "
            "rejected by the Kupiec POF test."
        )
    else:
        direction = "too many" if p_hat > p else "too few"
        interpretation = (
            f"Observed breach rate {p_hat:.4f} differs significantly from the "
            f"expected rate {p:.4f} ({direction} breaches, p={p_value:.4f}); "
            "the Kupiec POF test rejects the model."
        )

    return {
        "test_name": "Kupiec POF",
        "observations": n,
        "breaches": x,
        "expected_breach_probability": p,
        "observed_breach_probability": p_hat,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "pass_test": pass_test,
        "interpretation": interpretation,
    }


def christoffersen_independence_test(
    breaches: pd.Series | np.ndarray,
) -> dict:
    """Christoffersen Independence test.

    Tests whether VaR breaches are independently distributed in time.
    Clustered breaches indicate the model fails to capture volatility regimes.

    Parameters
    ----------
    breaches : pd.Series or np.ndarray
        Boolean or 0/1 integer array.

    Returns
    -------
    dict
        Keys: ``test_name``, ``n00``, ``n01``, ``n10``, ``n11``, ``pi01``,
        ``pi11``, ``pi``, ``lr_statistic``, ``p_value``, ``pass_test``,
        ``interpretation``.
    """
    arr = _to_breach_int_array(breaches)
    n = int(arr.size)

    base_result = {
        "test_name": "Christoffersen Independence",
        "n00": 0,
        "n01": 0,
        "n10": 0,
        "n11": 0,
        "pi01": 0.0,
        "pi11": 0.0,
        "pi": 0.0,
        "lr_statistic": float("nan"),
        "p_value": float("nan"),
        "pass_test": None,
        "interpretation": "Insufficient data for Christoffersen Independence test.",
    }

    if n < 2:
        return base_result

    prev = arr[:-1]
    curr = arr[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    total = n00 + n01 + n10 + n11
    breach_count = int(arr.sum())
    if breach_count == 0 or breach_count == n:
        denominator_0 = n00 + n01
        denominator_1 = n10 + n11
        pi01 = n01 / denominator_0 if denominator_0 > 0 else float("nan")
        pi11 = n11 / denominator_1 if denominator_1 > 0 else float("nan")
        missing_state = "breach" if breach_count == 0 else "non-breach"
        return {
            "test_name": "Christoffersen Independence",
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "pi01": float(pi01),
            "pi11": float(pi11),
            "pi": breach_count / n,
            "lr_statistic": float("nan"),
            "p_value": float("nan"),
            "pass_test": None,
            "interpretation": (
                "Christoffersen Independence test inconclusive: the hit "
                f"sequence contains no {missing_state} observations, so both "
                "first-order Markov transition probabilities cannot be "
                "identified."
            ),
        }

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / total if total > 0 else 0.0

    ll_r = _xlogx(n00 + n10, 1.0 - pi) + _xlogx(n01 + n11, pi)
    ll_u = (
        _xlogx(n00, 1.0 - pi01)
        + _xlogx(n01, pi01)
        + _xlogx(n10, 1.0 - pi11)
        + _xlogx(n11, pi11)
    )
    lr_stat = float(-2.0 * (ll_r - ll_u))
    if not np.isfinite(lr_stat):
        lr_stat = float("nan")
        p_value = float("nan")
    else:
        if lr_stat < 0.0:
            lr_stat = 0.0
        p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

    pass_test = (
        bool(p_value >= _DEFAULT_SIGNIFICANCE) if np.isfinite(p_value) else None
    )

    if pass_test is None:
        interpretation = (
            "Christoffersen Independence test inconclusive (degenerate sample)."
        )
    elif pass_test:
        interpretation = (
            f"Breaches appear independently distributed (p={p_value:.4f}); "
            "the independence assumption is not rejected."
        )
    else:
        interpretation = (
            f"Breaches appear clustered in time (p={p_value:.4f}); the "
            "independence assumption is rejected — model likely misses "
            "volatility regimes."
        )

    return {
        "test_name": "Christoffersen Independence",
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "pi01": float(pi01),
        "pi11": float(pi11),
        "pi": float(pi),
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "pass_test": pass_test,
        "interpretation": interpretation,
    }


def christoffersen_cc_test(
    breaches: pd.Series | np.ndarray,
    confidence_level: float,
) -> dict:
    """Christoffersen Conditional Coverage (CC) test.

    Combines the Kupiec POF test and the Independence test::

        LR_cc = LR_pof + LR_ind   ~  chi2(df=2)

    Parameters
    ----------
    breaches : pd.Series or np.ndarray
    confidence_level : float

    Returns
    -------
    dict
        Keys: ``test_name``, ``lr_pof``, ``lr_ind``, ``lr_cc``, ``p_value``,
        ``pass_test``, ``interpretation``.
    """
    pof = kupiec_pof_test(breaches, confidence_level)
    ind = christoffersen_independence_test(breaches)

    lr_pof = pof["lr_statistic"]
    lr_ind = ind["lr_statistic"]

    if pof["pass_test"] is None or ind["pass_test"] is None:
        return {
            "test_name": "Christoffersen CC",
            "lr_pof": lr_pof,
            "lr_ind": lr_ind,
            "lr_cc": float("nan"),
            "p_value": float("nan"),
            "pass_test": None,
            "interpretation": (
                "Conditional Coverage test inconclusive — its unconditional "
                "coverage or independence component is not identifiable for "
                "this hit sequence."
            ),
        }

    if not (np.isfinite(lr_pof) and np.isfinite(lr_ind)):
        return {
            "test_name": "Christoffersen CC",
            "lr_pof": lr_pof,
            "lr_ind": lr_ind,
            "lr_cc": float("nan"),
            "p_value": float("nan"),
            "pass_test": None,
            "interpretation": (
                "Conditional Coverage test inconclusive — at least one "
                "sub-test produced a non-finite statistic."
            ),
        }

    lr_cc = float(lr_pof + lr_ind)
    if lr_cc < 0.0:
        lr_cc = 0.0
    p_value = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
    pass_test = bool(p_value >= _DEFAULT_SIGNIFICANCE)

    if pass_test:
        interpretation = (
            f"Conditional coverage is acceptable (p={p_value:.4f}); the model "
            "passes the joint Kupiec + Independence test."
        )
    else:
        interpretation = (
            f"Conditional coverage is rejected (p={p_value:.4f}); breach "
            "frequency or clustering is inconsistent with the stated "
            "confidence level."
        )

    return {
        "test_name": "Christoffersen CC",
        "lr_pof": float(lr_pof),
        "lr_ind": float(lr_ind),
        "lr_cc": lr_cc,
        "p_value": p_value,
        "pass_test": pass_test,
        "interpretation": interpretation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Traffic light system
# ─────────────────────────────────────────────────────────────────────────────


def assign_traffic_light_status(
    actual_breach_rate: float,
    expected_breach_rate: float,
    actual_breaches: int,
    n_observations: int,
    mode: str = "auto",
) -> str:
    """Assign a model validation status using a two-tier traffic light system.

    Modes
    -----
    ``basel3``
        Basel III Market Risk framework (BCBS 2019), designed for the
        standard 250-trading-day backtest window. Uses absolute breach counts:

        * Green  : ``actual_breaches <= 4``
        * Yellow : ``5 <= actual_breaches <= 9``
        * Red    : ``actual_breaches >= 10``

    ``rate_based``
        Generalised ratio-based thresholds for non-standard windows, where
        ``breach_ratio = actual_breach_rate / expected_breach_rate``:

        * Green  : ``0.75 <= breach_ratio <= 1.25``
        * Yellow : ``0.50 <= breach_ratio <= 1.75`` and not Green
        * Red    : otherwise

    ``auto``
        Use ``basel3`` when ``240 <= n_observations <= 260``, else
        ``rate_based``.

    Parameters
    ----------
    actual_breach_rate : float
    expected_breach_rate : float
    actual_breaches : int
    n_observations : int
    mode : str, optional

    Returns
    -------
    str
        One of ``"Green"``, ``"Yellow"``, ``"Red"``.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported values.
    """
    if mode not in _SUPPORTED_TRAFFIC_LIGHT_MODES:
        raise ValueError(
            f"Unsupported mode '{mode}'. "
            f"Use one of: {list(_SUPPORTED_TRAFFIC_LIGHT_MODES)}."
        )

    effective_mode = mode
    if mode == "auto":
        effective_mode = (
            "basel3" if 240 <= n_observations <= 260 else "rate_based"
        )

    if effective_mode == "basel3":
        if actual_breaches <= 4:
            return "Green"
        if actual_breaches <= 9:
            return "Yellow"
        return "Red"

    # rate_based
    if expected_breach_rate <= 0.0 or not np.isfinite(expected_breach_rate):
        return "Red"
    ratio = actual_breach_rate / expected_breach_rate
    if 0.75 <= ratio <= 1.25:
        return "Green"
    if 0.50 <= ratio <= 1.75:
        return "Yellow"
    return "Red"


def interpret_traffic_light_status(status: str) -> str:
    """Return a human-readable interpretation of the traffic light status.

    Raises
    ------
    ValueError
        If ``status`` is not one of ``"Green"``, ``"Yellow"``, ``"Red"``.
    """
    mapping = {
        "Green": (
            "Breach frequency is broadly aligned with the selected confidence "
            "level."
        ),
        "Yellow": (
            "Breach frequency moderately deviates from expectation; model "
            "should be reviewed."
        ),
        "Red": (
            "Breach frequency materially deviates from expectation; model may "
            "be unreliable."
        ),
    }
    if status not in mapping:
        raise ValueError(
            f"Unknown traffic light status '{status}'. "
            "Use one of: ['Green', 'Yellow', 'Red']."
        )
    return mapping[status]


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipelines
# ─────────────────────────────────────────────────────────────────────────────


def backtest_var_model(
    returns: pd.Series,
    method: str,
    confidence_level: float = 0.95,
    window: int = 252,
    traffic_light_mode: str = "auto",
    horizon_days: int = 1,
    return_method: str = "simple",
    backtest_mode: str = "overlapping",
) -> tuple[pd.DataFrame, dict]:
    """Run the full backtesting pipeline for a single VaR model.

    Steps
    -----
    1. :func:`rolling_var_forecast` (horizon-aware)
    2. :func:`calculate_var_breaches`
    3. :func:`kupiec_pof_test`
    4. :func:`christoffersen_independence_test`
    5. :func:`christoffersen_cc_test`
    6. :func:`assign_traffic_light_status`
    7. :func:`interpret_traffic_light_status`

    Returns
    -------
    tuple
        ``(forecast_df, result_dict)`` where ``result_dict`` contains the keys
        documented in the Phase 3 / Phase 4 specifications (including
        ``horizon_days``).
    """
    forecast_df = rolling_var_forecast(
        returns=returns,
        method=method,
        confidence_level=confidence_level,
        window=window,
        horizon_days=horizon_days,
        return_method=return_method,
        backtest_mode=backtest_mode,
    )
    step_size = 1 if backtest_mode == "overlapping" else int(horizon_days)
    breach_summary = calculate_var_breaches(forecast_df, confidence_level)
    breaches = forecast_df["breach"].to_numpy(dtype=bool)

    pof = kupiec_pof_test(breaches, confidence_level)
    ind = christoffersen_independence_test(breaches)
    cc = christoffersen_cc_test(breaches, confidence_level)

    n_obs = breach_summary["observations"]
    if traffic_light_mode == "auto":
        effective_mode = "basel3" if 240 <= n_obs <= 260 else "rate_based"
    else:
        effective_mode = traffic_light_mode

    traffic_light = assign_traffic_light_status(
        actual_breach_rate=breach_summary["actual_breach_rate"],
        expected_breach_rate=breach_summary["expected_breach_rate"],
        actual_breaches=breach_summary["actual_breaches"],
        n_observations=n_obs,
        mode=traffic_light_mode,
    )
    interpretation = interpret_traffic_light_status(traffic_light)

    result = {
        "method": method,
        "confidence_level": float(confidence_level),
        "window": int(window),
        "horizon_days": int(horizon_days),
        "return_method": str(return_method),
        "backtest_mode": str(backtest_mode),
        "step_size": int(step_size),
        "observations": int(n_obs),
        "actual_breaches": int(breach_summary["actual_breaches"]),
        "expected_breaches": float(breach_summary["expected_breaches"]),
        "expected_breach_rate": float(breach_summary["expected_breach_rate"]),
        "actual_breach_rate": float(breach_summary["actual_breach_rate"]),
        "breach_ratio": float(breach_summary["breach_ratio"]),
        "kupiec_lr_statistic": pof["lr_statistic"],
        "kupiec_p_value": pof["p_value"],
        "kupiec_pass": pof["pass_test"],
        "christoffersen_lr_statistic": ind["lr_statistic"],
        "christoffersen_p_value": ind["p_value"],
        "christoffersen_pass": ind["pass_test"],
        "christoffersen_interpretation": ind["interpretation"],
        "cc_lr_statistic": cc["lr_cc"],
        "cc_p_value": cc["p_value"],
        "cc_pass": cc["pass_test"],
        "cc_interpretation": cc["interpretation"],
        "traffic_light": traffic_light,
        "traffic_light_mode_used": effective_mode,
        "interpretation": interpretation,
    }
    return forecast_df, result


def compare_var_models_backtest(
    returns: pd.Series,
    methods: list[str],
    confidence_level: float = 0.95,
    window: int = 252,
    traffic_light_mode: str = "auto",
    horizon_days: int = 1,
    return_method: str = "simple",
    backtest_mode: str = "overlapping",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Backtest multiple VaR methods (horizon-aware) and collect the results.

    A failure in one method does not abort the comparison; the error is
    recorded in the comparison row's ``error`` column and the next method
    proceeds.

    Returns
    -------
    tuple
        ``(forecasts_by_method, comparison_df)``.
    """
    forecasts_by_method: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []

    for method in methods:
        try:
            forecast_df, result = backtest_var_model(
                returns=returns,
                method=method,
                confidence_level=confidence_level,
                window=window,
                traffic_light_mode=traffic_light_mode,
                horizon_days=horizon_days,
                return_method=return_method,
                backtest_mode=backtest_mode,
            )
            forecasts_by_method[method] = forecast_df
            row = {
                "method": method,
                "confidence_level": result["confidence_level"],
                "window": result["window"],
                "horizon_days": result["horizon_days"],
                "backtest_mode": result["backtest_mode"],
                "step_size": result["step_size"],
                "observations": result["observations"],
                "actual_breaches": result["actual_breaches"],
                "expected_breaches": result["expected_breaches"],
                "expected_breach_rate": result["expected_breach_rate"],
                "actual_breach_rate": result["actual_breach_rate"],
                "breach_ratio": result["breach_ratio"],
                "kupiec_p_value": result["kupiec_p_value"],
                "kupiec_pass": result["kupiec_pass"],
                "christoffersen_p_value": result["christoffersen_p_value"],
                "christoffersen_pass": result["christoffersen_pass"],
                "christoffersen_interpretation": result[
                    "christoffersen_interpretation"
                ],
                "cc_p_value": result["cc_p_value"],
                "cc_pass": result["cc_pass"],
                "cc_interpretation": result["cc_interpretation"],
                "traffic_light": result["traffic_light"],
                "traffic_light_mode_used": result["traffic_light_mode_used"],
                "interpretation": result["interpretation"],
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "method": method,
                "confidence_level": float(confidence_level),
                "window": int(window),
                "horizon_days": int(horizon_days),
                "backtest_mode": str(backtest_mode),
                "step_size": 1 if backtest_mode == "overlapping" else int(horizon_days),
                "observations": 0,
                "actual_breaches": 0,
                "expected_breaches": float("nan"),
                "expected_breach_rate": float(1.0 - confidence_level),
                "actual_breach_rate": float("nan"),
                "breach_ratio": float("nan"),
                "kupiec_p_value": float("nan"),
                "kupiec_pass": None,
                "christoffersen_p_value": float("nan"),
                "christoffersen_pass": None,
                "christoffersen_interpretation": "Backtest failed.",
                "cc_p_value": float("nan"),
                "cc_pass": None,
                "cc_interpretation": "Backtest failed.",
                "traffic_light": "N/A",
                "interpretation": f"Backtest failed: {exc}",
                "error": str(exc),
            }
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    return forecasts_by_method, comparison_df


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def _format_pass(value: object) -> str:
    if value is True:
        return "✓ Pass"
    if value is False:
        return "✗ Fail"
    return "— Inconclusive"


def _format_float(value: object, decimals: int) -> str:
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(f):
        return "N/A"
    return f"{f:.{decimals}f}"


def create_backtesting_report_table(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """Format a multi-model comparison into a clean reporting table.

    Returns
    -------
    pd.DataFrame
        Columns: ``Method``, ``Observations``, ``Actual Breaches``,
        ``Expected Breaches``, ``Actual Breach Rate``, ``Expected Breach Rate``,
        ``Kupiec p-value``, ``Kupiec Pass``, ``Christoffersen p-value``,
        ``Christoffersen Pass``, ``CC p-value``, ``CC Pass``,
        ``Traffic Light``, ``Interpretation``.
    """
    base_columns = [
        "Method",
        "Horizon (days)",
        "Mode",
        "Observations",
        "Actual Breaches",
        "Expected Breaches",
        "Actual Breach Rate",
        "Expected Breach Rate",
        "Kupiec p-value",
        "Kupiec Pass",
        "Christoffersen p-value",
        "Christoffersen Pass",
        "CC p-value",
        "CC Pass",
        "Traffic Light",
        "Interpretation",
    ]
    if comparison_df.empty:
        return pd.DataFrame(columns=base_columns)

    rows = []
    for _, row in comparison_df.iterrows():
        method_label = str(row.get("method", "?")).replace("_", " ").title()
        horizon = row.get("horizon_days", 1)
        try:
            horizon_label = int(horizon)
        except (TypeError, ValueError):
            horizon_label = "N/A"
        mode_label = str(row.get("backtest_mode", "overlapping")).replace(
            "_", " "
        ).title()
        error = row.get("error")
        has_error = error is not None and not (
            isinstance(error, float) and np.isnan(error)
        )

        if has_error:
            rows.append(
                {
                    "Method": method_label,
                    "Horizon (days)": horizon_label,
                    "Mode": mode_label,
                    "Observations": "N/A",
                    "Actual Breaches": "N/A",
                    "Expected Breaches": "N/A",
                    "Actual Breach Rate": "N/A",
                    "Expected Breach Rate": "N/A",
                    "Kupiec p-value": "N/A",
                    "Kupiec Pass": "N/A",
                    "Christoffersen p-value": "N/A",
                    "Christoffersen Pass": "N/A",
                    "CC p-value": "N/A",
                    "CC Pass": "N/A",
                    "Traffic Light": "N/A",
                    "Interpretation": f"Error: {error}",
                }
            )
            continue

        rows.append(
            {
                "Method": method_label,
                "Horizon (days)": horizon_label,
                "Mode": mode_label,
                "Observations": int(row["observations"]),
                "Actual Breaches": int(row["actual_breaches"]),
                "Expected Breaches": _format_float(row["expected_breaches"], 1),
                "Actual Breach Rate": _format_float(row["actual_breach_rate"], 4),
                "Expected Breach Rate": _format_float(
                    row["expected_breach_rate"], 4
                ),
                "Kupiec p-value": _format_float(row["kupiec_p_value"], 4),
                "Kupiec Pass": _format_pass(row["kupiec_pass"]),
                "Christoffersen p-value": _format_float(
                    row["christoffersen_p_value"], 4
                ),
                "Christoffersen Pass": _format_pass(row["christoffersen_pass"]),
                "CC p-value": _format_float(row["cc_p_value"], 4),
                "CC Pass": _format_pass(row["cc_pass"]),
                "Traffic Light": str(row["traffic_light"]),
                "Interpretation": str(row["interpretation"]),
            }
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Breach-rate evolution, worst losses, and sub-period summaries
# ─────────────────────────────────────────────────────────────────────────────


def calculate_rolling_breach_rate(
    backtest_df: pd.DataFrame,
    window: int = 100,
) -> pd.Series:
    """Rolling average breach rate over time.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of :func:`rolling_var_forecast` (must contain ``breach``).
    window : int
        Rolling window length. Must be ``>= 1``.

    Returns
    -------
    pd.Series
        Rolling mean of the boolean breach indicator, indexed like the
        input. ``min_periods=1`` so the series is populated from the start.
    """
    if "breach" not in backtest_df.columns:
        raise ValueError("backtest_df must contain a 'breach' column.")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    breaches = backtest_df["breach"].astype(float)
    rolling = breaches.rolling(window=int(window), min_periods=1).mean()
    rolling.name = "rolling_breach_rate"
    return rolling


def get_worst_realized_losses(
    backtest_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Return the ``n`` worst realised horizon returns in the backtest.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of :func:`rolling_var_forecast` (needs ``actual_return``,
        ``var_forecast``, ``breach``).
    n : int
        Number of rows to return. Must be ``>= 1``.

    Returns
    -------
    pd.DataFrame
        Columns ``Date``, ``Actual Return``, ``VaR Forecast``, ``Breach``,
        ``Loss`` (``Loss = -actual_return``), sorted by the most negative
        realised return first.
    """
    required = {"actual_return", "var_forecast", "breach"}
    missing = required - set(backtest_df.columns)
    if missing:
        raise ValueError(f"backtest_df is missing required columns: {missing}.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    ordered = backtest_df.sort_values("actual_return", ascending=True).head(int(n))
    return pd.DataFrame(
        {
            "Date": list(ordered.index),
            "Actual Return": ordered["actual_return"].to_numpy(dtype=float),
            "VaR Forecast": ordered["var_forecast"].to_numpy(dtype=float),
            "Breach": ordered["breach"].astype(bool).to_numpy(),
            "Loss": -ordered["actual_return"].to_numpy(dtype=float),
        }
    )


def summarize_backtest_by_period(
    backtest_df: pd.DataFrame,
    confidence_level: float,
    freq: str = "Y",
) -> pd.DataFrame:
    """Summarise breach behaviour by calendar period (e.g. by year).

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of :func:`rolling_var_forecast` with a DatetimeIndex.
    confidence_level : float
        Used to derive the expected breach rate ``1 - confidence_level``.
    freq : str
        Period frequency passed to ``DatetimeIndex.to_period`` (``"Y"`` for
        calendar year, ``"Q"`` for quarter, ``"M"`` for month).

    Returns
    -------
    pd.DataFrame
        Columns ``Period``, ``Observations``, ``Actual Breaches``,
        ``Expected Breaches``, ``Actual Breach Rate``, ``Expected Breach Rate``.
    """
    if "breach" not in backtest_df.columns:
        raise ValueError("backtest_df must contain a 'breach' column.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}."
        )
    if not isinstance(backtest_df.index, pd.DatetimeIndex):
        raise ValueError("backtest_df must have a DatetimeIndex.")

    expected_rate = float(1.0 - confidence_level)
    periods = backtest_df.index.to_period(freq).astype(str)
    breaches = backtest_df["breach"].astype(bool)

    rows: list[dict] = []
    for period, sub in breaches.groupby(periods, sort=True):
        obs = int(len(sub))
        actual_breaches = int(sub.sum())
        rows.append(
            {
                "Period": str(period),
                "Observations": obs,
                "Actual Breaches": actual_breaches,
                "Expected Breaches": obs * expected_rate,
                "Actual Breach Rate": (
                    actual_breaches / obs if obs > 0 else float("nan")
                ),
                "Expected Breach Rate": expected_rate,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Period",
            "Observations",
            "Actual Breaches",
            "Expected Breaches",
            "Actual Breach Rate",
            "Expected Breach Rate",
        ],
    )
