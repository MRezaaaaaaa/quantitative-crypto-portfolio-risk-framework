"""Return calculation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple (arithmetic) returns ``r_t = P_t / P_{t-1} - 1``.

    Drops the first row (NaN from the shift). Returns a clean DataFrame.
    """
    returns = prices.pct_change()
    returns = returns.iloc[1:]
    return returns


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns ``r_t = ln(P_t / P_{t-1})``.

    Drops the first row (NaN from the shift). Returns a clean DataFrame.
    """
    log_prices = np.log(prices)
    returns = log_prices.diff()
    returns = returns.iloc[1:]
    return returns


def calculate_returns(
    prices: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """Dispatcher for return calculation.

    Parameters
    ----------
    prices : pandas.DataFrame
    method : {"simple", "log"}

    Raises
    ------
    ValueError
        If ``method`` is not ``"simple"`` or ``"log"``.
    """
    if method == "simple":
        return calculate_simple_returns(prices)
    if method == "log":
        return calculate_log_returns(prices)
    raise ValueError(
        f"Unknown return method '{method}'. Use 'simple' or 'log'."
    )


def calculate_cumulative_returns(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Calculate cumulative returns from a returns series.

    Uses the simple-returns convention: ``cumulative = (1 + r).cumprod() - 1``.
    """
    return (1.0 + returns).cumprod() - 1.0


def calculate_horizon_returns(
    returns: pd.Series,
    horizon_days: int,
    method: str = "simple",
    overlapping: bool = True,
) -> pd.Series:
    """Aggregate a daily return series into ``horizon_days``-day returns.

    Parameters
    ----------
    returns : pandas.Series
        Daily returns (NaNs are dropped first).
    horizon_days : int
        Aggregation horizon. ``1`` returns the cleaned input unchanged.
    method : {"simple", "log"}
        ``simple`` ⇒ ``prod(1 + r) - 1`` over the horizon; ``log`` ⇒ sum of
        log returns over the horizon.
    overlapping : bool
        ``True`` ⇒ rolling (overlapping) h-day returns labelled at the window
        end. ``False`` ⇒ contiguous non-overlapping blocks, labelled at the
        block end.

    Returns
    -------
    pandas.Series
        The horizon returns. For ``overlapping=True`` length is
        ``len(clean) - horizon_days + 1``; for ``overlapping=False`` length is
        ``len(clean) // horizon_days``.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pd.Series.")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}.")
    if method not in ("simple", "log"):
        raise ValueError(f"Unknown method '{method}'. Use 'simple' or 'log'.")

    clean = returns.dropna()
    h = int(horizon_days)
    if h == 1:
        return clean.copy()
    if len(clean) < h:
        raise ValueError(
            f"Need at least horizon_days={h} observations, got {len(clean)}."
        )

    if overlapping:
        if method == "simple":
            agg = (
                (1.0 + clean)
                .rolling(window=h)
                .apply(lambda x: float(np.prod(x) - 1.0), raw=True)
            )
        else:
            agg = clean.rolling(window=h).sum()
        agg = agg.dropna()
        agg.name = f"horizon_{h}d_return"
        return agg

    values = clean.to_numpy(dtype=float)
    index = clean.index
    n_blocks = len(values) // h
    out_values = np.empty(n_blocks, dtype=float)
    out_index = []
    for b in range(n_blocks):
        block = values[b * h : (b + 1) * h]
        if method == "simple":
            out_values[b] = float(np.prod(1.0 + block) - 1.0)
        else:
            out_values[b] = float(np.sum(block))
        out_index.append(index[(b + 1) * h - 1])
    return pd.Series(out_values, index=out_index, name=f"horizon_{h}d_return")


def annualize_return(
    returns: pd.Series,
    periods_per_year: int = 365,
) -> float:
    """Annualize a daily return series.

    Formula: ``(1 + mean_daily_return) ** periods_per_year - 1``.
    """
    mean_daily = float(returns.mean())
    return (1.0 + mean_daily) ** periods_per_year - 1.0


def annualize_volatility(
    returns: pd.Series,
    periods_per_year: int = 365,
) -> float:
    """Annualize daily volatility.

    Formula: ``daily_std * sqrt(periods_per_year)``.
    """
    daily_std = float(returns.std(ddof=1))
    return daily_std * np.sqrt(periods_per_year)
