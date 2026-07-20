"""Portfolio construction and aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd


WEIGHT_SUM_TOLERANCE = 1e-6


def get_weights_from_config(config: dict) -> pd.Series:
    """Extract asset weights from ``config["assets"]``.

    Returns
    -------
    pandas.Series
        Index = asset symbol, value = float weight.
    """
    assets: dict = config["assets"]
    weights = {symbol: float(meta.get("weight", 0.0)) for symbol, meta in assets.items()}
    return pd.Series(weights, dtype=float)


def validate_weights(
    weights: pd.Series,
    assets: list[str],
    allow_short_selling: bool = False,
) -> None:
    """Validate portfolio weights.

    Checks
    ------
    1. ``weights.index`` matches ``assets`` (same set of symbols).
    2. If ``allow_short_selling`` is False, all weights must be ``>= 0``.
    3. Weights must sum to ``1.0`` within ``1e-6`` tolerance.

    Raises
    ------
    ValueError
        If any check fails.
    """
    weight_set = set(weights.index)
    asset_set = set(assets)
    if weight_set != asset_set:
        missing = asset_set - weight_set
        extra = weight_set - asset_set
        raise ValueError(
            "Weights index does not match assets list. "
            f"Missing: {sorted(missing)}, Extra: {sorted(extra)}"
        )

    if not allow_short_selling:
        negatives = weights[weights < 0]
        if not negatives.empty:
            raise ValueError(
                "Negative weights are not allowed when "
                f"allow_short_selling=False. Got: {negatives.to_dict()}"
            )

    total = float(weights.sum())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"Weights must sum to 1.0 (within {WEIGHT_SUM_TOLERANCE}). "
            f"Got sum={total:.8f}."
        )


def normalize_weights(weights: pd.Series) -> pd.Series:
    """Normalize weights so they sum to exactly ``1.0``."""
    total = float(weights.sum())
    if total == 0.0:
        raise ValueError("Cannot normalize weights that sum to zero.")
    return weights / total


def calculate_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
    return_method: str = "simple",
) -> pd.Series:
    """Calculate exact daily portfolio returns under the input convention.

    Simple asset returns aggregate arithmetically as
    ``r_p_t = sum_i (w_i * r_i_t)``. For log-return inputs, asset gross
    returns are reconstructed first and the exact portfolio log return is
    ``log(1 + sum_i(w_i * (exp(g_i_t) - 1)))``. A weighted sum of asset log
    returns is not used because it is generally not the portfolio log return.
    """
    if return_method not in {"simple", "log"}:
        raise ValueError(
            "return_method must be 'simple' or 'log', "
            f"got {return_method!r}."
        )
    aligned_weights = weights.reindex(returns.columns)
    if aligned_weights.isna().any():
        missing = aligned_weights[aligned_weights.isna()].index.tolist()
        raise ValueError(
            f"Weights missing for return columns: {missing}"
        )
    values = returns.to_numpy(dtype=float)
    weights_array = aligned_weights.to_numpy(dtype=float)
    if return_method == "simple":
        portfolio = values @ weights_array
    else:
        simple_portfolio = np.expm1(values) @ weights_array
        if np.any(simple_portfolio <= -1.0):
            raise ValueError(
                "Log portfolio return is undefined because the reconstructed "
                "simple portfolio return is <= -1."
            )
        portfolio = np.log1p(simple_portfolio)
    return pd.Series(portfolio, index=returns.index, name="portfolio_return")


def calculate_portfolio_value(
    portfolio_returns: pd.Series,
    initial_capital: float,
    return_method: str = "simple",
) -> pd.Series:
    """Calculate portfolio value from simple or log portfolio returns.

    Simple-return formula: ``V_t = V_0 * (1 + r).cumprod()``.
    Log-return formula: ``V_t = V_0 * exp(g.cumsum())``.
    """
    if return_method not in {"simple", "log"}:
        raise ValueError(
            "return_method must be 'simple' or 'log', "
            f"got {return_method!r}."
        )
    if return_method == "simple":
        growth = (1.0 + portfolio_returns).cumprod()
    else:
        growth = np.exp(portfolio_returns.cumsum())
    value = initial_capital * growth
    value.name = "portfolio_value"
    return value
