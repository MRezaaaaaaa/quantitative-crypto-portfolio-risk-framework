"""Asset-return correlation and diversification analytics (Phase 5.5).

Pure analytics: no plotting, no Streamlit, no I/O. The plotting counterparts
live in :mod:`var_cvar_crypto_risk.plotting`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_SUPPORTED_CORR_METHODS: tuple[str, ...] = ("pearson", "spearman")


def calculate_correlation_matrix(
    asset_returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Return the asset-return correlation matrix.

    Parameters
    ----------
    asset_returns : pandas.DataFrame
        Rows = dates, columns = assets.
    method : {"pearson", "spearman"}
        Correlation estimator.

    Returns
    -------
    pandas.DataFrame
        Square correlation matrix indexed and columned by asset name.
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise ValueError("asset_returns must be a pandas DataFrame.")
    if asset_returns.shape[1] < 1:
        raise ValueError("asset_returns must contain at least one asset.")
    if method not in _SUPPORTED_CORR_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Use one of: {list(_SUPPORTED_CORR_METHODS)}."
        )
    return asset_returns.corr(method=method)


def calculate_rolling_average_correlation(
    asset_returns: pd.DataFrame,
    window: int = 90,
) -> pd.Series:
    """Average pairwise (off-diagonal) correlation through time.

    For each rolling window of length ``window`` the full pairwise
    correlation matrix is computed and the mean of its off-diagonal entries
    is recorded — a single scalar measuring how correlated the basket is at
    that point in time (diversification decay shows up as a rising line).

    Parameters
    ----------
    asset_returns : pandas.DataFrame
        Rows = dates, columns = assets (needs >= 2 assets).
    window : int
        Rolling window length. Must be ``>= 2``.

    Returns
    -------
    pandas.Series
        Indexed by the window-end date. Leading windows without enough data
        are omitted.
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise ValueError("asset_returns must be a pandas DataFrame.")
    n_assets = asset_returns.shape[1]
    if n_assets < 2:
        raise ValueError(
            "Rolling average correlation needs at least 2 assets, "
            f"got {n_assets}."
        )
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}.")

    clean = asset_returns.dropna()
    if len(clean) < window:
        raise ValueError(
            f"Need at least window={window} observations, got {len(clean)}."
        )

    denom = n_assets * (n_assets - 1)
    values = clean.to_numpy(dtype=float)
    index = clean.index
    out_values: list[float] = []
    out_index: list = []
    for end in range(window, len(clean) + 1):
        block = values[end - window : end]
        corr = np.corrcoef(block, rowvar=False)
        off_diag_mean = (np.nansum(corr) - np.trace(corr)) / denom
        out_values.append(float(off_diag_mean))
        out_index.append(index[end - 1])

    return pd.Series(
        out_values, index=out_index, name=f"avg_corr_{window}d"
    )


def calculate_weighted_average_correlation(
    correlation_matrix: pd.DataFrame,
    weights: pd.Series,
) -> float:
    """Portfolio-weighted average pairwise correlation.

    ``sum_{i != j} w_i w_j rho_ij / sum_{i != j} w_i w_j`` — the average
    off-diagonal correlation where each pair is weighted by the product of
    the portfolio weights, so the number reflects the correlation the
    *portfolio* actually experiences rather than an equal-weighted average.

    Parameters
    ----------
    correlation_matrix : pandas.DataFrame
        Square correlation matrix (e.g. from
        :func:`calculate_correlation_matrix`).
    weights : pandas.Series
        Portfolio weights; assets missing from the matrix raise
        ``ValueError``, extra matrix assets are ignored.
    """
    if not isinstance(correlation_matrix, pd.DataFrame):
        raise ValueError("correlation_matrix must be a pandas DataFrame.")
    if not isinstance(weights, pd.Series):
        raise ValueError("weights must be a pandas Series.")
    assets = [a for a in weights.index if weights.get(a, 0.0) != 0.0]
    missing = [a for a in assets if a not in correlation_matrix.index]
    if missing:
        raise ValueError(f"Assets missing from correlation matrix: {missing}")
    if len(assets) < 2:
        raise ValueError("Need at least 2 non-zero-weight assets.")

    corr = correlation_matrix.loc[assets, assets].to_numpy(dtype=float)
    w = weights.reindex(assets).to_numpy(dtype=float)
    outer = np.outer(w, w)
    off_diag = ~np.eye(len(assets), dtype=bool)
    denom = float(outer[off_diag].sum())
    if denom == 0.0:
        raise ValueError("Weight products sum to zero off-diagonal.")
    return float((outer[off_diag] * corr[off_diag]).sum() / denom)


def calculate_stress_vs_normal_correlation(
    asset_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    stress_quantile: float = 0.10,
    method: str = "pearson",
) -> dict:
    """Average pairwise correlation on stress days vs normal days.

    Stress days are the worst ``stress_quantile`` of portfolio-return
    days. Crypto correlations typically rise in drawdowns, so the stress
    correlation is usually higher — quantifying the loss of
    diversification exactly when it is needed most.

    Returns
    -------
    dict
        ``stress_avg_corr``, ``normal_avg_corr``, ``n_stress_days``,
        ``n_normal_days``, ``stress_threshold`` (the portfolio-return
        cutoff that defines a stress day).
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise ValueError("asset_returns must be a pandas DataFrame.")
    if asset_returns.shape[1] < 2:
        raise ValueError("Need at least 2 assets.")
    if not isinstance(portfolio_returns, pd.Series):
        raise ValueError("portfolio_returns must be a pandas Series.")
    if not (0.0 < stress_quantile < 0.5):
        raise ValueError(
            f"stress_quantile must be in (0, 0.5), got {stress_quantile}."
        )

    joined = asset_returns.join(
        portfolio_returns.rename("__portfolio__"), how="inner"
    ).dropna()
    if len(joined) < 20:
        raise ValueError(
            f"Need at least 20 overlapping observations, got {len(joined)}."
        )
    pf = joined["__portfolio__"]
    rets = joined.drop(columns="__portfolio__")

    threshold = float(pf.quantile(stress_quantile))
    stress_mask = pf <= threshold

    def _avg_pairwise(frame: pd.DataFrame) -> float:
        corr = frame.corr(method=method).to_numpy(dtype=float)
        n = corr.shape[0]
        off_diag = ~np.eye(n, dtype=bool)
        return float(np.nanmean(corr[off_diag]))

    return {
        "stress_avg_corr": _avg_pairwise(rets.loc[stress_mask]),
        "normal_avg_corr": _avg_pairwise(rets.loc[~stress_mask]),
        "n_stress_days": int(stress_mask.sum()),
        "n_normal_days": int((~stress_mask).sum()),
        "stress_threshold": threshold,
    }
