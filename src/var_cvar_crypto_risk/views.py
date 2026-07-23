"""Manual expected-return views layer.

A deliberately small, dependency-light seam for injecting user opinions about
expected returns into the optimization pipeline. It is the clean input layer
that a future Black-Litterman or Meucci Entropy-Pooling implementation can plug
into — but it does **not** implement either of those. Today it only blends a
base expected-return vector with the user's point views.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class AssetReturnView:
    """A single user view on an asset's per-horizon expected return.

    Parameters
    ----------
    asset : str
        Asset symbol; must match a column in the base expected-return vector.
    expected_return : float
        The user's expected return for this asset (per horizon).
    confidence : float, optional
        Reserved for future Black-Litterman / Entropy-Pooling weighting. Not
        used by :func:`apply_manual_expected_return_views` yet.
    """

    asset: str
    expected_return: float
    confidence: float | None = None


def apply_manual_expected_return_views(
    base_expected_returns: pd.Series,
    views: list[AssetReturnView],
    blend_weight: float = 1.0,
) -> pd.Series:
    """Blend a base expected-return vector with manual views.

    For every asset that has a view::

        adjusted = blend_weight * view + (1 - blend_weight) * base

    Assets without a view are left unchanged. ``blend_weight = 1.0`` fully
    replaces the base return with the user's view; ``0.5`` is an even blend.

    Parameters
    ----------
    base_expected_returns : pandas.Series
        Indexed by asset name.
    views : list[AssetReturnView]
        User views. A view for an asset not present in
        ``base_expected_returns`` raises ``ValueError``.
    blend_weight : float, in ``[0, 1]``

    Returns
    -------
    pandas.Series
        A new expected-return vector (the input is not mutated).
    """
    if not isinstance(base_expected_returns, pd.Series):
        raise ValueError("base_expected_returns must be a pd.Series.")
    if not (0.0 <= blend_weight <= 1.0):
        raise ValueError(f"blend_weight must be in [0, 1], got {blend_weight}.")

    adjusted = base_expected_returns.astype(float).copy()
    for view in views:
        if view.asset not in adjusted.index:
            raise ValueError(
                f"View for unknown asset '{view.asset}'. "
                f"Known assets: {list(adjusted.index)}."
            )
        base_value = float(adjusted.loc[view.asset])
        adjusted.loc[view.asset] = (
            blend_weight * float(view.expected_return)
            + (1.0 - blend_weight) * base_value
        )
    adjusted.name = "expected_return"
    return adjusted
