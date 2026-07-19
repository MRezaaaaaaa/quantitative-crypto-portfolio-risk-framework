"""Conditional Value-at-Risk (CVaR / Expected Shortfall) models.

All CVaR values are returned as **positive loss numbers**.
For the same data and confidence level, ``CVaR >= VaR`` always holds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats


class CVaRModel(ABC):
    """Abstract base class for all CVaR / Expected Shortfall models.

    Future models (Monte Carlo CVaR, GARCH CVaR) must extend this.
    """

    @abstractmethod
    def compute(
        self,
        returns: pd.Series,
        confidence_level: float,
    ) -> float:
        """Compute 1-day CVaR (Expected Shortfall).

        Returns
        -------
        float
            CVaR as a positive loss number. ``CVaR >= VaR`` for the same
            method and confidence level.
        """

    def validate_confidence(self, confidence_level: float) -> None:
        """Raise ``ValueError`` if ``confidence_level`` is not strictly in (0, 1)."""
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(
                f"confidence_level must be in (0, 1), got {confidence_level}."
            )


class HistoricalCVaR(CVaRModel):
    """Historical (non-parametric) Expected Shortfall.

    Average of all observed returns at or worse than the historical VaR
    threshold.
    """

    def compute(self, returns: pd.Series, confidence_level: float) -> float:
        self.validate_confidence(confidence_level)
        clean = returns.dropna()
        if clean.empty:
            raise ValueError("Cannot compute Historical CVaR on empty return series.")
        alpha = 1.0 - confidence_level
        threshold = float(np.quantile(clean.values, alpha))
        tail = clean[clean <= threshold]
        if tail.empty:
            raise ValueError(
                "No tail observations found for Historical CVaR. "
                "Increase the sample size or lower the confidence level."
            )
        return -float(tail.mean())


class GaussianCVaR(CVaRModel):
    """Parametric Gaussian CVaR (analytical Expected Shortfall).

    Formula::

        CVaR = -(mu - sigma * pdf(z) / (1 - alpha))

    where ``z = norm.ppf(1 - alpha)`` and ``alpha = 1 - confidence_level``.
    """

    def compute(self, returns: pd.Series, confidence_level: float) -> float:
        self.validate_confidence(confidence_level)
        clean = returns.dropna()
        if len(clean) < 2:
            raise ValueError(
                "Gaussian CVaR requires at least 2 return observations."
            )
        mu = float(clean.mean())
        sigma = float(clean.std(ddof=1))
        alpha = 1.0 - confidence_level
        z = stats.norm.ppf(alpha)
        pdf_z = stats.norm.pdf(z)
        cvar_value = -(mu - sigma * pdf_z / alpha)
        return cvar_value


_HISTORICAL = HistoricalCVaR()
_GAUSSIAN = GaussianCVaR()


def historical_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Convenience wrapper for :class:`HistoricalCVaR`."""
    return _HISTORICAL.compute(returns, confidence_level)


def gaussian_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Convenience wrapper for :class:`GaussianCVaR`."""
    return _GAUSSIAN.compute(returns, confidence_level)


_DISPATCH: dict[str, CVaRModel] = {
    "historical": _HISTORICAL,
    "gaussian": _GAUSSIAN,
}


def calculate_cvar(
    returns: pd.Series,
    method: str,
    confidence_level: float = 0.95,
) -> float:
    """Dispatcher.

    Parameters
    ----------
    method : {"historical", "gaussian"}

    Raises
    ------
    ValueError
        For unknown methods.
    """
    model = _DISPATCH.get(method)
    if model is None:
        raise ValueError(
            f"Unknown CVaR method '{method}'. "
            f"Use one of: {sorted(_DISPATCH.keys())}."
        )
    return model.compute(returns, confidence_level)


def return_cvar_to_money_cvar(cvar_return: float, portfolio_value: float) -> float:
    """Convert a return-based CVaR to a monetary CVaR."""
    return cvar_return * portfolio_value
