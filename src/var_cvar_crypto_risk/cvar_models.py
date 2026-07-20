"""Conditional Value-at-Risk (CVaR / Expected Shortfall) models.

CVaR is returned as a decimal value in signed loss space. Positive values are
losses and negative values are gains. For matching methods and confidence
levels, ``CVaR >= VaR`` in loss space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats

from .risk_conventions import (
    loss_value_to_money,
    return_threshold_to_loss_value,
)


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
        """Compute CVaR for the observation horizon of ``returns``.

        Returns
        -------
        float
            Signed decimal loss value. Positive values are losses and negative
            values are gains. ``CVaR >= VaR`` in loss space for matching
            methods and confidence levels.
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
        return return_threshold_to_loss_value(float(tail.mean()))


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
        tail_return = mu - sigma * pdf_z / alpha
        return return_threshold_to_loss_value(tail_return)


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
    """Convert signed decimal CVaR to monetary CVaR, preserving its sign."""
    return loss_value_to_money(cvar_return, portfolio_value)
