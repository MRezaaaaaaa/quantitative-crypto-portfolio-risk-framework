"""Value-at-Risk (VaR) models.

VaR is returned as a decimal value in signed loss space. ``0.042`` represents
a 4.2% loss threshold; ``-0.02`` represents a 2% gain threshold.
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


class VaRModel(ABC):
    """Abstract base class for all VaR models.

    Future models (GARCH, Monte Carlo, Filtered Historical) must extend this.
    """

    @abstractmethod
    def compute(
        self,
        returns: pd.Series,
        confidence_level: float,
    ) -> float:
        """Compute VaR for the observation horizon of ``returns``.

        Returns
        -------
        float
            Signed decimal loss value. Positive values are losses and negative
            values are gains at the VaR threshold.
        """

    def validate_confidence(self, confidence_level: float) -> None:
        """Raise ``ValueError`` if ``confidence_level`` is not strictly in (0, 1)."""
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(
                f"confidence_level must be in (0, 1), got {confidence_level}."
            )


class HistoricalVaR(VaRModel):
    """Historical simulation VaR.

    Uses the empirical left-tail percentile of observed returns.
    No distributional assumption.
    """

    def compute(self, returns: pd.Series, confidence_level: float) -> float:
        self.validate_confidence(confidence_level)
        clean = returns.dropna()
        if clean.empty:
            raise ValueError("Cannot compute Historical VaR on empty return series.")
        alpha = 1.0 - confidence_level
        threshold = float(np.quantile(clean.values, alpha))
        return return_threshold_to_loss_value(threshold)


class GaussianVaR(VaRModel):
    """Parametric Gaussian VaR.

    Assumes returns are normally distributed and uses the sample mean
    and sample standard deviation.
    """

    def compute(self, returns: pd.Series, confidence_level: float) -> float:
        self.validate_confidence(confidence_level)
        clean = returns.dropna()
        if len(clean) < 2:
            raise ValueError(
                "Gaussian VaR requires at least 2 return observations."
            )
        mu = float(clean.mean())
        sigma = float(clean.std(ddof=1))
        z = stats.norm.ppf(1.0 - confidence_level)
        return_threshold = mu + sigma * z
        return return_threshold_to_loss_value(return_threshold)


class CornishFisherVaR(VaRModel):
    """Modified VaR using the Cornish-Fisher expansion.

    Adjusts the Gaussian z-score for observed skewness and excess
    kurtosis. More accurate for fat-tailed distributions like crypto.

    Cornish-Fisher z adjustment::

        z_cf = z + (z^2 - 1) * S/6
                 + (z^3 - 3z) * K/24
                 - (2z^3 - 5z) * S^2/36

    where ``S`` is sample skewness, ``K`` is sample excess kurtosis,
    and ``z = norm.ppf(1 - confidence_level)``.
    """

    def compute(self, returns: pd.Series, confidence_level: float) -> float:
        self.validate_confidence(confidence_level)
        clean = returns.dropna()
        if len(clean) < 4:
            raise ValueError(
                "Cornish-Fisher VaR requires at least 4 return observations."
            )
        mu = float(clean.mean())
        sigma = float(clean.std(ddof=1))
        skewness = float(stats.skew(clean.values, bias=False))
        excess_kurt = float(stats.kurtosis(clean.values, fisher=True, bias=False))
        z = stats.norm.ppf(1.0 - confidence_level)
        z_cf = (
            z
            + (z**2 - 1.0) * skewness / 6.0
            + (z**3 - 3.0 * z) * excess_kurt / 24.0
            - (2.0 * z**3 - 5.0 * z) * (skewness**2) / 36.0
        )
        return_threshold = mu + sigma * z_cf
        return return_threshold_to_loss_value(return_threshold)


_HISTORICAL = HistoricalVaR()
_GAUSSIAN = GaussianVaR()
_CORNISH_FISHER = CornishFisherVaR()


def historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Convenience wrapper for :class:`HistoricalVaR`."""
    return _HISTORICAL.compute(returns, confidence_level)


def gaussian_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Convenience wrapper for :class:`GaussianVaR`."""
    return _GAUSSIAN.compute(returns, confidence_level)


def cornish_fisher_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Convenience wrapper for :class:`CornishFisherVaR`."""
    return _CORNISH_FISHER.compute(returns, confidence_level)


_DISPATCH: dict[str, VaRModel] = {
    "historical": _HISTORICAL,
    "gaussian": _GAUSSIAN,
    "cornish_fisher": _CORNISH_FISHER,
}


def calculate_var(
    returns: pd.Series,
    method: str,
    confidence_level: float = 0.95,
) -> float:
    """Dispatcher.

    Parameters
    ----------
    method : {"historical", "gaussian", "cornish_fisher"}

    Raises
    ------
    ValueError
        For unknown methods.
    """
    model = _DISPATCH.get(method)
    if model is None:
        raise ValueError(
            f"Unknown VaR method '{method}'. "
            f"Use one of: {sorted(_DISPATCH.keys())}."
        )
    return model.compute(returns, confidence_level)


def scale_var_to_horizon(var_1day: float, horizon_days: int) -> float:
    """Scale 1-day VaR to multi-day VaR via the square-root-of-time rule.

    Formula: ``VaR_h = VaR_1 * sqrt(horizon_days)``.

    Warning
    -------
    This is an approximation. It assumes i.i.d. returns. For crypto, it
    can underestimate multi-day risk because volatility clustering and
    fat tails violate the i.i.d. assumption.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1.")
    return var_1day * np.sqrt(horizon_days)


def return_var_to_money_var(var_return: float, portfolio_value: float) -> float:
    """Convert signed decimal VaR to monetary VaR in portfolio currency.

    Formula: ``money_var = var_return * portfolio_value``.
    The sign is preserved: positive means loss and negative means gain.
    """
    return loss_value_to_money(var_return, portfolio_value)
