"""Canonical sign and unit conversions for portfolio risk measures.

VaR and CVaR are represented as decimal values in the input return convention
and in **signed loss space**:

* positive values represent losses;
* zero represents break-even;
* negative values represent gains at the measured tail threshold.

For example, ``0.04`` means a 4% loss and ``-0.02`` means a 2% gain.
The corresponding return threshold always has the opposite sign.
"""

from __future__ import annotations

import math


LOSS_SPACE_CONVENTION = (
    "Signed loss in the input-return units: positive = loss, zero = "
    "break-even, negative = gain."
)


def _finite_float(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite, got {value}.")
    return converted


def return_threshold_to_loss_value(return_threshold: float) -> float:
    """Convert a signed return threshold to the signed loss-space value."""
    return -_finite_float(return_threshold, "return_threshold")


def loss_value_to_return_threshold(loss_value: float) -> float:
    """Convert a signed loss-space value to its signed return threshold."""
    return -_finite_float(loss_value, "loss_value")


def loss_value_to_money(loss_value: float, portfolio_value: float) -> float:
    """Linearly convert a decimal loss-space value to portfolio currency.

    The sign is preserved: a positive result is a monetary loss and a negative
    result is a monetary gain. ``portfolio_value`` may be zero but cannot be
    negative or non-finite. The multiplication is exact for simple-return loss
    fractions and only a first-order monetary approximation for log-return
    metrics.
    """
    loss = _finite_float(loss_value, "loss_value")
    capital = _finite_float(portfolio_value, "portfolio_value")
    if capital < 0.0:
        raise ValueError(
            f"portfolio_value must be non-negative, got {portfolio_value}."
        )
    return loss * capital
