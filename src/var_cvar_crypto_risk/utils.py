"""Small reusable utilities."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def parse_date(date_value: str | None) -> pd.Timestamp | None:
    """Parse a date string to ``pd.Timestamp``. Return ``None`` if input is None."""
    if date_value is None:
        return None
    return pd.Timestamp(date_value).normalize()


def get_today_str() -> str:
    """Return today's UTC date as a ``YYYY-MM-DD`` string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def set_random_seed(seed: int = 42) -> None:
    """Set numpy's global random seed for reproducibility."""
    np.random.seed(seed)


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string. ``0.0425 -> '4.25%'``."""
    return f"{value * 100:.{decimals}f}%"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Divide ``a`` by ``b``. Return ``default`` if ``b == 0``."""
    if b == 0:
        return default
    return a / b


def annual_to_horizon_rate(
    annual_rate: float,
    horizon_days: int,
    day_count: int = 365,
) -> float:
    """Convert an annual rate to the equivalent compounded per-horizon rate.

    ``cash_return_per_horizon = (1 + annual_rate) ** (horizon_days / day_count) - 1``

    Parameters
    ----------
    annual_rate : float
        Annual risk-free rate (e.g. ``0.05`` for 5%). ``0.0`` ⇒ returns 0.0.
    horizon_days : int
        Horizon length in days. Must be ``>= 1``.
    day_count : int
        Days per year for the compounding base. Must be ``> 0``.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}.")
    if day_count <= 0:
        raise ValueError(f"day_count must be > 0, got {day_count}.")
    return (1.0 + float(annual_rate)) ** (horizon_days / day_count) - 1.0
