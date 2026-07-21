"""Contracts for small deterministic utility helpers."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.utils import (
    annual_to_horizon_rate,
    format_percentage,
    get_today_str,
    parse_date,
    safe_divide,
    set_random_seed,
)


def test_parse_date_normalizes_and_preserves_none() -> None:
    assert parse_date(None) is None
    assert parse_date("2024-01-02 18:45") == pd.Timestamp("2024-01-02")


def test_today_string_uses_iso_date_shape() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", get_today_str())


def test_random_seed_is_reproducible() -> None:
    set_random_seed(17)
    first = np.random.random(3)
    set_random_seed(17)
    second = np.random.random(3)
    np.testing.assert_array_equal(first, second)


def test_format_percentage_and_safe_divide() -> None:
    assert format_percentage(0.04255, decimals=2) == "4.25%"
    assert safe_divide(6.0, 3.0) == pytest.approx(2.0)
    assert safe_divide(1.0, 0.0, default=-1.0) == pytest.approx(-1.0)


def test_annual_to_horizon_rate_rejects_nonpositive_day_count() -> None:
    with pytest.raises(ValueError, match="day_count must be > 0"):
        annual_to_horizon_rate(0.05, horizon_days=7, day_count=0)
