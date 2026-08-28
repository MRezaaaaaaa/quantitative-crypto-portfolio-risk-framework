"""Strict monitoring-price normalization and launch-selection tests."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.monitoring.domain import DomainValidationError
from var_cvar_crypto_risk.monitoring.prices import (
    fingerprint_price_slice,
    normalize_monitoring_prices,
    resolve_launch_prices,
)


def _normalize(frame: pd.DataFrame):
    return normalize_monitoring_prices(
        frame,
        source="fixture",
        retrieved_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
    )


def test_normalization_preserves_missing_cells_and_never_forward_fills() -> None:
    frame = pd.DataFrame(
        {"btc": [100.0, np.nan, 120.0], "eth": [10.0, 11.0, 12.0]},
        index=["2026-01-01", "2026-01-02", "2026-01-03"],
    )
    normalized = _normalize(frame)
    assert pd.isna(normalized.prices.loc["2026-01-02", "BTC"])
    assert len(normalized.observations()) == 5
    assert normalized.symbols == ("BTC", "ETH")


def test_duplicate_utc_dates_are_rejected_instead_of_deduplicated() -> None:
    frame = pd.DataFrame(
        {"BTC": [100.0, 101.0]},
        index=["2026-01-01T00:00:00Z", "2025-12-31T19:00:00-05:00"],
    )
    with pytest.raises(DomainValidationError, match="duplicate monitoring dates"):
        _normalize(frame)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, "not-a-price"])
def test_invalid_present_prices_are_rejected(bad) -> None:
    frame = pd.DataFrame({"BTC": [100.0, bad]}, index=["2026-01-01", "2026-01-02"])
    with pytest.raises(DomainValidationError):
        _normalize(frame)


def test_launch_is_first_complete_observation_after_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "BTC": [100.0, 101.0, 102.0],
            "ETH": [10.0, np.nan, 12.0],
            "BENCH": [50.0, 51.0, 52.0],
        },
        index=["2026-01-01", "2026-01-02", "2026-01-03"],
    )
    selection = resolve_launch_prices(
        _normalize(frame),
        universe=["BTC", "ETH"],
        optimization_as_of=date(2026, 1, 1),
        benchmark_symbol="BENCH",
    )
    assert selection.launch_date == date(2026, 1, 3)
    assert selection.prices == {"BTC": 102.0, "ETH": 12.0, "BENCH": 52.0}


def test_requested_incomplete_launch_blocks_without_silent_shift() -> None:
    frame = pd.DataFrame(
        {"BTC": [100.0, 101.0, 102.0], "ETH": [10.0, np.nan, 12.0]},
        index=["2026-01-01", "2026-01-02", "2026-01-03"],
    )
    with pytest.raises(DomainValidationError, match="2026-01-02.*ETH"):
        resolve_launch_prices(
            _normalize(frame),
            universe=["BTC", "ETH"],
            optimization_as_of=date(2026, 1, 1),
            requested_launch_date=date(2026, 1, 2),
        )


def test_requested_date_cannot_skip_an_earlier_complete_launch() -> None:
    frame = pd.DataFrame(
        {"BTC": [100.0, 101.0, 102.0]},
        index=["2026-01-01", "2026-01-02", "2026-01-03"],
    )
    with pytest.raises(DomainValidationError, match="next complete"):
        resolve_launch_prices(
            _normalize(frame),
            universe=["BTC"],
            optimization_as_of=date(2026, 1, 1),
            requested_launch_date=date(2026, 1, 3),
        )


def test_point_in_time_slice_hash_ignores_values_outside_the_slice() -> None:
    frame = pd.DataFrame(
        {"BTC": [100.0, 101.0, 999.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )
    first = fingerprint_price_slice(
        frame.iloc[:2], source="fixture", quote_currency="USD"
    )
    frame.iloc[2, 0] = 1.0
    second = fingerprint_price_slice(
        frame.iloc[:2], source="fixture", quote_currency="USD"
    )
    assert first == second
