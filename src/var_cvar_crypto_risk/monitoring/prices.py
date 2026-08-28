"""Strict point-in-time price normalization for portfolio monitoring.

Unlike the research preprocessing path, this module never sorts away duplicate
observations, forward-fills prices, or drops an incomplete date silently.
Missing values remain visible so valuation can record an incomplete state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from .domain import DomainValidationError, PriceDataStatus, PriceObservation
from .hashing import sha256_fingerprint


@dataclass(frozen=True)
class NormalizedPriceData:
    """Validated wide price matrix plus explicit source provenance."""

    prices: pd.DataFrame
    source: str
    quote_currency: str
    retrieved_at: datetime
    fingerprint: str

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self.prices.columns)

    def observations(self) -> tuple[PriceObservation, ...]:
        """Return only explicit non-missing observations; no values are invented."""
        rows: list[PriceObservation] = []
        for timestamp, row in self.prices.iterrows():
            observation_date = timestamp.date()
            for symbol, value in row.items():
                if pd.isna(value):
                    continue
                rows.append(
                    PriceObservation(
                        symbol=str(symbol),
                        observation_date=observation_date,
                        price=float(value),
                        quote_currency=self.quote_currency,
                        source=self.source,
                        retrieved_at=self.retrieved_at,
                        data_status=PriceDataStatus.COMPLETE,
                    )
                )
        return tuple(rows)


@dataclass(frozen=True)
class LaunchPriceSelection:
    """The reviewed first complete observation after an optimization cutoff."""

    launch_date: date
    prices: dict[str, float]


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    try:
        parsed = pd.to_datetime(index, errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise DomainValidationError(
            "monitoring price index must contain parseable UTC dates"
        ) from exc
    normalized = pd.DatetimeIndex(parsed).normalize().tz_convert(None)
    if normalized.has_duplicates:
        duplicated = sorted(
            {item.date().isoformat() for item in normalized[normalized.duplicated(False)]}
        )
        raise DomainValidationError(
            "duplicate monitoring dates after UTC normalization: "
            + ", ".join(duplicated)
        )
    return normalized


def _normalize_columns(columns: pd.Index) -> list[str]:
    normalized = [str(column).strip().upper() for column in columns]
    if any(not column for column in normalized):
        raise DomainValidationError("monitoring price columns must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise DomainValidationError(
            "duplicate asset symbols after monitoring column normalization"
        )
    return normalized


def _fingerprint_payload(
    prices: pd.DataFrame,
    *,
    source: str,
    quote_currency: str,
) -> dict:
    records: list[dict] = []
    for timestamp, row in prices.iterrows():
        records.append(
            {
                "date": timestamp.date(),
                "prices": {
                    str(symbol): None if pd.isna(value) else float(value)
                    for symbol, value in row.items()
                },
            }
        )
    return {
        "source": source,
        "quote_currency": quote_currency,
        "symbols": list(prices.columns),
        "records": records,
    }


def normalize_monitoring_prices(
    prices: pd.DataFrame,
    *,
    source: str,
    quote_currency: str = "USD",
    retrieved_at: datetime | None = None,
) -> NormalizedPriceData:
    """Validate wide daily prices while retaining explicit missing values.

    The input is copied. Dates are converted to UTC calendar dates, symbols are
    normalized to uppercase, and present values must be numeric, finite, and
    strictly positive. Missing cells remain ``NaN`` for quality reporting.
    """
    if not isinstance(prices, pd.DataFrame):
        raise DomainValidationError("monitoring prices must be a pandas DataFrame")
    if prices.empty or len(prices.columns) == 0:
        raise DomainValidationError("monitoring prices must not be empty")
    declared_source = source.strip()
    quote = quote_currency.strip().upper()
    if not declared_source or not quote:
        raise DomainValidationError("source and quote_currency are required")
    retrieved = retrieved_at or datetime.now(timezone.utc)
    if retrieved.tzinfo is None or retrieved.utcoffset() is None:
        raise DomainValidationError("retrieved_at must be timezone-aware")
    retrieved = retrieved.astimezone(timezone.utc)

    frame = prices.copy(deep=True)
    frame.index = _normalize_index(frame.index)
    frame.columns = _normalize_columns(frame.columns)
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()

    converted = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        try:
            series = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise DomainValidationError(
                f"monitoring price column {column!r} contains non-numeric values"
            ) from exc
        present = series.dropna()
        if not np.isfinite(present.to_numpy(dtype=float)).all():
            raise DomainValidationError(
                f"monitoring price column {column!r} contains non-finite values"
            )
        if (present <= 0.0).any():
            raise DomainValidationError(
                f"monitoring price column {column!r} contains non-positive values"
            )
        converted[column] = series

    fingerprint = sha256_fingerprint(
        _fingerprint_payload(
            converted,
            source=declared_source,
            quote_currency=quote,
        )
    )
    return NormalizedPriceData(
        prices=converted,
        source=declared_source,
        quote_currency=quote,
        retrieved_at=retrieved,
        fingerprint=fingerprint,
    )


def fingerprint_price_slice(
    prices: pd.DataFrame,
    *,
    source: str,
    quote_currency: str,
) -> str:
    """Fingerprint an already-normalized point-in-time slice."""
    return sha256_fingerprint(
        _fingerprint_payload(
            prices,
            source=source.strip(),
            quote_currency=quote_currency.strip().upper(),
        )
    )


def missing_symbols_on_date(
    normalized: NormalizedPriceData,
    observation_date: date,
    required_symbols: Iterable[str],
) -> tuple[str, ...]:
    """Return absent columns and explicit missing cells for one date."""
    required = tuple(dict.fromkeys(str(item).strip().upper() for item in required_symbols))
    timestamp = pd.Timestamp(observation_date)
    missing: list[str] = []
    if timestamp not in normalized.prices.index:
        return required
    row = normalized.prices.loc[timestamp]
    for symbol in required:
        if symbol not in normalized.prices.columns or pd.isna(row.get(symbol)):
            missing.append(symbol)
    return tuple(missing)


def resolve_launch_prices(
    normalized: NormalizedPriceData,
    *,
    universe: Iterable[str],
    optimization_as_of: date,
    requested_launch_date: date | None = None,
    benchmark_symbol: str | None = None,
) -> LaunchPriceSelection:
    """Resolve the first complete price date after the point-in-time cutoff.

    A supplied requested date is validation input, not permission to shift the
    launch. It must equal the first complete observation after the cutoff.
    """
    assets = tuple(dict.fromkeys(str(item).strip().upper() for item in universe))
    if not assets or any(not item for item in assets):
        raise DomainValidationError("a non-empty frozen universe is required")
    required = assets
    if benchmark_symbol is not None:
        benchmark = benchmark_symbol.strip().upper()
        if benchmark and benchmark not in required:
            required = (*required, benchmark)
    absent_columns = [item for item in required if item not in normalized.prices.columns]
    if absent_columns:
        raise DomainValidationError(
            "monitoring prices are missing required symbols: "
            + ", ".join(absent_columns)
        )

    candidates = normalized.prices.loc[
        normalized.prices.index > pd.Timestamp(optimization_as_of), list(required)
    ]
    complete = candidates.dropna(how="any")
    if complete.empty:
        raise DomainValidationError(
            "no complete launch observation exists after optimization_as_of"
        )
    first_complete = complete.index[0].date()
    if requested_launch_date is not None:
        if requested_launch_date <= optimization_as_of:
            raise DomainValidationError(
                "requested launch date must follow optimization_as_of"
            )
        missing = missing_symbols_on_date(normalized, requested_launch_date, required)
        if missing:
            raise DomainValidationError(
                f"requested launch date {requested_launch_date.isoformat()} is "
                "incomplete; missing: " + ", ".join(missing)
            )
        if requested_launch_date != first_complete:
            raise DomainValidationError(
                "requested launch date must equal the next complete observation "
                f"({first_complete.isoformat()})"
            )
    launch_date = requested_launch_date or first_complete
    row = normalized.prices.loc[pd.Timestamp(launch_date)]
    return LaunchPriceSelection(
        launch_date=launch_date,
        prices={symbol: float(row[symbol]) for symbol in required},
    )


__all__ = [
    "LaunchPriceSelection",
    "NormalizedPriceData",
    "fingerprint_price_slice",
    "missing_symbols_on_date",
    "normalize_monitoring_prices",
    "resolve_launch_prices",
]
