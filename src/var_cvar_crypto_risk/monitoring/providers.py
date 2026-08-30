"""Refreshable, dependency-injected price providers for one-shot monitoring.

Provider adapters return raw explicit observations plus provenance.  They do not
decide portfolio cutoffs, persist data, forward-fill missing values, or run on a
schedule.  Tests use fake implementations of :class:`RefreshablePriceProvider`
and never call a live API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import re
from typing import Any, Mapping, Protocol

import pandas as pd

from .domain import DomainValidationError, ensure_utc
from .recipes import SourceRecipe


_SAFE_SOURCE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


@dataclass(frozen=True)
class PriceFetchRequest:
    """Bounded request expressed in frozen project symbols."""

    symbols: tuple[str, ...]
    symbol_mapping: Mapping[str, str]
    quote_currency: str
    start_date: date
    end_date: date
    requested_at: datetime

    def __post_init__(self) -> None:
        symbols = tuple(str(item).strip().upper() for item in self.symbols)
        if not symbols or any(not item for item in symbols):
            raise DomainValidationError("price fetch requires a frozen universe")
        if len(symbols) != len(set(symbols)):
            raise DomainValidationError("price fetch symbols must be unique")
        if self.start_date > self.end_date:
            raise DomainValidationError("price fetch start_date exceeds end_date")
        mapping = {
            str(key).strip().upper(): str(value).strip()
            for key, value in self.symbol_mapping.items()
        }
        missing = [symbol for symbol in symbols if not mapping.get(symbol)]
        if missing:
            raise DomainValidationError(
                "source mapping is missing symbols: " + ", ".join(missing)
            )
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "symbol_mapping", mapping)
        object.__setattr__(self, "quote_currency", self.quote_currency.strip().upper())
        object.__setattr__(
            self, "requested_at", ensure_utc(self.requested_at, "requested_at")
        )


@dataclass(frozen=True)
class ProviderPriceBatch:
    """Raw provider response with the actual source and completeness boundary."""

    prices: pd.DataFrame
    actual_source: str
    quote_currency: str
    retrieved_at: datetime
    complete_through: date
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = self.actual_source.strip()
        quote = self.quote_currency.strip().upper()
        if not _SAFE_SOURCE_NAME.fullmatch(source) or not quote:
            raise DomainValidationError("provider source and quote currency are required")
        if not isinstance(self.prices, pd.DataFrame) or self.prices.empty:
            raise DomainValidationError("provider returned no price observations")
        object.__setattr__(self, "prices", self.prices.copy(deep=True))
        object.__setattr__(self, "actual_source", source)
        object.__setattr__(self, "quote_currency", quote)
        object.__setattr__(
            self, "retrieved_at", ensure_utc(self.retrieved_at, "retrieved_at")
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


class RefreshablePriceProvider(Protocol):
    """Synchronous one-shot provider seam suitable for scheduler invocation."""

    @property
    def provider_name(self) -> str: ...

    def fetch(self, request: PriceFetchRequest) -> ProviderPriceBatch: ...


class PriceProviderRegistry:
    """Resolve requested/fallback adapters without persisting credentials."""

    def __init__(self, providers: tuple[RefreshablePriceProvider, ...] | list[RefreshablePriceProvider]):
        self._providers = {
            provider.provider_name.strip().lower(): provider for provider in providers
        }
        if len(self._providers) != len(providers):
            raise DomainValidationError("refreshable provider names must be unique")
        if any(not _SAFE_SOURCE_NAME.fullmatch(name) for name in self._providers):
            raise DomainValidationError("refreshable provider name is not persistence-safe")

    def fetch(
        self,
        *,
        source: SourceRecipe,
        symbols: tuple[str, ...],
        start_date: date,
        end_date: date,
        requested_at: datetime,
    ) -> tuple[ProviderPriceBatch, bool]:
        """Fetch from the frozen provider, optionally using a declared fallback."""
        if not source.refreshable:
            raise DomainValidationError("monitoring source is not refreshable")
        requested_name = source.provider.strip().lower()
        primary = self._providers.get(requested_name)
        if primary is None:
            raise DomainValidationError(
                f"no refreshable adapter is registered for {source.provider!r}"
            )
        request = PriceFetchRequest(
            symbols=symbols,
            symbol_mapping=source.symbol_mapping,
            quote_currency=source.quote_currency,
            start_date=start_date,
            end_date=end_date,
            requested_at=requested_at,
        )
        try:
            return primary.fetch(request), False
        except Exception:
            fallback_name = str(source.metadata.get("fallback_provider", "")).strip().lower()
            if not fallback_name:
                raise
            fallback = self._providers.get(fallback_name)
            if fallback is None:
                raise DomainValidationError(
                    f"no refreshable fallback adapter is registered for {fallback_name!r}"
                )
            fallback_mapping = source.metadata.get("fallback_symbol_mapping")
            if fallback_mapping is None:
                fallback_mapping = source.symbol_mapping
            fallback_request = PriceFetchRequest(
                symbols=symbols,
                symbol_mapping=fallback_mapping,
                quote_currency=source.quote_currency,
                start_date=start_date,
                end_date=end_date,
                requested_at=requested_at,
            )
            return fallback.fetch(fallback_request), True


class CoinGeckoRefreshableProvider:
    """CoinGecko adapter that retains missing dates and never forward-fills."""

    provider_name = "coingecko"

    def fetch(self, request: PriceFetchRequest) -> ProviderPriceBatch:
        from ..coingecko_client import fetch_coingecko_market_chart

        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            provider_symbol = request.symbol_mapping[symbol]
            frame = fetch_coingecko_market_chart(
                coin_id=provider_symbol,
                vs_currency=request.quote_currency.lower(),
                start_date=request.start_date.isoformat(),
                end_date=request.end_date.isoformat(),
                cache_dir=None,
                use_cache=False,
            ).rename(columns={provider_symbol: symbol})
            frames.append(frame)
        prices = pd.concat(frames, axis=1, join="outer").sort_index()
        retrieved_at = datetime.now(timezone.utc)
        complete_through = min(request.end_date, prices.index.max().date())
        return ProviderPriceBatch(
            prices=prices,
            actual_source=self.provider_name,
            quote_currency=request.quote_currency,
            retrieved_at=retrieved_at,
            complete_through=complete_through,
            metadata={"requested_provider": self.provider_name},
        )


class YFinanceRefreshableProvider:
    """yfinance adapter; source limitations remain research-grade only."""

    provider_name = "yfinance"

    def fetch(self, request: PriceFetchRequest) -> ProviderPriceBatch:
        from ..yfinance_client import fetch_yfinance_prices

        tickers = [request.symbol_mapping[symbol] for symbol in request.symbols]
        exclusive_end = request.end_date + timedelta(days=1)
        prices = fetch_yfinance_prices(
            tickers=tickers,
            start_date=request.start_date.isoformat(),
            end_date=exclusive_end.isoformat(),
        )
        provider_to_project = {
            ticker.upper().removesuffix("-USD"): symbol
            for symbol, ticker in request.symbol_mapping.items()
        }
        prices = prices.rename(
            columns={
                column: provider_to_project.get(str(column).upper(), str(column).upper())
                for column in prices.columns
            }
        ).reindex(columns=list(request.symbols))
        retrieved_at = datetime.now(timezone.utc)
        complete_through = min(request.end_date, prices.index.max().date())
        return ProviderPriceBatch(
            prices=prices,
            actual_source=self.provider_name,
            quote_currency=request.quote_currency,
            retrieved_at=retrieved_at,
            complete_through=complete_through,
            metadata={
                "requested_provider": self.provider_name,
                "research_grade_vendor": True,
            },
        )


def default_provider_registry() -> PriceProviderRegistry:
    """Return public refresh adapters; credentials stay in provider environments."""
    return PriceProviderRegistry(
        [CoinGeckoRefreshableProvider(), YFinanceRefreshableProvider()]
    )


__all__ = [
    "CoinGeckoRefreshableProvider",
    "PriceFetchRequest",
    "PriceProviderRegistry",
    "ProviderPriceBatch",
    "RefreshablePriceProvider",
    "YFinanceRefreshableProvider",
    "default_provider_registry",
]
