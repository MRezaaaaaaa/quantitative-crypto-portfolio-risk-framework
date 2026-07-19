"""CoinGecko data source client.

This is the **primary** data source for the risk engine.
Uses the public free endpoint:

    https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range

If the environment variable ``COINGECKO_API_KEY`` is set, it is added as
the ``x-cg-demo-api-key`` request header.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


class CoinGeckoError(Exception):
    """Raised when CoinGecko API fails after all retries."""


def _to_unix_seconds(date_str: str) -> int:
    """Convert a ``YYYY-MM-DD`` date string to a UTC Unix timestamp (seconds)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _cache_path(cache_dir: str, coin_id: str, start: str, end: str) -> Path:
    return Path(cache_dir) / f"{coin_id}_{start}_{end}.json"


def _load_cached_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _save_cached_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _request_with_retries(url: str, params: dict, headers: dict) -> dict:
    """GET ``url`` with up to 3 retries, exponential backoff, and 429 handling."""
    backoff = [1, 2, 4]
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(backoff[attempt])
            continue

        if response.status_code == 429:
            time.sleep(60)
            try:
                response = requests.get(
                    url, params=params, headers=headers, timeout=30
                )
            except requests.exceptions.RequestException as exc:
                raise CoinGeckoError(
                    f"CoinGecko request failed after rate-limit retry: {exc}"
                ) from exc

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise CoinGeckoError(
                    f"CoinGecko returned non-JSON response: {exc}"
                ) from exc

        last_error = CoinGeckoError(
            f"CoinGecko returned HTTP {response.status_code}: {response.text[:200]}"
        )
        if response.status_code in (500, 502, 503, 504) and attempt < 2:
            time.sleep(backoff[attempt])
            continue
        break

    raise CoinGeckoError(
        f"CoinGecko request failed after retries: {last_error}"
    )


def fetch_coingecko_market_chart(
    coin_id: str,
    vs_currency: str,
    start_date: str,
    end_date: str | None = None,
    cache_dir: str | None = "data/cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily historical prices from CoinGecko for a single coin.

    Parameters
    ----------
    coin_id : str
        CoinGecko coin id (e.g. ``"bitcoin"``).
    vs_currency : str
        Quote currency (e.g. ``"usd"``).
    start_date : str
        Start date in ``YYYY-MM-DD`` format.
    end_date : str | None, optional
        End date in ``YYYY-MM-DD`` format. ``None`` means today (UTC).
    cache_dir : str | None, optional
        Directory for cached JSON payloads.
    use_cache : bool, optional
        If True, read from / write to ``cache_dir``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a ``DatetimeIndex`` (date only) and a single
        column named ``coin_id`` containing daily prices in ``vs_currency``.

    Raises
    ------
    CoinGeckoError
        If all network retries fail or the API returns an error response.
    """
    if end_date is None:
        end_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    payload: dict | None = None
    cache_file: Path | None = None
    if use_cache and cache_dir is not None:
        cache_file = _cache_path(cache_dir, coin_id, start_date, end_date)
        payload = _load_cached_payload(cache_file)

    if payload is None:
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": vs_currency,
            "from": _to_unix_seconds(start_date),
            "to": _to_unix_seconds(end_date) + 86400,  # inclusive end
        }
        headers = {"accept": "application/json"}
        api_key = os.environ.get("COINGECKO_API_KEY")
        if api_key:
            headers["x-cg-demo-api-key"] = api_key

        payload = _request_with_retries(url, params, headers)

        if use_cache and cache_file is not None:
            try:
                _save_cached_payload(cache_file, payload)
            except OSError:
                pass  # caching is best-effort

    prices = payload.get("prices") if isinstance(payload, dict) else None
    if not prices:
        raise CoinGeckoError(
            f"CoinGecko returned no price data for '{coin_id}' between "
            f"{start_date} and {end_date}."
        )

    df = pd.DataFrame(prices, columns=["timestamp_ms", coin_id])
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.normalize()
    df = df.drop(columns=["timestamp_ms"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.set_index("date").sort_index()
    df.index.name = None
    return df


def fetch_multiple_coingecko_prices(
    assets: dict,
    vs_currency: str,
    start_date: str,
    end_date: str | None = None,
    cache_dir: str | None = "data/cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch prices for multiple assets and return a combined DataFrame.

    Parameters
    ----------
    assets : dict
        Mapping from ``assets.yaml``, e.g.
        ``{"BTC": {"coingecko_id": "bitcoin", ...}}``.
    vs_currency : str
        Quote currency.
    start_date, end_date : str
        Date range (``YYYY-MM-DD``).
    cache_dir, use_cache : optional
        See :func:`fetch_coingecko_market_chart`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``DatetimeIndex`` and one column per asset symbol.
    """
    series_list: list[pd.DataFrame] = []
    for symbol, meta in assets.items():
        coin_id = meta.get("coingecko_id")
        if not coin_id:
            raise CoinGeckoError(
                f"Asset '{symbol}' is missing 'coingecko_id' in assets.yaml."
            )
        single = fetch_coingecko_market_chart(
            coin_id=coin_id,
            vs_currency=vs_currency,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
        single = single.rename(columns={coin_id: symbol})
        series_list.append(single)

    if not series_list:
        raise CoinGeckoError("No assets provided to fetch_multiple_coingecko_prices.")

    combined = pd.concat(series_list, axis=1, join="inner")
    combined = combined.sort_index()
    combined = combined.ffill(limit=1)
    return combined
