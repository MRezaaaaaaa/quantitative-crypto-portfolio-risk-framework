"""Offline tests for CoinGecko and yfinance adapters."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from var_cvar_crypto_risk import coingecko_client, yfinance_client


class _Response:
    def __init__(
        self,
        status_code: int,
        *,
        payload: dict | None = None,
        text: str = "response",
        json_error: ValueError | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._json_error = json_error

    def json(self) -> dict:
        if self._json_error is not None:
            raise self._json_error
        return self._payload or {}


def _timestamp(date: str, *, hour: int = 0) -> int:
    value = pd.Timestamp(f"{date}T{hour:02d}:00:00Z")
    return int(value.timestamp() * 1000)


def test_coingecko_date_and_cache_helpers(tmp_path: Path) -> None:
    assert coingecko_client._to_unix_seconds("1970-01-02") == 86_400
    cache = coingecko_client._cache_path(
        str(tmp_path), "bitcoin", "2024-01-01", "2024-01-02"
    )
    payload = {"prices": [[_timestamp("2024-01-01"), 100.0]]}

    assert coingecko_client._load_cached_payload(cache) is None
    coingecko_client._save_cached_payload(cache, payload)
    assert coingecko_client._load_cached_payload(cache) == payload

    cache.write_text("not-json", encoding="utf-8")
    assert coingecko_client._load_cached_payload(cache) is None


def test_coingecko_request_retries_transport_and_server_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            requests.ConnectionError("offline"),
            _Response(503, text="temporary"),
            _Response(200, payload={"prices": [[1, 2.0]]}),
        ]
    )
    sleeps: list[int] = []

    def fake_get(*_args, **_kwargs):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(coingecko_client.requests, "get", fake_get)
    monkeypatch.setattr(coingecko_client.time, "sleep", sleeps.append)

    payload = coingecko_client._request_with_retries("url", {}, {})

    assert payload == {"prices": [[1, 2.0]]}
    assert sleeps == [1, 2]


def test_coingecko_request_handles_rate_limit_and_bad_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            _Response(429),
            _Response(200, json_error=ValueError("invalid JSON")),
        ]
    )
    monkeypatch.setattr(
        coingecko_client.requests,
        "get",
        lambda *_args, **_kwargs: next(responses),
    )
    monkeypatch.setattr(coingecko_client.time, "sleep", lambda _seconds: None)

    with pytest.raises(coingecko_client.CoinGeckoError, match="non-JSON"):
        coingecko_client._request_with_retries("url", {}, {})


def test_coingecko_rate_limit_retry_transport_error_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter([_Response(429), requests.Timeout("timeout")])

    def fake_get(*_args, **_kwargs):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(coingecko_client.requests, "get", fake_get)
    monkeypatch.setattr(coingecko_client.time, "sleep", lambda _seconds: None)

    with pytest.raises(coingecko_client.CoinGeckoError, match="rate-limit retry"):
        coingecko_client._request_with_retries("url", {}, {})


@pytest.mark.parametrize(
    "responses",
    [
        [_Response(400, text="bad request")],
        [
            requests.ConnectionError("one"),
            requests.ConnectionError("two"),
            requests.ConnectionError("three"),
        ],
    ],
)
def test_coingecko_request_wraps_terminal_failures(
    monkeypatch: pytest.MonkeyPatch,
    responses: list,
) -> None:
    results = iter(responses)

    def fake_get(*_args, **_kwargs):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(coingecko_client.requests, "get", fake_get)
    monkeypatch.setattr(coingecko_client.time, "sleep", lambda _seconds: None)

    with pytest.raises(coingecko_client.CoinGeckoError, match="after retries"):
        coingecko_client._request_with_retries("url", {}, {})


def test_fetch_coingecko_uses_api_key_cache_and_last_daily_price(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    payload = {
        "prices": [
            [_timestamp("2024-01-01", hour=1), 100.0],
            [_timestamp("2024-01-01", hour=23), 101.0],
            [_timestamp("2024-01-02", hour=12), 102.0],
        ]
    }

    def fake_request(url: str, params: dict, headers: dict) -> dict:
        captured.update(url=url, params=params, headers=headers)
        return payload

    monkeypatch.setenv("COINGECKO_API_KEY", "test-placeholder")
    monkeypatch.setattr(coingecko_client, "_request_with_retries", fake_request)

    result = coingecko_client.fetch_coingecko_market_chart(
        "bitcoin",
        "usd",
        "2024-01-01",
        "2024-01-02",
        cache_dir=str(tmp_path),
    )

    assert result["bitcoin"].tolist() == [101.0, 102.0]
    assert captured["headers"]["x-cg-demo-api-key"] == "test-placeholder"
    assert captured["params"]["to"] - captured["params"]["from"] == 172_800
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    assert json.loads(cache_files[0].read_text(encoding="utf-8")) == payload

    monkeypatch.setattr(
        coingecko_client,
        "_request_with_retries",
        lambda *_args, **_kwargs: pytest.fail("cache should avoid request"),
    )
    cached = coingecko_client.fetch_coingecko_market_chart(
        "bitcoin",
        "usd",
        "2024-01-01",
        "2024-01-02",
        cache_dir=str(tmp_path),
    )
    pd.testing.assert_frame_equal(cached, result)


def test_fetch_coingecko_cache_failure_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = {"prices": [[_timestamp("2024-01-01"), 100.0]]}
    monkeypatch.setattr(
        coingecko_client, "_request_with_retries", lambda *_args, **_kwargs: payload
    )
    monkeypatch.setattr(
        coingecko_client,
        "_save_cached_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read-only")),
    )

    result = coingecko_client.fetch_coingecko_market_chart(
        "bitcoin", "usd", "2024-01-01", "2024-01-01", cache_dir=str(tmp_path)
    )

    assert result.iloc[0, 0] == pytest.approx(100.0)


def test_fetch_coingecko_rejects_empty_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        coingecko_client, "_request_with_retries", lambda *_args, **_kwargs: {}
    )

    with pytest.raises(coingecko_client.CoinGeckoError, match="no price data"):
        coingecko_client.fetch_coingecko_market_chart(
            "bitcoin", "usd", "2024-01-01", "2024-01-02", use_cache=False
        )


def test_fetch_multiple_coingecko_prices_aligns_and_renames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch(coin_id: str, **_kwargs) -> pd.DataFrame:
        if coin_id == "bitcoin":
            index = pd.date_range("2024-01-01", periods=3)
            return pd.DataFrame({coin_id: [100.0, 101.0, 102.0]}, index=index)
        index = pd.date_range("2024-01-02", periods=3)
        return pd.DataFrame({coin_id: [50.0, np.nan, 52.0]}, index=index)

    monkeypatch.setattr(
        coingecko_client, "fetch_coingecko_market_chart", fake_fetch
    )
    result = coingecko_client.fetch_multiple_coingecko_prices(
        {
            "BTC": {"coingecko_id": "bitcoin"},
            "ETH": {"coingecko_id": "ethereum"},
        },
        "usd",
        "2024-01-01",
    )

    assert list(result.columns) == ["BTC", "ETH"]
    assert list(result.index) == list(pd.date_range("2024-01-02", periods=2))
    assert result.loc[pd.Timestamp("2024-01-03"), "ETH"] == pytest.approx(50.0)


def test_fetch_multiple_coingecko_prices_validates_assets() -> None:
    with pytest.raises(coingecko_client.CoinGeckoError, match="missing"):
        coingecko_client.fetch_multiple_coingecko_prices(
            {"BTC": {}}, "usd", "2024-01-01"
        )
    with pytest.raises(coingecko_client.CoinGeckoError, match="No assets"):
        coingecko_client.fetch_multiple_coingecko_prices({}, "usd", "2024-01-01")


def _install_fake_yfinance(
    monkeypatch: pytest.MonkeyPatch,
    returned: pd.DataFrame | None,
    captured: dict | None = None,
) -> None:
    def download(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return returned

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))


def test_yfinance_normalizes_multi_asset_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = pd.MultiIndex.from_product(
        [["Close", "Volume"], ["BTC-USD", "ETH-USD"]]
    )
    raw = pd.DataFrame(
        [[100.0, 50.0, 1_000.0, 2_000.0], [101.0, 51.0, 1_100.0, 2_100.0]],
        index=pd.date_range("2024-01-01 12:00", periods=2),
        columns=columns,
    )
    captured: dict = {}
    _install_fake_yfinance(monkeypatch, raw, captured)

    result = yfinance_client.fetch_yfinance_prices(
        ["BTC-USD", "ETH-USD"], "2024-01-01", "2024-01-03"
    )

    assert list(result.columns) == ["BTC", "ETH"]
    assert result.index[0] == pd.Timestamp("2024-01-01")
    assert captured["auto_adjust"] is True
    assert captured["progress"] is False


def test_yfinance_single_ticker_flat_result(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame(
        {"Open": [99.0, 100.0], "Close": [100.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    _install_fake_yfinance(monkeypatch, raw)

    result = yfinance_client.fetch_yfinance_prices(
        ["BTC-USD"], "2024-01-01", "2024-01-03"
    )

    assert list(result.columns) == ["BTC"]
    assert result["BTC"].tolist() == [100.0, 101.0]
    assert yfinance_client._normalize_symbol("SPY") == "SPY"


@pytest.mark.parametrize("returned", [None, pd.DataFrame()])
def test_yfinance_rejects_empty_download(
    monkeypatch: pytest.MonkeyPatch,
    returned: pd.DataFrame | None,
) -> None:
    _install_fake_yfinance(monkeypatch, returned)

    with pytest.raises(ValueError, match="empty data"):
        yfinance_client.fetch_yfinance_prices(
            ["BTC-USD"], "2024-01-01", "2024-01-03"
        )


def test_yfinance_rejects_missing_price_field(monkeypatch: pytest.MonkeyPatch) -> None:
    columns = pd.MultiIndex.from_product([["Open"], ["BTC-USD", "ETH-USD"]])
    raw = pd.DataFrame(
        [[99.0, 49.0]], index=pd.date_range("2024-01-01", periods=1), columns=columns
    )
    _install_fake_yfinance(monkeypatch, raw)

    with pytest.raises(ValueError, match="not in yfinance result"):
        yfinance_client.fetch_yfinance_prices(
            ["BTC-USD", "ETH-USD"], "2024-01-01", "2024-01-03"
        )


def test_yfinance_rejects_flat_multi_asset_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = pd.DataFrame(
        {"Close": [100.0]}, index=pd.date_range("2024-01-01", periods=1)
    )
    _install_fake_yfinance(monkeypatch, raw)

    with pytest.raises(ValueError, match="Unexpected yfinance result shape"):
        yfinance_client.fetch_yfinance_prices(
            ["BTC-USD", "ETH-USD"], "2024-01-01", "2024-01-03"
        )


def test_yfinance_rejects_all_nan_price_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = pd.DataFrame(
        {"Close": [np.nan]}, index=pd.date_range("2024-01-01", periods=1)
    )
    _install_fake_yfinance(monkeypatch, raw)

    with pytest.raises(ValueError, match="no usable price columns"):
        yfinance_client.fetch_yfinance_prices(
            ["BTC-USD"], "2024-01-01", "2024-01-03"
        )
