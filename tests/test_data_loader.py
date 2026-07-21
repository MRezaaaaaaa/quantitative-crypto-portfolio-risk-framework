"""Tests for the data loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from var_cvar_crypto_risk import coingecko_client, yfinance_client
from var_cvar_crypto_risk.data_loader import (
    load_price_data,
    load_price_data_from_csv,
    validate_price_data,
)


def _write_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "BTC": [30000.0, 30500.0, 30100.0, 30900.0, 31000.0],
            "ETH": [2000.0, 2050.0, 2010.0, 2080.0, 2100.0],
        }
    )
    path = tmp_path / "prices.csv"
    df.to_csv(path, index=False)
    return path


def test_load_price_data_from_csv(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    loaded = load_price_data_from_csv(str(csv_path))
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert list(loaded.columns) == ["BTC", "ETH"]
    assert len(loaded) == 5
    assert (loaded > 0).all().all()


def test_validate_price_data_rejects_empty() -> None:
    with pytest.raises(ValueError):
        validate_price_data(pd.DataFrame())


def test_validate_price_data_rejects_negative() -> None:
    df = pd.DataFrame(
        {"BTC": [-1.0, 30000.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    with pytest.raises(ValueError):
        validate_price_data(df)


def test_validate_price_data_rejects_non_datetime_index() -> None:
    df = pd.DataFrame({"BTC": [30000.0, 30100.0]}, index=[0, 1])
    with pytest.raises(ValueError):
        validate_price_data(df)


def test_validate_price_data_passes_for_valid_data() -> None:
    df = pd.DataFrame(
        {"BTC": [30000.0, 30100.0, 30050.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    validate_price_data(df)


def _runtime_config(source: str = "coingecko") -> dict:
    return {
        "data": {
            "source": source,
            "fallback_source": "yfinance",
            "start_date": "2024-01-01",
            "end_date": "2024-01-03",
            "quote_currency": "usd",
            "cache_dir": "unused-cache",
            "cache_enabled": False,
        },
        "assets": {
            "BTC": {
                "coingecko_id": "bitcoin",
                "yfinance_ticker": "BTC-USD",
            },
            "ETH": {
                "coingecko_id": "ethereum",
                "yfinance_ticker": "ETH-USD",
            },
        },
    }


def _valid_prices(*, columns: tuple[str, ...] = ("BTC", "ETH")) -> pd.DataFrame:
    values = {
        "BTC": [100.0, 101.0, 102.0],
        "ETH": [50.0, 51.0, 52.0],
        "EXTRA": [10.0, 11.0, 12.0],
    }
    return pd.DataFrame(
        {column: values[column] for column in columns},
        index=pd.date_range("2024-01-01", periods=3),
    )


def test_load_price_data_uses_coingecko_and_asset_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_fetch(**kwargs) -> pd.DataFrame:
        captured.update(kwargs)
        return _valid_prices(columns=("EXTRA", "ETH", "BTC"))

    monkeypatch.setattr(
        coingecko_client, "fetch_multiple_coingecko_prices", fake_fetch
    )

    result = load_price_data(_runtime_config())

    assert list(result.columns) == ["BTC", "ETH"]
    assert captured["use_cache"] is False
    assert captured["vs_currency"] == "usd"


def test_load_price_data_falls_back_to_yfinance(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        coingecko_client,
        "fetch_multiple_coingecko_prices",
        lambda **_kwargs: (_ for _ in ()).throw(
            coingecko_client.CoinGeckoError("offline")
        ),
    )
    monkeypatch.setattr(
        yfinance_client,
        "fetch_yfinance_prices",
        lambda **_kwargs: _valid_prices(),
    )

    result = load_price_data(_runtime_config())

    assert list(result.columns) == ["BTC", "ETH"]
    assert "Falling back to yfinance" in caplog.text


def test_load_price_data_reraises_coingecko_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config()
    config["data"]["fallback_source"] = None
    monkeypatch.setattr(
        coingecko_client,
        "fetch_multiple_coingecko_prices",
        lambda **_kwargs: (_ for _ in ()).throw(
            coingecko_client.CoinGeckoError("offline")
        ),
    )

    with pytest.raises(coingecko_client.CoinGeckoError, match="offline"):
        load_price_data(config)


def test_load_price_data_uses_yfinance_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_fetch(**kwargs) -> pd.DataFrame:
        captured.update(kwargs)
        return _valid_prices()

    monkeypatch.setattr(yfinance_client, "fetch_yfinance_prices", fake_fetch)

    result = load_price_data(_runtime_config("yfinance"))

    assert list(result.columns) == ["BTC", "ETH"]
    assert captured["tickers"] == ["BTC-USD", "ETH-USD"]


@pytest.mark.parametrize(
    ("source", "message"),
    [("csv", "must be loaded"), ("database", "Unknown data source")],
)
def test_load_price_data_rejects_unsupported_or_indirect_source(
    source: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_price_data(_runtime_config(source))


def test_load_csv_reports_missing_file_date_and_numeric_columns(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        load_price_data_from_csv(str(tmp_path / "missing.csv"))

    no_date = tmp_path / "no-date.csv"
    pd.DataFrame({"BTC": [100.0]}).to_csv(no_date, index=False)
    with pytest.raises(ValueError, match="Expected date column"):
        load_price_data_from_csv(str(no_date))

    no_prices = tmp_path / "no-prices.csv"
    pd.DataFrame({"date": ["2024-01-01"], "label": ["BTC"]}).to_csv(
        no_prices, index=False
    )
    with pytest.raises(ValueError, match="No numeric price columns"):
        load_price_data_from_csv(str(no_prices))


def test_load_csv_sorts_deduplicates_and_selects_explicit_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01", "2024-01-02"],
            "BTC": [101.0, 100.0, 102.0],
            "ETH": [51.0, 50.0, 52.0],
            "note": ["first", "second", "last"],
        }
    ).to_csv(path, index=False)

    result = load_price_data_from_csv(str(path), price_cols=["ETH"])

    assert list(result.columns) == ["ETH"]
    assert list(result.index) == list(pd.date_range("2024-01-01", periods=2))
    assert result.iloc[-1, 0] == pytest.approx(52.0)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame(
                {"BTC": [101.0, 100.0]},
                index=pd.to_datetime(["2024-01-02", "2024-01-01"]),
            ),
            "sorted ascending",
        ),
        (
            pd.DataFrame(
                {"BTC": [100.0, 101.0]},
                index=pd.to_datetime(["2024-01-01", "2024-01-01"]),
            ),
            "duplicate dates",
        ),
        (
            pd.DataFrame(
                {"BTC": ["100", "101"]},
                index=pd.date_range("2024-01-01", periods=2),
            ),
            "not numeric",
        ),
        (
            pd.DataFrame(
                {"BTC": [float("nan"), float("nan")]},
                index=pd.date_range("2024-01-01", periods=2),
            ),
            "entirely NaN",
        ),
        (
            pd.DataFrame(
                {"BTC": [0.0, 101.0]},
                index=pd.date_range("2024-01-01", periods=2),
            ),
            "non-positive",
        ),
    ],
)
def test_validate_price_data_rejects_invalid_contracts(
    frame: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_price_data(frame)
