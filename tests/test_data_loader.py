"""Tests for the data loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from var_cvar_crypto_risk.data_loader import (
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
