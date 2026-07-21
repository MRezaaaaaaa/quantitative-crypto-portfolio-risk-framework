"""Price preprocessing contracts and edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.preprocessing import (
    align_price_data,
    clean_price_data,
    handle_missing_values,
)


def test_clean_price_data_sorts_normalizes_deduplicates_and_aligns() -> None:
    prices = pd.DataFrame(
        {
            "BTC": [102.0, 100.0, 101.0, 103.0],
            "ETH": [np.nan, 50.0, 51.0, 52.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-03 12:00",
                "2024-01-01 08:00",
                "2024-01-02 09:00",
                "2024-01-03 18:00",
            ]
        ),
    )

    cleaned = clean_price_data(prices)

    assert list(cleaned.index) == list(pd.date_range("2024-01-01", periods=3))
    assert cleaned.loc[pd.Timestamp("2024-01-03"), "BTC"] == pytest.approx(103.0)
    assert cleaned.loc[pd.Timestamp("2024-01-03"), "ETH"] == pytest.approx(52.0)
    assert prices.index[0].hour == 12


def test_align_price_data_handles_empty_inputs_and_one_day_gap() -> None:
    empty = pd.DataFrame()
    assert align_price_data(empty).empty

    no_columns = pd.DataFrame(index=pd.date_range("2024-01-01", periods=2))
    assert align_price_data(no_columns).shape == (2, 0)

    prices = pd.DataFrame(
        {
            "BTC": [100.0, np.nan, np.nan, 104.0],
            "ETH": [50.0, 51.0, 52.0, 53.0],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )
    aligned = align_price_data(prices)

    assert list(aligned.index) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-04"),
    ]
    assert aligned.iloc[1]["BTC"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("method", "expected_middle"),
    [("drop", None), ("ffill", 1.0), ("interpolate", 2.0)],
)
def test_handle_missing_values_methods(method: str, expected_middle: float | None) -> None:
    prices = pd.DataFrame({"BTC": [1.0, np.nan, 3.0]})

    result = handle_missing_values(prices, method)

    assert prices.isna().sum().sum() == 1
    if expected_middle is None:
        assert result["BTC"].tolist() == [1.0, 3.0]
    else:
        assert result["BTC"].tolist() == [1.0, expected_middle, 3.0]


def test_handle_missing_values_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        handle_missing_values(pd.DataFrame({"BTC": [1.0]}), "backfill")
