"""Data loading orchestrator.

Selects the appropriate data source (CoinGecko, yfinance, CSV) based on
configuration, validates the result, and applies preprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .preprocessing import clean_price_data


logger = logging.getLogger(__name__)


def load_price_data(config: dict) -> pd.DataFrame:
    """Load price data according to ``config["data"]["source"]``.

    If the primary source is ``coingecko`` and it raises
    :class:`CoinGeckoError`, the loader falls back to ``yfinance`` when
    ``data.fallback_source == "yfinance"`` and emits a warning.

    Parameters
    ----------
    config : dict
        Merged config dictionary (see :func:`config.load_project_config`).

    Returns
    -------
    pandas.DataFrame
        Clean price DataFrame with ``DatetimeIndex`` and one column per asset.
    """
    data_cfg = config["data"]
    source = data_cfg["source"]
    start_date = data_cfg["start_date"]
    end_date = data_cfg.get("end_date")
    quote_currency = data_cfg.get("quote_currency", "usd")
    cache_dir = data_cfg.get("cache_dir", "data/cache")
    use_cache = bool(data_cfg.get("cache_enabled", True))
    assets: dict = config["assets"]

    if source == "coingecko":
        from .coingecko_client import (  # noqa: PLC0415
            CoinGeckoError,
            fetch_multiple_coingecko_prices,
        )
        try:
            prices = fetch_multiple_coingecko_prices(
                assets=assets,
                vs_currency=quote_currency,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
        except CoinGeckoError as exc:
            fallback = data_cfg.get("fallback_source")
            if fallback == "yfinance":
                logger.warning(
                    "CoinGecko fetch failed (%s). Falling back to yfinance.", exc
                )
                from .yfinance_client import fetch_yfinance_prices  # noqa: PLC0415

                tickers = [meta["yfinance_ticker"] for meta in assets.values()]
                prices = fetch_yfinance_prices(
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                raise
    elif source == "yfinance":
        from .yfinance_client import fetch_yfinance_prices  # noqa: PLC0415

        tickers = [meta["yfinance_ticker"] for meta in assets.values()]
        prices = fetch_yfinance_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
    elif source == "csv":
        raise ValueError(
            "source='csv' must be loaded with load_price_data_from_csv()."
        )
    else:
        raise ValueError(
            f"Unknown data source '{source}'. "
            f"Use 'coingecko', 'yfinance', or 'csv'."
        )

    asset_symbols = list(assets.keys())
    available = [sym for sym in asset_symbols if sym in prices.columns]
    if available:
        prices = prices[available]

    cleaned = clean_price_data(prices)
    validate_price_data(cleaned)
    return cleaned


def load_price_data_from_csv(
    path: str,
    date_col: str = "date",
    price_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load price data from a local CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    date_col : str, optional
        Name of the date column (default ``"date"``).
    price_cols : list[str] | None, optional
        Columns to keep as price columns. If ``None`` use all numeric columns
        except the date column.

    Returns
    -------
    pandas.DataFrame
        Cleaned price DataFrame.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    raw = pd.read_csv(file_path)
    if date_col not in raw.columns:
        raise ValueError(
            f"Expected date column '{date_col}' in CSV. "
            f"Found columns: {list(raw.columns)}"
        )

    raw[date_col] = pd.to_datetime(raw[date_col]).dt.normalize()
    raw = raw.sort_values(date_col)
    raw = raw.drop_duplicates(subset=[date_col], keep="last")
    raw = raw.set_index(date_col)
    raw.index.name = None

    if price_cols is None:
        price_cols = [
            col for col in raw.columns if pd.api.types.is_numeric_dtype(raw[col])
        ]

    if not price_cols:
        raise ValueError("No numeric price columns found in CSV.")

    df = raw[price_cols].copy()
    validate_price_data(df)
    return df


def validate_price_data(prices: pd.DataFrame) -> None:
    """Validate that a price DataFrame meets all requirements.

    Raises
    ------
    ValueError
        With a clear message if any check fails.
    """
    if prices is None or prices.empty:
        raise ValueError("Price data is empty (no rows).")
    if prices.shape[1] == 0:
        raise ValueError("Price data has no columns.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError(
            "Price data index must be a DatetimeIndex, "
            f"got {type(prices.index).__name__}."
        )
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price data index must be sorted ascending by date.")
    if prices.index.duplicated().any():
        raise ValueError("Price data index contains duplicate dates.")

    for col in prices.columns:
        if not pd.api.types.is_numeric_dtype(prices[col]):
            raise ValueError(
                f"Column '{col}' is not numeric (dtype={prices[col].dtype})."
            )
        if prices[col].isna().all():
            raise ValueError(f"Column '{col}' is entirely NaN.")
        non_nan = prices[col].dropna()
        if (non_nan <= 0).any():
            raise ValueError(
                f"Column '{col}' contains non-positive prices. "
                f"All prices must be > 0."
            )
