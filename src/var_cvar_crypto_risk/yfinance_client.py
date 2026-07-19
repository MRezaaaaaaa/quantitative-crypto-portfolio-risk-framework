"""yfinance fallback data source.

yfinance is used as a fallback for educational and research purposes only.
Data accuracy is not guaranteed. For production use, prefer a licensed
data provider.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd


_YFINANCE_INSTALL_HINT = (
    "yfinance is required for the yfinance data source. "
    "Install it with: pip install yfinance"
)


def _normalize_symbol(ticker: str) -> str:
    """Strip the ``-USD`` suffix so ``BTC-USD`` becomes ``BTC``."""
    if ticker.upper().endswith("-USD"):
        return ticker[:-4]
    return ticker


def fetch_yfinance_prices(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
    price_field: str = "Close",
) -> pd.DataFrame:
    """Fetch historical prices using yfinance.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance tickers, e.g. ``["BTC-USD", "ETH-USD", "SOL-USD"]``.
    start_date : str
        ``YYYY-MM-DD``.
    end_date : str | None, optional
        ``YYYY-MM-DD``. ``None`` means tomorrow (so today's bar is included).
    price_field : str, optional
        ``"Close"`` or ``"Adj Close"``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``DatetimeIndex`` (date only). Column names are
        normalized asset symbols (e.g. ``"BTC"`` rather than ``"BTC-USD"``).

    Raises
    ------
    ValueError
        If the result is empty after cleaning.
    """
    if end_date is None:
        end_date = (datetime.now(tz=timezone.utc) + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised in test environment
        raise ImportError(_YFINANCE_INSTALL_HINT) from exc

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"yfinance returned empty data for tickers={tickers} "
            f"between {start_date} and {end_date}."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        if price_field not in raw.columns.get_level_values(0):
            available = sorted(set(raw.columns.get_level_values(0)))
            raise ValueError(
                f"price_field='{price_field}' not in yfinance result. "
                f"Available: {available}"
            )
        prices = raw[price_field].copy()
    else:
        if len(tickers) == 1:
            prices = raw[[price_field]].copy()
            prices.columns = [tickers[0]]
        else:
            raise ValueError(
                "Unexpected yfinance result shape: expected MultiIndex columns."
            )

    prices.columns = [_normalize_symbol(str(col)) for col in prices.columns]
    prices.index = pd.to_datetime(prices.index).normalize()
    prices.index.name = None
    prices = prices.dropna(axis=1, how="all")

    if prices.empty or prices.shape[1] == 0:
        raise ValueError("yfinance returned no usable price columns after cleaning.")

    return prices
