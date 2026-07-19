"""Price-data preprocessing utilities."""

from __future__ import annotations

import pandas as pd


def clean_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline: sort, dedupe, handle missing values.

    Returns a clean copy. Does not modify ``prices`` in place.
    """
    df = prices.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = align_price_data(df)
    return df


def align_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Align all asset price series to a common date index.

    Uses an inner join (only dates where all assets have data), then
    forward-fills gaps of at most one trading day.
    """
    if prices.empty or prices.shape[1] == 0:
        return prices.copy()

    df = prices.copy()
    df = df.dropna(how="all")
    df = df.ffill(limit=1)
    df = df.dropna(how="any")
    return df


def handle_missing_values(
    prices: pd.DataFrame,
    method: str = "drop",
) -> pd.DataFrame:
    """Handle missing values in a price DataFrame.

    Parameters
    ----------
    prices : pandas.DataFrame
        Input price data.
    method : str, optional
        - ``"drop"``: drop rows containing any NaN.
        - ``"ffill"``: forward-fill, then drop any remaining NaN.
        - ``"interpolate"``: linear interpolate, then drop remaining NaN.

    Returns
    -------
    pandas.DataFrame
    """
    if method == "drop":
        return prices.dropna(how="any").copy()
    if method == "ffill":
        return prices.ffill().dropna(how="any").copy()
    if method == "interpolate":
        return prices.interpolate(method="linear").dropna(how="any").copy()
    raise ValueError(
        f"Unknown method '{method}'. Use 'drop', 'ffill', or 'interpolate'."
    )
