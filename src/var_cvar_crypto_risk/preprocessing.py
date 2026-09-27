"""Price-data preprocessing utilities."""

from __future__ import annotations

import pandas as pd


def clean_price_data(
    prices: pd.DataFrame, *, preserve_missing: bool = False
) -> pd.DataFrame:
    """Sort/deduplicate prices and optionally retain auditable missing values.

    Returns a clean copy. Does not modify ``prices`` in place.

    ``preserve_missing=True`` is required by stateful portfolio paths: a missing
    scheduled close must remain visible so rebalancing can be deferred rather
    than silently valued from a forward-filled price.
    """
    df = prices.copy()
    source_index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    source_duplicate_dates = tuple(
        pd.Timestamp(value)
        for value in source_index[source_index.duplicated(keep=False)].unique()
    )
    source_unsorted_dates = not source_index.is_monotonic_increasing
    source_missing_price_rows = tuple(
        pd.Timestamp(value)
        for value in source_index[df.isna().any(axis=1).to_numpy()].unique()
    )
    df.index = source_index
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if preserve_missing:
        df = df.dropna(how="all")
    else:
        df = align_price_data(df)
    # Preserve observable source defects after the normalization required by
    # downstream accounting.  The audited path consumes the clean frame, while
    # the retrospective summary can still disclose what arrived from the vendor.
    df.attrs["source_duplicate_dates"] = source_duplicate_dates
    df.attrs["source_unsorted_dates"] = source_unsorted_dates
    df.attrs["source_missing_price_rows"] = source_missing_price_rows
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
