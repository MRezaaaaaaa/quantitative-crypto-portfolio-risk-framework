"""Retrospective, policy-specific summary of an audited portfolio path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .cvar_models import calculate_cvar
from .portfolio_path import PortfolioPathResult, drawdown_statistics
from .risk_conventions import loss_value_to_money
from .var_models import calculate_var


_POLICY_LABELS = {
    "buy_and_hold": "Buy & Hold",
    "daily_rebalance": "Daily rebalancing",
    "weekly_rebalance": "Weekly rebalancing",
    "monthly_rebalance": "Monthly rebalancing",
    "quarterly_rebalance": "Quarterly rebalancing",
}


@dataclass(frozen=True)
class HistoricalPricePartition:
    """Raw-source split between completed UTC dates and provisional dates."""

    finalized: pd.DataFrame
    provisional: pd.DataFrame
    today_utc: pd.Timestamp


def _normalized_utc_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (
        index.tz_convert("UTC").tz_localize(None).normalize()
        if index.tz is not None
        else index.normalize()
    )


def partition_historical_prices(
    prices: pd.DataFrame,
    *,
    now_utc: datetime | None = None,
    minimum_finalized_observations: int = 2,
) -> HistoricalPricePartition:
    """Exclude the current/future UTC day from finalized historical analysis."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must use a DatetimeIndex")
    current_utc = now_utc or datetime.now(timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    today_utc = pd.Timestamp(current_utc.astimezone(timezone.utc).date())
    normalized = _normalized_utc_dates(pd.DatetimeIndex(prices.index))
    finalized = prices.loc[normalized < today_utc].copy()
    provisional = prices.loc[normalized >= today_utc].copy()
    finalized.attrs.update(prices.attrs)
    provisional.attrs.update(prices.attrs)
    if len(finalized) < int(minimum_finalized_observations):
        raise ValueError(
            "Insufficient completed UTC history after excluding provisional "
            f"current/future rows: {len(finalized)} finalized observation(s) remain; "
            f"at least {int(minimum_finalized_observations)} are required."
        )
    return HistoricalPricePartition(
        finalized=finalized,
        provisional=provisional,
        today_utc=today_utc,
    )


@dataclass(frozen=True)
class HistoricalDataQuality:
    """Observable quality state of the price input and finalized path."""

    missing_calendar_dates: tuple[pd.Timestamp, ...]
    missing_price_rows: tuple[pd.Timestamp, ...]
    non_one_day_intervals: tuple[tuple[pd.Timestamp, pd.Timestamp, int], ...]
    duplicate_dates: tuple[pd.Timestamp, ...]
    unsorted_dates: bool
    partial_current_utc_day: bool
    provisional_current_dates: tuple[pd.Timestamp, ...]
    first_finalized_date: pd.Timestamp
    last_finalized_date: pd.Timestamp

    @property
    def daily_statistics_available(self) -> bool:
        return not (
            self.missing_price_rows
            or self.non_one_day_intervals
            or self.duplicate_dates
            or self.unsorted_dates
        )

    @property
    def is_clean(self) -> bool:
        return not (
            self.missing_calendar_dates
            or self.missing_price_rows
            or self.non_one_day_intervals
            or self.duplicate_dates
            or self.unsorted_dates
        )

    @property
    def status(self) -> str:
        return "Clean daily history" if self.is_clean else "Review required"

    def rows(self) -> pd.DataFrame:
        """Return a stable table suitable for UI display and export."""
        return pd.DataFrame(
            [
                ("Status", self.status),
                ("First finalized date", self.first_finalized_date.date().isoformat()),
                ("Last finalized date", self.last_finalized_date.date().isoformat()),
                ("Missing calendar dates", len(self.missing_calendar_dates)),
                ("Rows with missing asset prices", len(self.missing_price_rows)),
                ("Non-one-day return intervals", len(self.non_one_day_intervals)),
                ("Duplicate dates", len(self.duplicate_dates)),
                ("Unsorted dates", "Yes" if self.unsorted_dates else "No"),
                (
                    "Partial current UTC day",
                    "Yes" if self.partial_current_utc_day else "No",
                ),
                (
                    "Provisional current/future rows excluded",
                    len(self.provisional_current_dates),
                ),
            ],
            columns=["Check", "Result"],
        )

    def warnings(self) -> tuple[str, ...]:
        messages: list[str] = []
        if self.missing_calendar_dates:
            messages.append(
                f"{len(self.missing_calendar_dates)} calendar date(s) are absent "
                "from the price index."
            )
        if self.missing_price_rows:
            messages.append(
                f"{len(self.missing_price_rows)} row(s) contain at least one "
                "missing asset price."
            )
        if self.non_one_day_intervals:
            messages.append(
                f"{len(self.non_one_day_intervals)} finalized return interval(s) "
                "span more than one calendar day; daily distribution statistics "
                "are unavailable."
            )
        if self.duplicate_dates:
            messages.append(
                f"{len(self.duplicate_dates)} duplicate UTC date(s) were detected."
            )
        if self.unsorted_dates:
            messages.append("The supplied price dates are not sorted ascending.")
        if self.partial_current_utc_day:
            messages.append(
                "The source contains the current UTC day. It was retained for "
                "audit but excluded from finalized NAV, returns, drawdown and "
                "tail statistics."
            )
        return tuple(messages)


@dataclass(frozen=True)
class HistoricalRiskSummary:
    """UI-ready tables derived from one finalized portfolio path."""

    context: pd.DataFrame
    primary: pd.DataFrame
    descriptive: pd.DataFrame
    distribution_shape: pd.DataFrame
    tail_distribution: pd.DataFrame
    export: pd.DataFrame
    quality: HistoricalDataQuality


def assess_historical_data_quality(
    prices: pd.DataFrame,
    path: PortfolioPathResult,
    *,
    now_utc: datetime | None = None,
) -> HistoricalDataQuality:
    """Inspect calendar integrity without mutating or filling the input."""
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.empty:
        raise ValueError("prices must be non-empty with a DatetimeIndex")
    original = pd.DatetimeIndex(prices.index)
    normalized = _normalized_utc_dates(original)
    cleaned_duplicate_dates = tuple(
        pd.Timestamp(value)
        for value in normalized[normalized.duplicated(keep=False)].unique()
    )
    source_duplicate_dates = tuple(
        pd.Timestamp(value) for value in prices.attrs.get("source_duplicate_dates", ())
    )
    duplicate_dates = tuple(
        sorted(set(cleaned_duplicate_dates).union(source_duplicate_dates))
    )
    unsorted = bool(prices.attrs.get("source_unsorted_dates", False)) or not (
        normalized.is_monotonic_increasing
    )
    unique_sorted = pd.DatetimeIndex(normalized.unique()).sort_values()
    expected = pd.date_range(unique_sorted.min(), unique_sorted.max(), freq="D")
    missing_calendar = tuple(
        pd.Timestamp(value) for value in expected.difference(unique_sorted)
    )
    cleaned_missing_rows = tuple(
        pd.Timestamp(value)
        for value in normalized[prices.isna().any(axis=1).to_numpy()].unique()
    )
    source_missing_rows = tuple(
        pd.Timestamp(value)
        for value in prices.attrs.get("source_missing_price_rows", ())
    )
    missing_rows = tuple(sorted(set(cleaned_missing_rows).union(source_missing_rows)))

    finalized = path.portfolio.index[path.portfolio["finalized"]]
    if len(finalized) == 0:
        raise ValueError("portfolio path has no finalized observations")
    finalized = pd.DatetimeIndex(finalized)
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for previous, current in zip(finalized[:-1], finalized[1:]):
        days = int((pd.Timestamp(current) - pd.Timestamp(previous)).days)
        if days != 1:
            intervals.append((pd.Timestamp(previous), pd.Timestamp(current), days))

    current_utc = now_utc or datetime.now(timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    today_utc = pd.Timestamp(current_utc.astimezone(timezone.utc).date())
    provisional_dates = tuple(
        pd.Timestamp(value) for value in unique_sorted[unique_sorted >= today_utc]
    )
    partial_current = today_utc in provisional_dates
    if pd.Timestamp(finalized.max()).normalize() >= today_utc:
        raise ValueError(
            "portfolio path includes a provisional current/future UTC date; "
            "partition prices before building finalized historical analytics"
        )
    return HistoricalDataQuality(
        missing_calendar_dates=missing_calendar,
        missing_price_rows=missing_rows,
        non_one_day_intervals=tuple(intervals),
        duplicate_dates=duplicate_dates,
        unsorted_dates=unsorted,
        partial_current_utc_day=partial_current,
        provisional_current_dates=provisional_dates,
        first_finalized_date=pd.Timestamp(finalized[0]),
        last_finalized_date=pd.Timestamp(finalized[-1]),
    )


def _table(rows: Iterable[tuple[str, Any, str, int | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit", "Sample Size"])


def format_historical_value(value: Any, unit: str = "") -> str:
    """Return the canonical UI/export representation of one historical value."""
    if pd.isna(value):
        return "N/A"
    if unit in {"%", "% loss", "weight"}:
        return f"{float(value):.3%}"
    if unit == "USD":
        return f"${float(value):,.2f}"
    if unit == "bps":
        return f"{float(value):.2f} bps"
    if unit == "count":
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def format_historical_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Format a metric table using the authoritative display contract."""
    display = frame.copy()
    display["Value"] = [
        format_historical_value(value, str(unit))
        for value, unit in zip(display["Value"], display["Unit"])
    ]
    return display


def _export_record(
    record_type: str,
    section: str,
    name: str,
    value: Any,
    unit: str = "",
    sample_size: int | None = None,
    *,
    display_value: str | None = None,
) -> dict[str, Any]:
    return {
        "Record Type": record_type,
        "Section": section,
        "Name": name,
        "Value": value,
        "Display Value": (
            format_historical_value(value, unit)
            if display_value is None
            else display_value
        ),
        "Unit": unit,
        "Sample Size": sample_size,
    }


def _tail_label(method: str, measure: str) -> str:
    normalized = method.strip().lower()
    if normalized == "historical":
        return f"Historical Sample {measure}"
    label = {
        "gaussian": "Gaussian",
        "cornish_fisher": "Cornish-Fisher",
    }.get(normalized, normalized.replace("_", " ").title())
    return f"{label} Fit to Historical Sample {measure}"


def build_historical_risk_summary(
    *,
    path: PortfolioPathResult,
    prices: pd.DataFrame,
    price_source: str,
    return_convention: str,
    confidence_level: float,
    var_methods: list[str],
    cvar_methods: list[str],
    now_utc: datetime | None = None,
) -> HistoricalRiskSummary:
    """Build a retrospective summary without reconstructing portfolio returns."""
    quality = assess_historical_data_quality(prices, path, now_utc=now_utc)
    finalized = path.portfolio.loc[path.portfolio["finalized"]].copy()
    net_returns = path.net_returns.astype(float)
    ending_nav = float(finalized["net_nav"].iloc[-1])
    cumulative_return = ending_nav / path.config.initial_capital - 1.0
    dd = drawdown_statistics(finalized["net_nav"])
    total_cost = float(finalized["transaction_cost"].sum())
    finalized_count = int(len(finalized))
    return_count = int(len(net_returns))
    period = (
        f"{quality.first_finalized_date.date().isoformat()} to "
        f"{quality.last_finalized_date.date().isoformat()}"
    )
    path_statuses = sorted(set(finalized["data_quality_status"].astype(str)))
    context_records: list[tuple[str, Any, str]] = [
        ("Analysis type", "Historical descriptive analysis", ""),
        ("Price source", str(price_source), ""),
        (
            "Effective start date",
            quality.first_finalized_date.date().isoformat(),
            "date",
        ),
        (
            "Effective end date",
            quality.last_finalized_date.date().isoformat(),
            "date",
        ),
        ("Finalized observations", finalized_count, "count"),
        (
            "Portfolio policy",
            _POLICY_LABELS.get(path.config.policy.value, path.config.policy.value),
            "",
        ),
        ("Initial capital", path.config.initial_capital, "USD"),
        ("Commission", path.config.commission_bps, "bps"),
        ("Slippage", path.config.slippage_bps, "bps"),
        ("Return convention", return_convention, ""),
        ("Methodology version", path.config.methodology_version, ""),
        (
            "Data-quality status",
            f"{quality.status}; path rows: {', '.join(path_statuses)}",
            "",
        ),
        (
            "Provisional current/future rows excluded",
            len(quality.provisional_current_dates),
            "count",
        ),
    ]
    context_records.extend(
        (f"Target weight — {asset}", float(weight), "weight")
        for asset, weight in path.target_weights.items()
    )
    context = pd.DataFrame(
        [
            (name, format_historical_value(value, unit))
            for name, value, unit in context_records
        ],
        columns=["Field", "Value"],
    )
    primary = _table(
        [
            ("Ending Net NAV", ending_nav, "USD", finalized_count),
            ("Net Cumulative Return", cumulative_return, "%", finalized_count),
            ("Maximum Drawdown", float(dd["maximum_drawdown"]), "%", finalized_count),
            ("Total Transaction Costs", total_cost, "USD", finalized_count),
            ("Historical Period", period, "", finalized_count),
            ("Finalized Observations", finalized_count, "count", finalized_count),
        ]
    )

    daily_ok = quality.daily_statistics_available
    descriptive = _table(
        [
            (
                "Mean Daily Return",
                float(net_returns.mean()) if daily_ok and return_count else np.nan,
                "%",
                return_count,
            ),
            (
                "Daily Volatility",
                float(net_returns.std(ddof=1))
                if daily_ok and return_count >= 2
                else np.nan,
                "%",
                return_count,
            ),
            (
                "Minimum Daily Return",
                float(net_returns.min()) if daily_ok and return_count else np.nan,
                "%",
                return_count,
            ),
            (
                "Maximum Daily Return",
                float(net_returns.max()) if daily_ok and return_count else np.nan,
                "%",
                return_count,
            ),
        ]
    )
    shape = _table(
        [
            (
                "Sample Skewness",
                float(stats.skew(net_returns, bias=False))
                if daily_ok and return_count >= 3
                else np.nan,
                "",
                return_count,
            ),
            (
                "Sample Excess Kurtosis",
                float(stats.kurtosis(net_returns, fisher=True, bias=False))
                if daily_ok and return_count >= 4
                else np.nan,
                "",
                return_count,
            ),
        ]
    )

    tail_rows: list[tuple[str, Any, str, int | None]] = []
    for method in var_methods:
        value = calculate_var(net_returns, method, confidence_level)
        label = _tail_label(method, f"VaR {confidence_level:.1%}")
        tail_rows.extend(
            [
                (label, value, "% loss", return_count),
                (
                    f"Monetary equivalent at ending NAV — {label}",
                    loss_value_to_money(value, ending_nav),
                    "USD",
                    return_count,
                ),
            ]
        )
    for method in cvar_methods:
        value = calculate_cvar(net_returns, method, confidence_level)
        label = _tail_label(method, f"CVaR / Expected Shortfall {confidence_level:.1%}")
        tail_rows.extend(
            [
                (label, value, "% loss", return_count),
                (
                    f"Monetary equivalent at ending NAV — {label}",
                    loss_value_to_money(value, ending_nav),
                    "USD",
                    return_count,
                ),
            ]
        )
    tail = _table(tail_rows)

    export_records: list[dict[str, Any]] = [
        _export_record("Context", "Analysis Context", name, value, unit)
        for name, value, unit in context_records
    ]
    for section, frame in (
        ("Primary", primary),
        ("Historical Descriptive Statistics", descriptive),
        ("Distribution Shape", shape),
        ("Historical Tail Distribution", tail),
    ):
        for row in frame.itertuples(index=False):
            export_records.append(
                _export_record(
                    "Metric",
                    section,
                    str(row.Metric),
                    row.Value,
                    str(row.Unit),
                    row._3,
                )
            )
    for row in quality.rows().itertuples(index=False):
        export_records.append(
            _export_record(
                "Data Quality",
                "Data Quality Audit",
                str(row.Check),
                row.Result,
                display_value=str(row.Result),
            )
        )
    for missing_date in quality.missing_calendar_dates:
        iso_date = missing_date.date().isoformat()
        export_records.append(
            _export_record(
                "Data Quality Detail",
                "Missing Calendar Dates",
                "Missing calendar date",
                iso_date,
                "date",
                display_value=iso_date,
            )
        )
    for missing_date in quality.missing_price_rows:
        iso_date = missing_date.date().isoformat()
        export_records.append(
            _export_record(
                "Data Quality Detail",
                "Missing Price Rows",
                "Row with missing asset price",
                iso_date,
                "date",
                display_value=iso_date,
            )
        )
    for previous, current, days in quality.non_one_day_intervals:
        label = f"{previous.date().isoformat()} → {current.date().isoformat()}"
        export_records.append(
            _export_record(
                "Data Quality Detail",
                "Non-One-Day Intervals",
                label,
                days,
                "calendar days",
                display_value=f"{days} calendar days",
            )
        )
    for provisional_date in quality.provisional_current_dates:
        iso_date = provisional_date.date().isoformat()
        export_records.append(
            _export_record(
                "Data Quality Detail",
                "Provisional Dates",
                "Excluded provisional UTC date",
                iso_date,
                "date",
                display_value=iso_date,
            )
        )
    for event in path.rebalance_events.itertuples(index=False):
        scheduled = pd.Timestamp(event.scheduled_rebalance_date).date().isoformat()
        effective = (
            "N/A"
            if pd.isna(event.effective_rebalance_date)
            else pd.Timestamp(event.effective_rebalance_date).date().isoformat()
        )
        export_records.append(
            _export_record(
                "Rebalance Audit",
                "Rebalance Events",
                f"Scheduled {scheduled}",
                effective if effective != "N/A" else np.nan,
                "date",
                display_value=f"{event.status}; effective {effective}",
            )
        )
    export = pd.DataFrame(export_records)
    return HistoricalRiskSummary(
        context=context,
        primary=primary,
        descriptive=descriptive,
        distribution_shape=shape,
        tail_distribution=tail,
        export=export,
        quality=quality,
    )


__all__ = [
    "HistoricalDataQuality",
    "HistoricalPricePartition",
    "HistoricalRiskSummary",
    "assess_historical_data_quality",
    "build_historical_risk_summary",
    "format_historical_table",
    "format_historical_value",
    "partition_historical_prices",
]
