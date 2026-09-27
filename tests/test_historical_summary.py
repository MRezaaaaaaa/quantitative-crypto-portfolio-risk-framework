"""Retrospective Risk Summary and data-quality contract tests."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.historical_summary import (
    build_historical_risk_summary,
    format_historical_table,
    partition_historical_prices,
)
from var_cvar_crypto_risk.portfolio_path import (
    PortfolioEvolutionPolicy,
    PortfolioPathConfig,
    build_portfolio_path,
)
from var_cvar_crypto_risk.preprocessing import clean_price_data


WEIGHTS = pd.Series({"BTC": 0.6, "ETH": 0.4})
FIXED_NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _prices(periods: int = 12) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    step = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "BTC": 100.0 * np.power(1.01, step),
            "ETH": 50.0 * np.power(0.995, step),
        },
        index=index,
    )


def _summary(
    prices: pd.DataFrame,
    *,
    policy: PortfolioEvolutionPolicy = PortfolioEvolutionPolicy.BUY_AND_HOLD,
    commission_bps: float = 0.0,
):
    path = build_portfolio_path(
        prices,
        WEIGHTS,
        PortfolioPathConfig(
            initial_capital=100_000.0,
            policy=policy,
            commission_bps=commission_bps,
        ),
    )
    summary = build_historical_risk_summary(
        path=path,
        prices=prices,
        price_source="synthetic",
        return_convention="Simple close-to-close net portfolio returns",
        confidence_level=0.95,
        var_methods=["historical", "gaussian"],
        cvar_methods=["historical"],
        now_utc=FIXED_NOW,
    )
    return path, summary


def test_required_metrics_are_present_and_removed_metrics_are_absent() -> None:
    _, summary = _summary(_prices())
    metric_rows = summary.export[summary.export["Record Type"] == "Metric"]
    metrics = set(metric_rows["Name"])
    assert {
        "Ending Net NAV",
        "Net Cumulative Return",
        "Maximum Drawdown",
        "Total Transaction Costs",
        "Historical Period",
        "Finalized Observations",
        "Mean Daily Return",
        "Daily Volatility",
        "Minimum Daily Return",
        "Maximum Daily Return",
        "Sample Skewness",
        "Sample Excess Kurtosis",
    }.issubset(metrics)
    assert "Annualized Return" not in metrics
    assert "Annualized Volatility" not in metrics
    assert not any("Sharpe" in metric for metric in metrics)
    assert list(metric_rows["Name"]).count("Maximum Drawdown") == 1


def test_headlines_use_final_audited_net_nav_and_initial_capital() -> None:
    path, summary = _summary(_prices())
    primary = summary.primary.set_index("Metric")["Value"]
    ending = float(path.portfolio.loc[path.portfolio["finalized"], "net_nav"].iloc[-1])
    assert primary["Ending Net NAV"] == pytest.approx(ending)
    assert primary["Net Cumulative Return"] == pytest.approx(ending / 100_000.0 - 1.0)


def test_first_period_loss_is_included_in_maximum_drawdown() -> None:
    prices = pd.DataFrame(
        {"BTC": [100.0, 80.0, 90.0], "ETH": [100.0, 80.0, 90.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    _, summary = _summary(prices)
    primary = summary.primary.set_index("Metric")["Value"]
    assert primary["Maximum Drawdown"] == pytest.approx(-0.20)


def test_policy_and_costs_flow_into_historical_summary() -> None:
    prices = _prices(40)
    hold_path, hold_summary = _summary(prices)
    weekly_path, weekly_summary = _summary(
        prices,
        policy=PortfolioEvolutionPolicy.WEEKLY_REBALANCE,
        commission_bps=25.0,
    )
    hold_primary = hold_summary.primary.set_index("Metric")["Value"]
    weekly_primary = weekly_summary.primary.set_index("Metric")["Value"]
    assert hold_primary["Total Transaction Costs"] == 0.0
    assert weekly_primary["Total Transaction Costs"] > 0.0
    assert hold_path.latest_net_nav != pytest.approx(weekly_path.latest_net_nav)
    assert hold_primary["Ending Net NAV"] == pytest.approx(hold_path.latest_net_nav)
    assert weekly_primary["Ending Net NAV"] == pytest.approx(weekly_path.latest_net_nav)


def test_insufficient_shape_samples_are_na_not_zero() -> None:
    _, summary = _summary(_prices(3))
    shape = summary.distribution_shape.set_index("Metric")["Value"]
    assert np.isnan(shape["Sample Skewness"])
    assert np.isnan(shape["Sample Excess Kurtosis"])


def test_multi_day_gap_disables_daily_labels_without_changing_nav() -> None:
    prices = _prices(8).drop(pd.Timestamp("2024-01-04"))
    path, summary = _summary(prices)
    assert summary.quality.non_one_day_intervals
    assert not summary.quality.daily_statistics_available
    assert summary.descriptive["Value"].isna().all()
    assert summary.distribution_shape["Value"].isna().all()
    assert path.latest_net_nav > 0.0
    assert any(
        "daily distribution statistics are unavailable" in warning
        for warning in summary.quality.warnings()
    )


def test_cleaning_preserves_source_defects_for_summary_disclosure() -> None:
    raw = _prices(8).iloc[[1, 0, 2, 3, 3, 4, 5, 6, 7]].copy()
    raw.iloc[2, 0] = np.nan
    cleaned = clean_price_data(raw, preserve_missing=True)
    _, summary = _summary(cleaned)
    assert summary.quality.unsorted_dates
    assert summary.quality.duplicate_dates == (pd.Timestamp("2024-01-04"),)
    assert pd.Timestamp("2024-01-03") in summary.quality.missing_price_rows
    assert not summary.quality.daily_statistics_available


def test_export_is_exact_union_of_visible_summary_sections() -> None:
    _, summary = _summary(_prices())
    expected_rows = sum(
        len(frame)
        for frame in (
            summary.primary,
            summary.descriptive,
            summary.distribution_shape,
            summary.tail_distribution,
        )
    )
    metric_export = summary.export[summary.export["Record Type"] == "Metric"]
    assert len(metric_export) == expected_rows
    assert set(metric_export["Section"]) == {
        "Primary",
        "Historical Descriptive Statistics",
        "Distribution Shape",
        "Historical Tail Distribution",
    }
    assert list(summary.export.columns) == [
        "Record Type",
        "Section",
        "Name",
        "Value",
        "Display Value",
        "Unit",
        "Sample Size",
    ]


def test_export_contains_context_quality_raw_numbers_and_ui_display_values() -> None:
    _, summary = _summary(_prices())
    export = summary.export
    names = set(export["Name"])
    assert {
        "Price source",
        "Portfolio policy",
        "Initial capital",
        "Commission",
        "Slippage",
        "Methodology version",
        "Target weight — BTC",
        "Target weight — ETH",
        "Status",
    }.issubset(names)

    ending = export.loc[export["Name"] == "Ending Net NAV"].iloc[0]
    assert isinstance(ending["Value"], (float, np.floating))
    assert (
        ending["Display Value"]
        == format_historical_table(summary.primary)
        .loc[summary.primary["Metric"] == "Ending Net NAV", "Value"]
        .iloc[0]
    )
    target = export.loc[export["Name"] == "Target weight — BTC"].iloc[0]
    assert target["Value"] == pytest.approx(0.6)
    assert target["Display Value"] == "60.000%"


def test_unavailable_values_remain_na_and_gap_details_survive_export() -> None:
    prices = _prices(8).drop(pd.Timestamp("2024-01-04"))
    _, summary = _summary(prices)
    export = summary.export
    mean_row = export.loc[export["Name"] == "Mean Daily Return"].iloc[0]
    assert pd.isna(mean_row["Value"])
    assert mean_row["Display Value"] == "N/A"
    interval = export.loc[export["Section"] == "Non-One-Day Intervals"].iloc[0]
    assert interval["Value"] == 2
    assert "2024-01-03" in interval["Name"]
    assert "2024-01-05" in interval["Name"]


@pytest.mark.parametrize("timezone_name", [None, "UTC", "Asia/Tehran"])
def test_current_utc_date_is_partitioned_from_finalized_history(timezone_name) -> None:
    now = datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc)
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    if timezone_name is not None:
        index = index.tz_localize("UTC").tz_convert(timezone_name)
    raw = pd.DataFrame(
        {"BTC": [100.0, 80.0, 160.0], "ETH": [100.0, 80.0, 160.0]},
        index=index,
    )
    partition = partition_historical_prices(
        raw, now_utc=now, minimum_finalized_observations=2
    )
    assert len(partition.finalized) == 2
    assert len(partition.provisional) == 1

    path = build_portfolio_path(
        partition.finalized,
        WEIGHTS,
        PortfolioPathConfig(initial_capital=100_000.0),
    )
    summary = build_historical_risk_summary(
        path=path,
        prices=raw,
        price_source="synthetic",
        return_convention="Simple close-to-close net portfolio returns",
        confidence_level=0.95,
        var_methods=["historical"],
        cvar_methods=["historical"],
        now_utc=now,
    )
    primary = summary.primary.set_index("Metric")["Value"]
    assert path.latest_date == pd.Timestamp("2026-01-02")
    assert primary["Ending Net NAV"] == pytest.approx(80_000.0)
    assert summary.quality.partial_current_utc_day
    assert summary.quality.daily_statistics_available
    assert summary.tail_distribution["Sample Size"].eq(1).all()
    assert "2026-01-03" in set(
        summary.export.loc[summary.export["Section"] == "Provisional Dates", "Value"]
    )


def test_summary_rejects_a_path_that_finalizes_the_current_utc_date() -> None:
    now = datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc)
    raw = pd.DataFrame(
        {"BTC": [100.0, 101.0, 102.0], "ETH": [100.0, 101.0, 102.0]},
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )
    path = build_portfolio_path(
        raw, WEIGHTS, PortfolioPathConfig(initial_capital=100_000.0)
    )
    with pytest.raises(ValueError, match="provisional current/future UTC date"):
        build_historical_risk_summary(
            path=path,
            prices=raw,
            price_source="synthetic",
            return_convention="simple",
            confidence_level=0.95,
            var_methods=["historical"],
            cvar_methods=["historical"],
            now_utc=now,
        )


def test_historical_dataset_before_today_is_unchanged_by_partition() -> None:
    prices = _prices()
    partition = partition_historical_prices(
        prices, now_utc=FIXED_NOW, minimum_finalized_observations=2
    )
    pd.testing.assert_frame_equal(partition.finalized, prices)
    assert partition.provisional.empty
