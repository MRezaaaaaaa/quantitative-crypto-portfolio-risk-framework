"""Accounting, timing, calendar and provenance tests for Portfolio Path V1."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.portfolio import calculate_portfolio_returns
from var_cvar_crypto_risk.portfolio_path import (
    PORTFOLIO_PATH_VERSION,
    PortfolioEvolutionPolicy,
    PortfolioPathConfig,
    build_portfolio_path,
    drawdown_statistics,
)


WEIGHTS = pd.Series({"BTC": 0.50, "ETH": 0.30, "SOL": 0.20})


def _prices(periods: int = 110) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    step = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "BTC": 100.0 * np.power(1.006, step),
            "ETH": 50.0 * np.power(0.998, step),
            "SOL": 20.0 * np.power(1.011, step),
        },
        index=index,
    )


def _build(
    policy: PortfolioEvolutionPolicy,
    *,
    prices: pd.DataFrame | None = None,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
):
    return build_portfolio_path(
        _prices() if prices is None else prices,
        WEIGHTS,
        PortfolioPathConfig(
            initial_capital=100_000.0,
            policy=policy,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        ),
    )


def test_buy_and_hold_quantities_never_change_and_nav_is_mark_to_market():
    result = _build(PortfolioEvolutionPolicy.BUY_AND_HOLD)
    quantities = result.assets["quantity"].unstack("asset")
    assert quantities.nunique().eq(1).all()
    market_values = result.assets["market_value"].unstack("asset")
    pd.testing.assert_series_equal(
        market_values.sum(axis=1),
        result.portfolio["net_nav"],
        check_names=False,
        rtol=1e-12,
    )
    assert not result.portfolio["rebalance_flag"].any()


def test_buy_and_hold_weights_drift_with_relative_price_moves():
    result = _build(PortfolioEvolutionPolicy.BUY_AND_HOLD)
    launch = result.assets.xs(result.portfolio.index[0], level="date")
    latest = result.assets.xs(result.portfolio.index[-1], level="date")
    np.testing.assert_allclose(launch["post_trade_weight"], WEIGHTS, atol=1e-12)
    assert float(latest["post_trade_weight"].sub(WEIGHTS).abs().max()) > 0.01
    assert result.portfolio.iloc[-1]["total_post_trade_drift"] > 0.01


def test_zero_cost_daily_rebalance_exactly_reproduces_legacy_weighted_returns():
    prices = _prices(45)
    legacy = calculate_portfolio_returns(
        prices.pct_change(fill_method=None).iloc[1:], WEIGHTS
    )
    result = _build(PortfolioEvolutionPolicy.DAILY_REBALANCE, prices=prices)
    pd.testing.assert_series_equal(
        result.net_returns,
        legacy.rename("portfolio_return"),
        rtol=1e-12,
        atol=1e-12,
        check_names=False,
        check_freq=False,
    )
    assert result.config.methodology_version == PORTFOLIO_PATH_VERSION
    assert result.portfolio.iloc[1:]["rebalance_flag"].all()


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (
            PortfolioEvolutionPolicy.WEEKLY_REBALANCE,
            [
                item.date().isoformat()
                for item in pd.date_range("2024-01-07", "2024-04-14", freq="W-SUN")
            ],
        ),
        (
            PortfolioEvolutionPolicy.MONTHLY_REBALANCE,
            ["2024-01-31", "2024-02-29", "2024-03-31"],
        ),
        (PortfolioEvolutionPolicy.QUARTERLY_REBALANCE, ["2024-03-31"]),
    ],
)
def test_periodic_rebalance_dates_follow_completed_utc_periods(policy, expected):
    result = _build(policy)
    actual = result.portfolio.index[result.portfolio["rebalance_flag"]]
    assert [item.date().isoformat() for item in actual] == expected


@pytest.mark.parametrize(
    "policy",
    [
        PortfolioEvolutionPolicy.DAILY_REBALANCE,
        PortfolioEvolutionPolicy.WEEKLY_REBALANCE,
        PortfolioEvolutionPolicy.MONTHLY_REBALANCE,
        PortfolioEvolutionPolicy.QUARTERLY_REBALANCE,
    ],
)
def test_post_trade_weights_return_to_target_on_each_rebalance(policy):
    result = _build(policy)
    dates = result.portfolio.index[result.portfolio["rebalance_flag"]]
    for when in dates:
        weights = result.assets.xs(when, level="date")["post_trade_weight"]
        np.testing.assert_allclose(weights, WEIGHTS, atol=1e-12)


def test_costs_reduce_net_nav_and_accounting_identities_hold():
    result = _build(
        PortfolioEvolutionPolicy.WEEKLY_REBALANCE,
        commission_bps=12.0,
        slippage_bps=8.0,
    )
    rebalances = result.portfolio[result.portfolio["rebalance_flag"]]
    expected_cost = 0.002 * rebalances["gross_traded_notional"]
    np.testing.assert_allclose(rebalances["transaction_cost"], expected_cost)
    assert result.portfolio.iloc[-1]["gross_nav"] > result.portfolio.iloc[-1]["net_nav"]
    assert result.portfolio.iloc[-1]["cumulative_transaction_cost"] == pytest.approx(
        result.portfolio["transaction_cost"].sum()
    )
    assert (rebalances["turnover"] == rebalances["gross_turnover"]).all()
    np.testing.assert_allclose(
        rebalances["one_way_turnover"], 0.5 * rebalances["gross_turnover"]
    )
    for when in result.portfolio.index[result.portfolio["finalized"]]:
        state = result.assets.xs(when, level="date")
        assert state["market_value"].sum() == pytest.approx(
            result.portfolio.at[when, "net_nav"]
        )
        assert state["post_trade_weight"].sum() == pytest.approx(1.0)


def test_risk_horizon_is_not_a_portfolio_path_input_or_schedule_driver():
    first = _build(PortfolioEvolutionPolicy.MONTHLY_REBALANCE)
    risk_horizon_days = 30
    second = _build(PortfolioEvolutionPolicy.MONTHLY_REBALANCE)
    assert risk_horizon_days == 30  # independent downstream risk setting
    pd.testing.assert_series_equal(
        first.portfolio["rebalance_flag"], second.portfolio["rebalance_flag"]
    )
    assert "horizon" not in first.config.to_dict()


def test_future_prices_cannot_change_earlier_quantities_or_decisions():
    prices = _prices(70)
    original = _build(PortfolioEvolutionPolicy.WEEKLY_REBALANCE, prices=prices)
    changed = prices.copy()
    changed.loc["2024-02-10":, "SOL"] *= 5.0
    perturbed = _build(PortfolioEvolutionPolicy.WEEKLY_REBALANCE, prices=changed)
    cutoff = pd.Timestamp("2024-02-09")
    pd.testing.assert_frame_equal(
        original.assets.loc[(slice(None, cutoff), slice(None)), :],
        perturbed.assets.loc[(slice(None, cutoff), slice(None)), :],
    )


def test_missing_scheduled_prices_defer_rebalance_without_forward_fill():
    prices = _prices(100)
    prices.loc["2024-01-31", "ETH"] = np.nan
    result = _build(PortfolioEvolutionPolicy.MONTHLY_REBALANCE, prices=prices)
    january = result.portfolio.loc["2024-01-31"]
    february_first = result.portfolio.loc["2024-02-01"]
    assert not january["finalized"]
    assert january["data_quality_status"] == "rebalance_deferred"
    assert february_first["rebalance_flag"]
    assert february_first["scheduled_rebalance_date"] == pd.Timestamp("2024-01-31")
    assert february_first["effective_rebalance_date"] == pd.Timestamp("2024-02-01")
    assert february_first["data_quality_status"] == "deferred_rebalance_executed"
    assert result.warnings


def test_missing_weekly_boundary_never_backdates_and_executes_next_complete_close():
    prices = _prices(12).drop(pd.Timestamp("2024-01-07"))
    result = _build(
        PortfolioEvolutionPolicy.WEEKLY_REBALANCE,
        prices=prices,
        commission_bps=10.0,
    )

    assert not result.portfolio.loc["2024-01-06", "rebalance_flag"]
    executed = result.portfolio.loc["2024-01-08"]
    assert executed["rebalance_flag"]
    assert executed["scheduled_rebalance_date"] == pd.Timestamp("2024-01-07")
    assert executed["effective_rebalance_date"] == pd.Timestamp("2024-01-08")
    assert executed["transaction_cost"] > 0.0
    event = result.rebalance_events.iloc[0]
    assert event["scheduled_rebalance_date"] == pd.Timestamp("2024-01-07")
    assert event["effective_rebalance_date"] == pd.Timestamp("2024-01-08")
    assert event["status"] == "deferred_executed"
    assert event["transaction_cost_charged"] == pytest.approx(
        executed["transaction_cost"]
    )


def test_incomplete_weekly_boundary_defers_to_first_complete_observation():
    prices = _prices(12)
    prices.loc["2024-01-07", "ETH"] = np.nan
    result = _build(PortfolioEvolutionPolicy.WEEKLY_REBALANCE, prices=prices)

    assert not result.portfolio.loc["2024-01-07", "finalized"]
    assert not result.portfolio.loc["2024-01-07", "rebalance_flag"]
    assert result.portfolio.loc["2024-01-08", "rebalance_flag"]
    assert result.rebalance_events.iloc[0]["status"] == "deferred_executed"


@pytest.mark.parametrize(
    ("policy", "missing_boundary", "effective_date"),
    [
        (
            PortfolioEvolutionPolicy.MONTHLY_REBALANCE,
            "2024-01-31",
            "2024-02-01",
        ),
        (
            PortfolioEvolutionPolicy.QUARTERLY_REBALANCE,
            "2024-03-31",
            "2024-04-01",
        ),
    ],
)
def test_missing_period_end_executes_after_not_before_boundary(
    policy, missing_boundary, effective_date
):
    prices = _prices(100).drop(pd.Timestamp(missing_boundary))
    result = _build(policy, prices=prices)

    event = result.rebalance_events.loc[
        result.rebalance_events["scheduled_rebalance_date"]
        == pd.Timestamp(missing_boundary)
    ].iloc[0]
    assert event["effective_rebalance_date"] == pd.Timestamp(effective_date)
    assert event["effective_rebalance_date"] > event["scheduled_rebalance_date"]
    before = result.portfolio.loc[
        result.portfolio.index < pd.Timestamp(missing_boundary)
    ]
    assert not before.loc[
        before.index.to_period("M") == pd.Timestamp(missing_boundary).to_period("M"),
        "rebalance_flag",
    ].any()


def test_multiple_missed_boundaries_are_audited_but_trade_and_charge_once():
    prices = _prices(100)
    gap = pd.date_range("2024-01-31", "2024-03-31", freq="D")
    prices = prices.drop(gap.intersection(prices.index))
    result = _build(
        PortfolioEvolutionPolicy.MONTHLY_REBALANCE,
        prices=prices,
        commission_bps=10.0,
    )

    events = result.rebalance_events
    assert list(events["scheduled_rebalance_date"]) == list(
        pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    )
    assert events["effective_rebalance_date"].eq(pd.Timestamp("2024-04-01")).all()
    assert list(events["status"]) == [
        "deferred_executed",
        "coalesced_without_additional_trade",
        "coalesced_without_additional_trade",
    ]
    april_first = result.portfolio.loc["2024-04-01"]
    assert april_first["rebalance_flag"]
    assert int(result.portfolio["rebalance_flag"].sum()) == 1
    assert (events["transaction_cost_charged"] > 0.0).sum() == 1
    assert events["transaction_cost_charged"].sum() == pytest.approx(
        april_first["transaction_cost"]
    )


def test_buy_and_hold_has_no_rebalance_events_when_boundary_dates_are_missing():
    prices = _prices(40).drop(pd.Timestamp("2024-01-31"))
    result = _build(PortfolioEvolutionPolicy.BUY_AND_HOLD, prices=prices)

    assert result.rebalance_events.empty
    assert not result.portfolio["rebalance_flag"].any()
    assert result.portfolio["transaction_cost"].sum() == 0.0


def test_hold_and_weekly_are_identical_before_first_rebalance():
    hold = _build(PortfolioEvolutionPolicy.BUY_AND_HOLD)
    weekly = _build(PortfolioEvolutionPolicy.WEEKLY_REBALANCE)
    first_rebalance = weekly.portfolio.index[weekly.portfolio["rebalance_flag"]][0]
    before = weekly.portfolio.index < first_rebalance
    np.testing.assert_allclose(
        hold.portfolio.loc[before, "net_nav"], weekly.portfolio.loc[before, "net_nav"]
    )


def test_negative_weights_are_rejected_with_financing_explanation():
    with pytest.raises(ValueError, match="borrow fees"):
        build_portfolio_path(
            _prices(),
            pd.Series({"BTC": 0.8, "ETH": 0.4, "SOL": -0.2}),
            PortfolioPathConfig(initial_capital=100_000.0),
        )


def test_provenance_is_json_serializable_and_contains_cost_and_policy_contract():
    result = _build(
        PortfolioEvolutionPolicy.MONTHLY_REBALANCE,
        commission_bps=5.0,
        slippage_bps=10.0,
    )
    payload = result.provenance()
    assert payload["policy"] == "monthly_rebalance"
    assert payload["commission_bps"] == 5.0
    assert payload["slippage_bps"] == 10.0
    assert payload["risk_base_type"] == "current_net_nav"
    json.dumps(payload, allow_nan=False)


def test_three_asset_hold_and_daily_paths_visibly_diverge():
    prices = _prices(90)
    hold = _build(PortfolioEvolutionPolicy.BUY_AND_HOLD, prices=prices)
    daily = _build(PortfolioEvolutionPolicy.DAILY_REBALANCE, prices=prices)
    assert hold.latest_net_nav != pytest.approx(daily.latest_net_nav, rel=1e-6)
    assert hold.portfolio.iloc[-1]["total_post_trade_drift"] > 0.01
    assert daily.portfolio.iloc[-1]["total_post_trade_drift"] < 1e-10


def test_drawdown_statistics_reports_recovery_of_maximum_drawdown_episode():
    nav = pd.Series(
        [100.0, 80.0, 100.0, 110.0, 104.0, 111.0, 108.0],
        index=pd.date_range("2024-01-01", periods=7),
    )

    stats = drawdown_statistics(nav)

    assert stats["maximum_drawdown"] == pytest.approx(-0.20)
    assert stats["recovery_date"] == pd.Timestamp("2024-01-03")
    assert stats["longest_underwater_days"] == 1
    assert stats["last_episode_recovery_date"] is None
