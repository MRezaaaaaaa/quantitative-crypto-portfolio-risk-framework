"""Financial semantics for Portfolio Path V1 Plotly figures."""

from dataclasses import replace

import pandas as pd
import pytest

pytest.importorskip("plotly")

from var_cvar_crypto_risk.portfolio_path import (  # noqa: E402
    PortfolioEvolutionPolicy,
    PortfolioPathConfig,
    build_portfolio_path,
)
from var_cvar_crypto_risk.portfolio_path_charts import (  # noqa: E402
    plot_comparative_drawdown,
    plot_gross_vs_net_nav,
    plot_hold_vs_selected_nav,
    plot_turnover_and_costs,
    plot_weights_drift_and_rebalances,
)


def _result(policy, costs=0.0):
    prices = pd.DataFrame(
        {
            "BTC": [100, 104, 102, 108, 112, 109, 116, 120, 118, 125],
            "ETH": [50, 49, 52, 51, 54, 56, 55, 58, 60, 59],
        },
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    return build_portfolio_path(
        prices,
        pd.Series({"BTC": 0.6, "ETH": 0.4}),
        PortfolioPathConfig(
            initial_capital=100_000,
            policy=policy,
            commission_bps=costs,
        ),
    )


def test_required_portfolio_path_charts_expose_correct_series_and_events():
    hold = _result(PortfolioEvolutionPolicy.BUY_AND_HOLD)
    daily = _result(PortfolioEvolutionPolicy.DAILY_REBALANCE, costs=10.0)

    comparison = plot_hold_vs_selected_nav(hold, daily, show_selected_gross=True)
    assert {trace.name for trace in comparison.data} == {
        "Buy & Hold — net NAV",
        "daily_rebalance — net NAV",
        "daily_rebalance — gross NAV",
    }
    assert comparison.layout.meta["selected"]["commission_bps"] == 10.0
    assert comparison.layout.meta["selected"]["target_weights"] == {
        "BTC": 0.6,
        "ETH": 0.4,
    }

    weights = plot_weights_drift_and_rebalances(daily)
    assert any("pre-trade" in trace.name for trace in weights.data)
    assert any("post-trade" in trace.name for trace in weights.data)
    assert len(weights.layout.shapes) >= int(daily.portfolio["rebalance_flag"].sum())
    assert weights.layout.meta["policy"] == "daily_rebalance"

    turnover = plot_turnover_and_costs(daily)
    assert {trace.name for trace in turnover.data} == {
        "Gross turnover",
        "Cumulative transaction cost",
    }
    assert "Gross traded notional" in turnover.data[0].hovertemplate

    drawdown = plot_comparative_drawdown(hold, daily)
    assert len(drawdown.data) == 2
    assert all("longest underwater" in trace.name for trace in drawdown.data)

    gross_net = plot_gross_vs_net_nav(daily)
    assert {trace.name for trace in gross_net.data} == {"Gross NAV", "Net NAV"}


def test_comparison_rejects_different_finalized_calendars():
    hold = _result(PortfolioEvolutionPolicy.BUY_AND_HOLD)
    daily = _result(PortfolioEvolutionPolicy.DAILY_REBALANCE)
    shortened = daily.portfolio.iloc[:-1].copy()
    daily = replace(daily, portfolio=shortened)
    with pytest.raises(ValueError, match="same finalized calendar"):
        plot_hold_vs_selected_nav(hold, daily)
