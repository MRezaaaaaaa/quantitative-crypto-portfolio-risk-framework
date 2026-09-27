"""Contract tests for Streamlit-independent Plotly Risk Lab charts."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from var_cvar_crypto_risk import plotting
from var_cvar_crypto_risk.backtesting import (
    calculate_rolling_breach_rate,
    rolling_var_forecast,
)
from var_cvar_crypto_risk.correlation import (
    calculate_correlation_matrix,
    calculate_rolling_average_correlation,
)
from var_cvar_crypto_risk.risk_metrics import calculate_asset_drawdowns


def _assert_plotly(figure: go.Figure) -> None:
    assert isinstance(figure, go.Figure)
    assert figure.layout.template is not None


def test_chart_module_has_no_streamlit_dependency() -> None:
    source = inspect.getsource(plotting)
    assert "import streamlit" not in source
    assert "st." not in source


def test_risk_lab_has_no_matplotlib_renderer() -> None:
    app_source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert "st.pyplot" not in app_source
    assert "import matplotlib" not in app_source


def test_distribution_has_observed_tail_and_reference_lines(long_returns) -> None:
    figure = plotting.plot_return_distribution_with_var_cvar(
        long_returns,
        var_value=0.04,
        cvar_value=0.06,
        provenance={"policy": "buy_and_hold"},
    )
    _assert_plotly(figure)
    assert {trace.name for trace in figure.data} >= {
        "Observed historical returns",
        "Observed tail (worst 5.0%)",
    }
    assert len(figure.layout.shapes) >= 2
    assert figure.layout.meta["policy"] == "buy_and_hold"
    assert figure.layout.xaxis.tickformat == ".1%"
    observed = next(
        trace for trace in figure.data if trace.name == "Observed historical returns"
    )
    tail = next(
        trace for trace in figure.data if trace.name == "Observed tail (worst 5.0%)"
    )
    assert observed.bingroup == tail.bingroup == "historical-return-distribution"
    assert observed.nbinsx == tail.nbinsx == 60


def test_qq_plot_is_plotly(long_returns) -> None:
    qq = plotting.plot_qq_vs_normal(long_returns)
    _assert_plotly(qq)
    assert [trace.name for trace in qq.data] == [
        "Observed sample quantiles",
        "Normal reference",
    ]


def test_distribution_has_no_standalone_tail_zoom_surface() -> None:
    app_source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert "Left-tail zoom" not in app_source
    assert "plot_tail_zoom_distribution" not in app_source
    assert "plot_tail_zoom" not in app_source
    assert not hasattr(plotting, "plot_tail_zoom_distribution")
    assert "plot_tail_zoom_distribution" not in plotting.__all__
    assert "scrollZoom" not in app_source


def test_asset_growth_drawdown_and_distributions_are_plotly(sample_returns) -> None:
    drawdowns = calculate_asset_drawdowns(sample_returns)
    figures = [
        plotting.plot_asset_cumulative_returns(sample_returns),
        plotting.plot_asset_drawdowns(drawdowns),
        plotting.plot_asset_return_distributions(sample_returns),
    ]
    for figure in figures:
        _assert_plotly(figure)
    assert list(figures[0].data[0].x) == list(sample_returns.index)
    assert figures[1].layout.yaxis.tickformat == ".1%"


def test_correlation_charts_fix_scale_and_preserve_dates(sample_returns) -> None:
    correlation = calculate_correlation_matrix(sample_returns)
    rolling = calculate_rolling_average_correlation(sample_returns, window=20)
    heatmap = plotting.plot_correlation_heatmap(correlation)
    rolling_figure = plotting.plot_rolling_average_correlation(
        rolling, method="pearson", window=20
    )
    _assert_plotly(heatmap)
    _assert_plotly(rolling_figure)
    assert heatmap.data[0].zmin == -1
    assert heatmap.data[0].zmax == 1
    assert heatmap.data[0].texttemplate == "%{text:.2f}"
    assert list(rolling_figure.data[0].x) == list(rolling.index)
    assert rolling_figure.layout.meta["method"] == "pearson"
    assert rolling_figure.layout.meta["window"] == 20


def test_backtest_charts_preserve_alignment_and_breach_labels(long_returns) -> None:
    forecast = rolling_var_forecast(long_returns, method="historical", window=100)
    rate = calculate_rolling_breach_rate(forecast, window=50)
    figures = [
        plotting.plot_var_backtest(forecast, "historical", 0.95),
        plotting.plot_breach_timeline(forecast, "historical"),
        plotting.plot_rolling_breach_rate(rate, 0.05, "historical"),
    ]
    for figure in figures:
        _assert_plotly(figure)
    assert list(figures[0].data[0].x) == list(forecast.index)
    assert "Observed return" in {trace.name for trace in figures[0].data}
    assert any("Historical VaR threshold" in trace.name for trace in figures[0].data)
    assert len(figures[2].layout.shapes) == 1


def test_simulation_and_optimization_charts_return_plotly_figures() -> None:
    scenarios = pd.Series(np.linspace(-0.1, 0.1, 500))
    paths = pd.DataFrame(
        {
            "p1": [100.0, 101.0, 99.0],
            "p2": [100.0, 98.0, 102.0],
        }
    )
    comparison = pd.DataFrame(
        {"Method": ["Historical", "Normal"], "VaR": [0.04, 0.05], "CVaR": [0.06, 0.07]}
    )
    portfolio_comparison = pd.DataFrame(
        {
            "Portfolio": ["Current", "Minimum CVaR"],
            "Expected Return": [0.01, 0.008],
            "Volatility": [0.04, 0.03],
            "VaR": [0.05, 0.04],
            "CVaR": [0.07, 0.055],
        }
    )
    frontier = pd.DataFrame({"expected_return": [0.01, 0.02], "CVaR": [0.04, 0.07]})
    backtest_comparison = pd.DataFrame(
        {
            "method": ["historical", "gaussian"],
            "actual_breaches": [4, 6],
            "expected_breaches": [5.0, 5.0],
            "traffic_light": ["Green", "Yellow"],
            "error": [None, None],
        }
    )
    figures = [
        plotting.plot_mc_loss_distribution(scenarios, 0.05, 0.07),
        plotting.plot_mc_portfolio_paths(paths),
        plotting.plot_normal_vs_student_t_distribution(scenarios, scenarios * 1.2),
        plotting.plot_var_cvar_method_comparison(comparison),
        plotting.plot_optimized_weights(pd.Series({"BTC": 0.6, "ETH": 0.4})),
        plotting.plot_portfolio_comparison(portfolio_comparison),
        plotting.plot_allocation_comparison(
            {
                "Current": pd.Series({"BTC": 0.5, "ETH": 0.5}),
                "Optimized": pd.Series({"BTC": 0.6, "ETH": 0.4}),
            }
        ),
        plotting.plot_cvar_efficient_frontier(frontier),
        plotting.plot_model_comparison_backtest(backtest_comparison),
    ]
    for figure in figures:
        _assert_plotly(figure)


def test_empty_inputs_return_useful_plotly_empty_state() -> None:
    figure = plotting.plot_asset_drawdowns(pd.DataFrame())
    _assert_plotly(figure)
    assert figure.layout.annotations
    assert "No asset drawdowns" in figure.layout.annotations[0].text
