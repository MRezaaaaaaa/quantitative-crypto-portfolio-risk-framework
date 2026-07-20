"""Smoke tests for the Phase 5.5 chart helpers (figures are produced)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from var_cvar_crypto_risk.backtesting import (
    calculate_rolling_breach_rate,
    rolling_var_forecast,
)
from var_cvar_crypto_risk.correlation import (
    calculate_correlation_matrix,
    calculate_rolling_average_correlation,
)
from var_cvar_crypto_risk.risk_metrics import calculate_asset_drawdowns
from var_cvar_crypto_risk.plotting import (
    plot_asset_cumulative_returns,
    plot_asset_drawdowns,
    plot_asset_return_distributions,
    plot_correlation_heatmap,
    plot_qq_vs_normal,
    plot_rolling_average_correlation,
    plot_rolling_breach_rate,
    plot_tail_zoom_distribution,
)


def test_plot_qq_vs_normal_returns_figure(long_returns):
    fig = plot_qq_vs_normal(long_returns)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_tail_zoom_returns_figure(long_returns):
    fig = plot_tail_zoom_distribution(
        long_returns, var_value=0.04, cvar_value=0.06, confidence_level=0.95
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_asset_cumulative_returns_figure(sample_returns):
    fig = plot_asset_cumulative_returns(sample_returns)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_asset_drawdowns_figure(sample_returns):
    dd = calculate_asset_drawdowns(sample_returns)
    fig = plot_asset_drawdowns(dd)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_asset_return_distributions_figure(sample_returns):
    fig = plot_asset_return_distributions(sample_returns, horizon_days=1)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    fig2 = plot_asset_return_distributions(sample_returns, horizon_days=5)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)


def test_plot_correlation_heatmap_figure(sample_returns):
    corr = calculate_correlation_matrix(sample_returns)
    fig = plot_correlation_heatmap(corr)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_rolling_average_correlation_figure(sample_returns):
    rolling = calculate_rolling_average_correlation(sample_returns, window=20)
    fig = plot_rolling_average_correlation(rolling)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_rolling_breach_rate_figure(long_returns):
    forecast = rolling_var_forecast(long_returns, method="historical", window=100)
    rolling = calculate_rolling_breach_rate(forecast, window=50)
    fig = plot_rolling_breach_rate(rolling, expected_breach_rate=0.05, method="historical")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
