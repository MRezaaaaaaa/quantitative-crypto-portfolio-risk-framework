"""Plotly charts for persisted portfolio-monitoring read models.

Chart functions only select and present chart-ready columns.  They never fetch
data or recompute portfolio valuation, risk forecasts, or optimization output.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_PALETTE = (
    "#2563EB",
    "#F59E0B",
    "#10B981",
    "#8B5CF6",
    "#EF4444",
    "#06B6D4",
    "#EC4899",
    "#84CC16",
    "#F97316",
    "#6366F1",
)


def stable_asset_color(asset: str) -> str:
    """Return a deterministic color across processes and chart reruns."""
    normalized = asset.strip().upper()
    if normalized == "CASH":
        return "#94A3B8"
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return _PALETTE[int.from_bytes(digest[:2], "big") % len(_PALETTE)]


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=0.5, y=0.5)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(template="plotly_white", height=360)
    return figure


def _layout(figure: go.Figure, *, title: str, y_title: str | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        title=title,
        hovermode="x unified",
        legend_title_text="",
        margin=dict(l=50, r=30, t=60, b=45),
    )
    if y_title:
        figure.update_yaxes(title_text=y_title)
    return figure


def _add_boundary(figure: go.Figure, boundary) -> None:
    if boundary is None:
        return
    figure.add_vline(x=pd.Timestamp(boundary).timestamp() * 1000, line_dash="dash", line_color="#475569")
    figure.add_annotation(
        x=pd.Timestamp(boundary),
        y=1,
        yref="paper",
        text="Historical / live boundary",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="#475569", size=11),
    )


def nav_chart(
    portfolio: pd.DataFrame,
    *,
    unit: str,
    historical_boundary=None,
) -> go.Figure:
    """Plot persisted portfolio and benchmark NAV in currency or Base-100 units."""
    complete = portfolio[portfolio["finalized"]]
    if complete.empty:
        return _empty_figure("No finalized NAV observations")
    if unit not in {"currency", "base_100"}:
        raise ValueError("NAV unit must be currency or base_100")
    portfolio_column = "nav" if unit == "currency" else "base_100_nav"
    benchmark_column = "benchmark_nav" if unit == "currency" else "benchmark_base_100"
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=complete["date"],
            y=complete[portfolio_column],
            mode="lines",
            name="Portfolio",
            line=dict(color="#2563EB", width=3),
        )
    )
    if complete[benchmark_column].notna().any():
        figure.add_trace(
            go.Scatter(
                x=complete["date"],
                y=complete[benchmark_column],
                mode="lines",
                name="Benchmark",
                line=dict(color="#64748B", dash="dot", width=2),
            )
        )
    _add_boundary(figure, historical_boundary)
    return _layout(
        figure,
        title="Portfolio NAV and benchmark",
        y_title="Portfolio value" if unit == "currency" else "Base-100 index",
    )


def allocation_chart(
    allocation: pd.DataFrame, *, historical_boundary=None
) -> go.Figure:
    """Render persisted current weights as a stable-color 100% stack."""
    complete = allocation[
        allocation["finalized"] & allocation["current_weight"].notna()
    ]
    if complete.empty:
        return _empty_figure("No finalized allocation observations")
    figure = go.Figure()
    assets = sorted(complete["asset"].unique(), key=lambda item: (item == "CASH", item))
    for asset in assets:
        rows = complete[complete["asset"] == asset].sort_values("date")
        custom = rows[
            ["market_value", "quantity", "price", "current_weight"]
        ].to_numpy()
        figure.add_trace(
            go.Scatter(
                x=rows["date"],
                y=rows["current_weight"],
                name=asset,
                mode="lines",
                line=dict(width=0.5, color=stable_asset_color(asset)),
                stackgroup="one",
                groupnorm="percent",
                customdata=custom,
                hovertemplate=(
                    "Date=%{x|%Y-%m-%d}<br>Asset=" + asset
                    + "<br>Weight=%{customdata[3]:.2%}<br>Market value=%{customdata[0]:,.2f}"
                    + "<br>Fixed quantity=%{customdata[1]:,.8f}"
                    + "<br>Price=%{customdata[2]:,.4f}<extra></extra>"
                ),
            )
        )
    _add_boundary(figure, historical_boundary)
    figure.update_yaxes(range=[0, 100], ticksuffix="%")
    return _layout(
        figure,
        title="Asset allocation through time — fixed holdings, no rebalancing",
        y_title="Current portfolio weight",
    )


def target_current_chart(allocation: pd.DataFrame) -> go.Figure:
    """Compare immutable target weights with the latest persisted current weights."""
    complete = allocation[
        allocation["finalized"] & allocation["current_weight"].notna()
    ]
    if complete.empty:
        return _empty_figure("No finalized target/current allocation")
    latest = complete[complete["date"] == complete["date"].max()].sort_values("asset")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=latest["asset"],
            x=latest["target_weight"] * 100.0,
            name="Target",
            orientation="h",
            marker_color="#94A3B8",
        )
    )
    figure.add_trace(
        go.Bar(
            y=latest["asset"],
            x=latest["current_weight"] * 100.0,
            name="Current",
            orientation="h",
            marker_color=[stable_asset_color(item) for item in latest["asset"]],
            customdata=(latest["drift_percentage_points"] * 100.0).to_numpy(),
            hovertemplate=(
                "Asset=%{y}<br>Current=%{x:.2f}%"
                "<br>Drift=%{customdata:+.2f} pp<extra></extra>"
            ),
        )
    )
    figure.update_layout(barmode="group")
    return _layout(figure, title="Target versus current weights", y_title=None)


def drift_chart(allocation: pd.DataFrame, portfolio: pd.DataFrame) -> go.Figure:
    """Show persisted per-asset drift and total portfolio drift."""
    complete = allocation[
        allocation["finalized"] & allocation["drift_percentage_points"].notna()
    ]
    portfolio_complete = portfolio[portfolio["finalized"] & portfolio["total_drift"].notna()]
    if complete.empty:
        return _empty_figure("No finalized drift observations")
    matrix = complete.pivot(index="asset", columns="date", values="drift_percentage_points") * 100.0
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
        subplot_titles=("Asset drift (percentage points)", "Total drift"),
    )
    figure.add_trace(
        go.Heatmap(
            x=matrix.columns,
            y=matrix.index,
            z=matrix.to_numpy(),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="pp"),
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Asset=%{y}<br>Drift=%{z:+.2f} pp<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=portfolio_complete["date"],
            y=portfolio_complete["total_drift"] * 100.0,
            name="Total drift",
            line=dict(color="#7C3AED", width=2),
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Total drift=%{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(template="plotly_white", height=620, title="Weight drift")
    figure.update_yaxes(title_text="Percent", ticksuffix="%", row=2, col=1)
    return figure


def drawdown_chart(portfolio: pd.DataFrame, *, historical_boundary=None) -> go.Figure:
    """Render persisted drawdown as an underwater area."""
    complete = portfolio[portfolio["finalized"] & portfolio["drawdown"].notna()]
    if complete.empty:
        return _empty_figure("No finalized drawdown observations")
    figure = go.Figure(
        go.Scatter(
            x=complete["date"],
            y=complete["drawdown"] * 100.0,
            fill="tozeroy",
            line=dict(color="#DC2626", width=1.5),
            name="Drawdown",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Drawdown=%{y:.2f}%<extra></extra>",
        )
    )
    _add_boundary(figure, historical_boundary)
    return _layout(figure, title="Portfolio drawdown", y_title="Drawdown (%)")


def risk_history_chart(risk: pd.DataFrame) -> go.Figure:
    """Plot persisted horizon-aligned loss, VaR, and CVaR histories."""
    available = risk[risk["forecast_var"].notna()]
    if available.empty:
        return _empty_figure("No persisted risk forecasts")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=available["target_date"],
            y=available["forecast_var"] * 100.0,
            name="Forecast VaR",
            line=dict(color="#F59E0B", width=2),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=available["target_date"],
            y=available["forecast_cvar"] * 100.0,
            name="Forecast CVaR / ES",
            line=dict(color="#7C3AED", width=2),
        )
    )
    evaluated = available[available["realized_loss"].notna()]
    if not evaluated.empty:
        figure.add_trace(
            go.Scatter(
                x=evaluated["target_date"],
                y=evaluated["realized_loss"] * 100.0,
                name="Realized horizon loss",
                line=dict(color="#334155", width=1.5),
            )
        )
    return _layout(
        figure,
        title="Horizon-aligned VaR, CVaR, and realized loss",
        y_title="Loss (%)",
    )


def breach_timeline_chart(risk: pd.DataFrame) -> go.Figure:
    """Mark only persisted VaR exceptions; CVaR is never a breach threshold."""
    evaluated = risk[risk["realized_loss"].notna() & risk["forecast_var"].notna()]
    if evaluated.empty:
        return _empty_figure("No evaluated VaR forecasts")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=evaluated["target_date"],
            y=evaluated["forecast_var"] * 100.0,
            name="VaR threshold",
            line=dict(color="#F59E0B", width=2),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=evaluated["target_date"],
            y=evaluated["realized_loss"] * 100.0,
            name="Realized loss",
            mode="lines+markers",
            line=dict(color="#475569", width=1),
            marker=dict(color="#475569", size=5),
        )
    )
    breaches = evaluated[evaluated["var_breach"] == True]  # noqa: E712
    if not breaches.empty:
        figure.add_trace(
            go.Scatter(
                x=breaches["target_date"],
                y=breaches["realized_loss"] * 100.0,
                name="VaR exception",
                mode="markers",
                marker=dict(color="#DC2626", size=10, symbol="x"),
            )
        )
    return _layout(
        figure,
        title="VaR exception timeline — CVaR is not an exception threshold",
        y_title="Loss (%)",
    )


def forecast_realized_chart(risk: pd.DataFrame) -> go.Figure:
    """Compare point forecasts with outcomes without fabricating a path fan."""
    evaluated = risk[risk["realized_loss"].notna() & risk["forecast_var"].notna()]
    if evaluated.empty:
        return _empty_figure("Forecast path unavailable · no matured point forecasts")
    figure = go.Figure()
    for column, label, color in (
        ("forecast_var", "Forecast VaR", "#F59E0B"),
        ("forecast_cvar", "Forecast CVaR / ES", "#7C3AED"),
        ("realized_loss", "Realized loss", "#334155"),
    ):
        figure.add_trace(
            go.Bar(
                x=evaluated["target_date"],
                y=evaluated[column] * 100.0,
                name=label,
                marker_color=color,
            )
        )
    figure.update_layout(barmode="group")
    figure.add_annotation(
        text="Forecast path unavailable — point forecasts only",
        xref="paper",
        yref="paper",
        x=1,
        y=1.12,
        showarrow=False,
        xanchor="right",
        font=dict(color="#64748B", size=11),
    )
    return _layout(
        figure,
        title="Forecast versus realized horizon loss",
        y_title="Loss (%)",
    )


def comparison_nav_chart(nav: pd.DataFrame, *, alignment: str) -> go.Figure:
    """Plot explicitly aligned Base-100 experiment paths."""
    if nav.empty:
        return _empty_figure("No overlapping comparison observations")
    figure = go.Figure()
    for column in nav.columns:
        figure.add_trace(
            go.Scatter(
                x=nav.index,
                y=nav[column],
                name=str(column),
                mode="lines",
                line=dict(width=2),
            )
        )
    x_title = "Date" if alignment == "common_calendar" else "Days since launch"
    figure.update_xaxes(title_text=x_title)
    return _layout(figure, title="Experiment comparison — Base-100", y_title="Base-100 NAV")


def comparison_scatter_chart(summary: pd.DataFrame) -> go.Figure:
    """Plot persisted realized volatility versus cumulative return."""
    available = summary.dropna(subset=["realized_volatility", "cumulative_return"])
    if available.empty:
        return _empty_figure("No comparable persisted risk/return metrics")
    figure = go.Figure(
        go.Scatter(
            x=available["realized_volatility"] * 100.0,
            y=available["cumulative_return"] * 100.0,
            text=available["name"],
            customdata=available[["mode", "observations"]].to_numpy(),
            mode="markers+text",
            textposition="top center",
            marker=dict(size=12, color="#2563EB"),
            hovertemplate=(
                "%{text}<br>Mode=%{customdata[0]}<br>Observations=%{customdata[1]}"
                "<br>Realized volatility=%{x:.2f}%<br>Cumulative return=%{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title_text="Latest persisted realized volatility (%)")
    figure.update_yaxes(title_text="Cumulative return (%)")
    return _layout(figure, title="Persisted risk/return comparison")


__all__ = [
    "allocation_chart",
    "breach_timeline_chart",
    "comparison_nav_chart",
    "comparison_scatter_chart",
    "drawdown_chart",
    "drift_chart",
    "forecast_realized_chart",
    "nav_chart",
    "risk_history_chart",
    "stable_asset_color",
    "target_current_chart",
]
