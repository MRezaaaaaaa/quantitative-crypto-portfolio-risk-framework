"""Interactive Plotly charts for audited portfolio-path results."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plotly_theme import COLORS, PALETTE, apply_risk_theme
from .portfolio_path import PortfolioPathResult, drawdown_statistics


def _complete(result: PortfolioPathResult) -> pd.DataFrame:
    return result.portfolio[result.portfolio["finalized"]].copy()


def _context(result: PortfolioPathResult) -> str:
    return (
        f"{result.config.policy.value}; commission "
        f"{result.config.commission_bps:g} bps; slippage "
        f"{result.config.slippage_bps:g} bps; "
        f"{result.config.methodology_version}"
    )


def plot_hold_vs_selected_nav(
    hold: PortfolioPathResult,
    selected: PortfolioPathResult,
    *,
    show_selected_gross: bool = False,
) -> go.Figure:
    """Compare hold and selected policy on the same dates and inputs."""
    hold_frame, selected_frame = _complete(hold), _complete(selected)
    if not hold_frame.index.equals(selected_frame.index):
        raise ValueError("hold and selected paths require the same finalized calendar")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=hold_frame.index,
            y=hold_frame["net_nav"],
            name="Buy & Hold — net NAV",
            mode="lines",
            line={"color": COLORS["neutral"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Net NAV: $%{y:,.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=selected_frame.index,
            y=selected_frame["net_nav"],
            name=f"{selected.config.policy.value} — net NAV",
            mode="lines",
            line={"color": COLORS["primary"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Net NAV: $%{y:,.2f}<extra></extra>",
        )
    )
    if show_selected_gross and selected.config.policy.is_rebalanced:
        figure.add_trace(
            go.Scatter(
                x=selected_frame.index,
                y=selected_frame["gross_nav"],
                name=f"{selected.config.policy.value} — gross NAV",
                mode="lines",
                line={"color": COLORS["secondary"], "dash": "dot"},
                hovertemplate=("%{x|%Y-%m-%d}<br>Gross NAV: $%{y:,.2f}<extra></extra>"),
            )
        )
    apply_risk_theme(
        figure,
        title=f"Historical Buy & Hold vs Selected Policy — NAV<br><sup>{_context(selected)}</sup>",
        x_title="Date (UTC)",
        y_title="Portfolio NAV (USD)",
        hovermode="x unified",
        meta={"hold": hold.provenance(), "selected": selected.provenance()},
    )
    figure.update_yaxes(tickprefix="$", separatethousands=True)
    return figure


def plot_weights_drift_and_rebalances(result: PortfolioPathResult) -> go.Figure:
    """Show pre/post-trade weights, targets, total drift and rebalance closes."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
        subplot_titles=("Asset weights", "Total allocation drift"),
    )
    assets = result.assets.reset_index()
    for index, asset in enumerate(result.target_weights.index):
        color = PALETTE[index % len(PALETTE)]
        subset = assets[assets["asset"] == asset]
        figure.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["pre_trade_weight"],
                name=f"{asset} pre-trade",
                mode="lines",
                line={"color": color, "dash": "dot", "width": 1},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["post_trade_weight"],
                name=f"{asset} post-trade",
                mode="lines",
                line={"color": color, "width": 2},
            ),
            row=1,
            col=1,
        )
        figure.add_hline(
            y=float(result.target_weights[asset]),
            line={"color": color, "dash": "dash", "width": 1},
            annotation_text=f"{asset} target {result.target_weights[asset]:.1%}",
            annotation_position="right",
            row=1,
            col=1,
        )
    complete = _complete(result)
    figure.add_trace(
        go.Scatter(
            x=complete.index,
            y=complete["total_pre_trade_drift"],
            name="Pre-trade total drift",
            mode="lines",
            line={"color": COLORS["dark"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Total drift=%{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    for when in complete.index[complete["rebalance_flag"]]:
        figure.add_vline(x=when, line={"color": COLORS["neutral"], "dash": "dash"})
    figure.update_yaxes(tickformat=".0%", row=1, col=1)
    figure.update_yaxes(tickformat=".1%", title="0.5 × L1 drift", row=2, col=1)
    apply_risk_theme(
        figure,
        title=(
            "Weights, Drift and Effective Rebalance Events"
            f"<br><sup>{_context(result)}</sup>"
        ),
        hovermode="x unified",
        meta=result.provenance(),
    )
    return figure


def plot_turnover_and_costs(result: PortfolioPathResult) -> go.Figure:
    """Turnover bars and cumulative transaction-cost line with audit details."""
    complete = _complete(result)
    custom = complete[
        ["gross_traded_notional", "one_way_turnover", "transaction_cost"]
    ].to_numpy()
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=complete.index,
            y=complete["gross_turnover"],
            name="Gross turnover",
            customdata=custom,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Gross turnover=%{y:.2%}"
                "<br>Gross traded notional=%{customdata[0]:,.2f}"
                "<br>One-way turnover=%{customdata[1]:.2%}"
                "<br>Cost=%{customdata[2]:,.2f}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=complete.index,
            y=complete["cumulative_transaction_cost"],
            name="Cumulative transaction cost",
            mode="lines",
            line={"color": COLORS["danger"], "width": 2},
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Cumulative cost: $%{y:,.2f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )
    figure.update_yaxes(title="Gross turnover", tickformat=".1%", secondary_y=False)
    figure.update_yaxes(title="Cumulative cost", secondary_y=True)
    apply_risk_theme(
        figure,
        title=f"Historical Turnover and Cumulative Costs<br><sup>{_context(result)}</sup>",
        hovermode="x unified",
        meta=result.provenance(),
    )
    return figure


def plot_comparative_drawdown(
    hold: PortfolioPathResult, selected: PortfolioPathResult
) -> go.Figure:
    """Compare net drawdowns and expose duration/recovery metadata."""
    hold_frame, selected_frame = _complete(hold), _complete(selected)
    if not hold_frame.index.equals(selected_frame.index):
        raise ValueError("drawdown comparison requires the same finalized calendar")
    figure = go.Figure()
    for label, frame, color in (
        ("Buy & Hold", hold_frame, COLORS["neutral"]),
        (selected.config.policy.value, selected_frame, COLORS["danger"]),
    ):
        stats = drawdown_statistics(frame["net_nav"])
        recovery = stats["recovery_date"]
        trough = stats["trough_date"]
        meta = (
            f"max={stats['maximum_drawdown']:.2%}; "
            f"trough={trough.date().isoformat() if trough is not None else 'N/A'}; "
            f"longest underwater={stats['longest_underwater_days']}d; "
            f"recovery={recovery.date().isoformat() if recovery is not None else 'not recovered'}"
        )
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["drawdown"],
                name=f"{label} ({meta})",
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    apply_risk_theme(
        figure,
        title=f"Historical Comparative Drawdown<br><sup>{_context(selected)}</sup>",
        x_title="Date (UTC)",
        y_title="Drawdown",
        hovermode="x unified",
        meta={"hold": hold.provenance(), "selected": selected.provenance()},
    )
    figure.update_yaxes(tickformat=".1%")
    return figure


def plot_gross_vs_net_nav(result: PortfolioPathResult) -> go.Figure:
    """Make transaction-cost drag visible for one selected policy."""
    complete = _complete(result)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=complete.index,
            y=complete["gross_nav"],
            name="Gross NAV",
            mode="lines",
            line={"color": COLORS["success"], "dash": "dot"},
            hovertemplate="%{x|%Y-%m-%d}<br>Gross NAV: $%{y:,.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=complete.index,
            y=complete["net_nav"],
            name="Net NAV",
            mode="lines",
            line={"color": COLORS["primary"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Net NAV: $%{y:,.2f}<extra></extra>",
        )
    )
    apply_risk_theme(
        figure,
        title=f"Historical Gross vs Net NAV<br><sup>{_context(result)}</sup>",
        x_title="Date (UTC)",
        y_title="Portfolio NAV (USD)",
        hovermode="x unified",
        meta=result.provenance(),
    )
    figure.update_yaxes(tickprefix="$", separatethousands=True)
    return figure


__all__ = [
    "plot_comparative_drawdown",
    "plot_gross_vs_net_nav",
    "plot_hold_vs_selected_nav",
    "plot_turnover_and_costs",
    "plot_weights_drift_and_rebalances",
]
