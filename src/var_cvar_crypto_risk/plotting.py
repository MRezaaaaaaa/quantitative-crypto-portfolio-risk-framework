"""Pure Plotly chart builders for Risk Lab.

These functions consume completed financial results and return
``plotly.graph_objects.Figure`` objects. They do not import Streamlit and do
not estimate risk, optimize weights, classify breaches, or rebuild portfolio
accounting.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from .plotly_theme import COLORS, PALETTE, apply_risk_theme, empty_figure
from .risk_conventions import loss_value_to_return_threshold


_TRAFFIC_LIGHT_EDGE = {
    "Green": COLORS["success"],
    "Yellow": COLORS["warning"],
    "Red": COLORS["danger"],
}


def _meta(provenance: Mapping[str, Any] | None, **extra: Any) -> dict[str, Any]:
    result = dict(provenance or {})
    result.update(extra)
    return result


def _save_if_requested(figure: go.Figure, output_path: str | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".html":
        figure.write_html(path, include_plotlyjs=True, full_html=True)
        return
    try:
        figure.write_image(path)
    except Exception as exc:  # pragma: no cover - optional static exporter
        raise RuntimeError(
            "Static Plotly export requires a compatible Kaleido/Chrome setup. "
            "Use HTML export or the Plotly modebar PNG control instead."
        ) from exc


def _clean_series(values: pd.Series, *, name: str) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    clean.name = name
    return clean


def plot_return_distribution_with_var_cvar(
    returns: pd.Series,
    var_value: float,
    cvar_value: float,
    title: str = "Historical Return Distribution with VaR and CVaR",
    confidence_level: float = 0.95,
    output_path: str | None = None,
    xlabel: str = "Historical Return",
    extra_var_lines: dict[str, float] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot an empirical historical distribution and supplied risk limits."""
    clean = _clean_series(returns, name="return")
    if clean.empty:
        return empty_figure(
            "No finalized returns are available.", title=title, meta=provenance
        )
    var_threshold = loss_value_to_return_threshold(var_value)
    cvar_threshold = loss_value_to_return_threshold(cvar_value)
    tail = clean[clean <= var_threshold]
    histogram_bin_group = "historical-return-distribution"
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=clean,
            nbinsx=60,
            bingroup=histogram_bin_group,
            name="Observed historical returns",
            marker_color=COLORS["primary"],
            opacity=0.78,
            hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
        )
    )
    if not tail.empty:
        figure.add_trace(
            go.Histogram(
                x=tail,
                nbinsx=60,
                bingroup=histogram_bin_group,
                name=f"Observed tail (worst {(1 - confidence_level):.1%})",
                marker_color=COLORS["danger"],
                opacity=0.72,
                hovertemplate="Tail return: %{x:.2%}<br>Count: %{y}<extra></extra>",
            )
        )
    figure.add_vline(
        x=var_threshold,
        line_color=COLORS["danger"],
        line_dash="dash",
        annotation_text=f"Historical VaR threshold {var_threshold:.2%}",
        annotation_textangle=-90,
        annotation_position="top right",
    )
    figure.add_vline(
        x=cvar_threshold,
        line_color=COLORS["purple"],
        line_dash="dot",
        annotation_text=f"Historical CVaR {cvar_threshold:.2%}",
        annotation_textangle=-90,
        annotation_position="top left",
    )
    for index, (label, value) in enumerate((extra_var_lines or {}).items()):
        figure.add_vline(
            x=loss_value_to_return_threshold(float(value)),
            line_color=PALETTE[(index + 3) % len(PALETTE)],
            line_dash="dashdot",
            annotation_text=str(label),
            annotation_textangle=-90,
            annotation_position="bottom right",
        )
    figure.update_layout(barmode="overlay")
    apply_risk_theme(
        figure,
        title=title,
        x_title=xlabel,
        y_title="Observation count",
        meta=_meta(
            provenance,
            chart="historical_return_distribution",
            confidence_level=confidence_level,
            var_threshold=var_threshold,
            cvar_threshold=cvar_threshold,
        ),
    )
    figure.update_xaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_cumulative_returns(
    portfolio_returns: pd.Series,
    title: str = "Historical Portfolio Cumulative Return",
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot supplied portfolio returns as a historical growth series."""
    clean = _clean_series(portfolio_returns, name="return")
    if clean.empty:
        return empty_figure("No returns are available.", title=title, meta=provenance)
    cumulative = (1.0 + clean).cumprod() - 1.0
    figure = go.Figure(
        go.Scatter(
            x=cumulative.index,
            y=cumulative,
            mode="lines",
            name="Historical cumulative return",
            line={"color": COLORS["primary"]},
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>",
        )
    )
    apply_risk_theme(
        figure,
        title=title,
        x_title="Date (UTC)",
        y_title="Cumulative return",
        hovermode="x unified",
        meta=_meta(provenance, chart="historical_cumulative_return"),
    )
    figure.update_yaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_drawdown(
    portfolio_returns: pd.Series,
    title: str = "Historical Portfolio Drawdown",
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot drawdown from supplied returns; audited paths use path charts."""
    clean = _clean_series(portfolio_returns, name="return")
    if clean.empty:
        return empty_figure("No returns are available.", title=title, meta=provenance)
    launch_date = clean.index[0] - pd.Timedelta(days=1)
    wealth = pd.concat([pd.Series([1.0], index=[launch_date]), (1.0 + clean).cumprod()])
    drawdown = wealth / wealth.cummax() - 1.0
    figure = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            name="Historical drawdown",
            line={"color": COLORS["danger"]},
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>",
        )
    )
    apply_risk_theme(
        figure,
        title=title,
        x_title="Date (UTC)",
        y_title="Drawdown",
        hovermode="x unified",
        meta=_meta(provenance, chart="historical_drawdown"),
    )
    figure.update_yaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_var_backtest(
    backtest_df: pd.DataFrame,
    method: str,
    confidence_level: float,
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot aligned actual returns, supplied VaR forecasts, and breaches."""
    required = {"actual_return", "var_forecast", "breach"}
    if backtest_df.empty or not required.issubset(backtest_df.columns):
        return empty_figure(
            "No valid backtest observations are available.",
            title="Historical VaR Backtest",
            meta=provenance,
        )
    actual = backtest_df["actual_return"].astype(float)
    threshold = -backtest_df["var_forecast"].astype(float)
    breaches = backtest_df[backtest_df["breach"].astype(bool)]
    label = method.replace("_", " ").title()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual,
            mode="lines",
            name="Observed return",
            line={"color": COLORS["neutral"], "width": 1},
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=threshold.index,
            y=threshold,
            mode="lines",
            name=f"Historical VaR threshold — {label}",
            line={"color": COLORS["danger"], "dash": "dash"},
            hovertemplate="%{x|%Y-%m-%d}<br>Threshold: %{y:.2%}<extra></extra>",
        )
    )
    if not breaches.empty:
        figure.add_trace(
            go.Scatter(
                x=breaches.index,
                y=breaches["actual_return"],
                mode="markers",
                name=f"Breach ({len(breaches)})",
                marker={"color": COLORS["danger"], "symbol": "x", "size": 8},
                hovertemplate="%{x|%Y-%m-%d}<br>Breach return: %{y:.2%}<extra></extra>",
            )
        )
    apply_risk_theme(
        figure,
        title=f"Historical VaR Backtest — {label} ({confidence_level:.1%})",
        x_title="Date (UTC)",
        y_title="Return / threshold",
        hovermode="x unified",
        meta=_meta(
            provenance,
            chart="var_backtest",
            method=method,
            confidence_level=confidence_level,
        ),
    )
    figure.update_yaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_breach_timeline(
    backtest_df: pd.DataFrame,
    method: str,
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot supplied breach classifications without reclassifying them."""
    title = f"Historical Breach Timeline — {method.replace('_', ' ').title()}"
    if backtest_df.empty or "breach" not in backtest_df:
        return empty_figure("No backtest observations are available.", title=title)
    breach = backtest_df["breach"].astype(bool)
    colors = np.where(breach, COLORS["danger"], "rgba(160,174,192,0.18)")
    figure = go.Figure(
        go.Bar(
            x=backtest_df.index,
            y=breach.astype(int),
            name="Breach event",
            marker_color=colors,
            hovertemplate="%{x|%Y-%m-%d}<br>Breach: %{y}<extra></extra>",
        )
    )
    apply_risk_theme(
        figure,
        title=title,
        x_title="Date (UTC)",
        y_title="Breach indicator",
        meta=_meta(provenance, chart="breach_timeline", method=method),
        height=340,
    )
    figure.update_yaxes(tickvals=[0, 1], ticktext=["No", "Yes"], range=[0, 1.15])
    _save_if_requested(figure, output_path)
    return figure


def plot_mc_loss_distribution(
    scenario_returns: pd.Series,
    var_value: float,
    cvar_value: float,
    title: str = "Monte Carlo Scenario Return Distribution",
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot already-simulated scenario returns and supplied risk limits."""
    return plot_return_distribution_with_var_cvar(
        scenario_returns,
        var_value,
        cvar_value,
        title=title,
        confidence_level=float((provenance or {}).get("confidence_level", 0.95)),
        output_path=output_path,
        xlabel="Simulated scenario return",
        provenance=_meta(provenance, chart="monte_carlo_distribution"),
    )


def plot_mc_portfolio_paths(
    paths: pd.DataFrame,
    output_path: str | None = None,
    max_paths_to_plot: int = 100,
    title: str = "Monte Carlo Portfolio Value Paths",
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot supplied simulated portfolio paths and their cross-path mean."""
    if paths.empty:
        return empty_figure("No simulated paths are available.", title=title)
    columns = list(paths.columns)
    if len(columns) > max_paths_to_plot:
        positions = np.linspace(0, len(columns) - 1, max_paths_to_plot, dtype=int)
        columns = [columns[position] for position in positions]
    figure = go.Figure()
    for position, column in enumerate(columns):
        figure.add_trace(
            go.Scatter(
                x=paths.index,
                y=paths[column],
                mode="lines",
                name="Simulated path" if position == 0 else str(column),
                legendgroup="paths",
                showlegend=position == 0,
                line={"color": "rgba(76,120,168,0.18)", "width": 1},
                hovertemplate="Step %{x}<br>Value: $%{y:,.2f}<extra></extra>",
            )
        )
    mean_path = paths.mean(axis=1)
    figure.add_trace(
        go.Scatter(
            x=mean_path.index,
            y=mean_path,
            mode="lines",
            name="Mean simulated path",
            line={"color": COLORS["warning"], "width": 3},
            hovertemplate="Step %{x}<br>Mean: $%{y:,.2f}<extra></extra>",
        )
    )
    apply_risk_theme(
        figure,
        title=title,
        x_title="Simulation step",
        y_title="Portfolio value (USD)",
        hovermode="x unified",
        meta=_meta(provenance, chart="monte_carlo_paths"),
    )
    figure.update_yaxes(tickprefix="$", separatethousands=True)
    _save_if_requested(figure, output_path)
    return figure


def plot_normal_vs_student_t_distribution(
    normal_returns: pd.Series,
    student_t_returns: pd.Series,
    output_path: str | None = None,
    title: str = "Normal vs Student-t Monte Carlo Return Distribution",
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Overlay supplied Normal and Student-t scenario returns."""
    normal = _clean_series(normal_returns, name="normal")
    student = _clean_series(student_t_returns, name="student_t")
    if normal.empty or student.empty:
        return empty_figure("Both scenario samples are required.", title=title)
    figure = go.Figure()
    for values, name, color in (
        (normal, "Normal scenarios", COLORS["primary"]),
        (student, "Student-t scenarios", COLORS["danger"]),
    ):
        figure.add_trace(
            go.Histogram(
                x=values,
                nbinsx=70,
                histnorm="probability",
                opacity=0.58,
                name=name,
                marker_color=color,
                hovertemplate=(
                    "Return: %{x:.2%}<br>Probability: %{y:.3%}<extra></extra>"
                ),
            )
        )
    figure.update_layout(barmode="overlay")
    apply_risk_theme(
        figure,
        title=title,
        x_title="Simulated scenario return",
        y_title="Probability",
        meta=_meta(provenance, chart="normal_vs_student_t"),
    )
    figure.update_xaxes(tickformat=".1%")
    figure.update_yaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_var_cvar_method_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | None = None,
    title: str = "VaR and CVaR by Method",
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot precomputed VaR/CVaR values by method."""
    required = {"Method", "VaR", "CVaR"}
    if comparison_df.empty or not required.issubset(comparison_df.columns):
        return empty_figure("No method comparison is available.", title=title)
    figure = go.Figure()
    for metric, color in (("VaR", COLORS["primary"]), ("CVaR", COLORS["danger"])):
        figure.add_trace(
            go.Bar(
                x=comparison_df["Method"].astype(str),
                y=comparison_df[metric].astype(float),
                name=metric,
                marker_color=color,
                texttemplate="%{y:.2%}",
                hovertemplate=f"%{{x}}<br>{metric}: %{{y:.2%}}<extra></extra>",
            )
        )
    figure.update_layout(barmode="group")
    apply_risk_theme(
        figure,
        title=title,
        x_title="Method",
        y_title="Loss magnitude",
        meta=_meta(provenance, chart="var_cvar_method_comparison"),
    )
    figure.update_yaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_optimized_weights(
    weights: pd.Series,
    title: str = "Optimized Portfolio Weights",
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot supplied optimized weights without running optimization."""
    clean = (
        pd.to_numeric(weights, errors="coerce").dropna().sort_values(ascending=False)
    )
    if clean.empty:
        return empty_figure("No optimized weights are available.", title=title)
    figure = go.Figure(
        go.Bar(
            x=clean.index.astype(str),
            y=clean,
            name="Optimized weight",
            marker_color=[
                COLORS["danger"] if value < 0 else COLORS["primary"] for value in clean
            ],
            texttemplate="%{y:.1%}",
            hovertemplate="%{x}<br>Weight: %{y:.2%}<extra></extra>",
        )
    )
    apply_risk_theme(
        figure,
        title=title,
        x_title="Asset",
        y_title="Portfolio weight",
        meta=_meta(provenance, chart="optimized_weights"),
    )
    figure.update_yaxes(tickformat=".0%")
    _save_if_requested(figure, output_path)
    return figure


def plot_portfolio_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | None = None,
    title: str = "Current vs Optimized Portfolio Comparison",
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot precomputed risk/return metrics across portfolios."""
    metrics = [
        column
        for column in ("Expected Return", "Volatility", "VaR", "CVaR")
        if column in comparison_df
    ]
    if comparison_df.empty or "Portfolio" not in comparison_df or not metrics:
        return empty_figure("No portfolio comparison is available.", title=title)
    figure = go.Figure()
    for position, row in comparison_df.reset_index(drop=True).iterrows():
        figure.add_trace(
            go.Bar(
                x=metrics,
                y=[float(row[metric]) for metric in metrics],
                name=str(row["Portfolio"]),
                marker_color=PALETTE[position % len(PALETTE)],
                texttemplate="%{y:.1%}",
                hovertemplate=(
                    "%{x}<br>Value: %{y:.2%}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(barmode="group")
    apply_risk_theme(
        figure,
        title=title,
        x_title="Metric",
        y_title="Per-horizon value",
        meta=_meta(provenance, chart="portfolio_comparison"),
    )
    figure.update_yaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_cvar_efficient_frontier(
    frontier_df: pd.DataFrame,
    output_path: str | None = None,
    title: str = "CVaR Efficient Frontier",
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot precomputed feasible frontier points."""
    required = {"expected_return", "CVaR"}
    if frontier_df.empty or not required.issubset(frontier_df.columns):
        return empty_figure("No feasible frontier points are available.", title=title)
    clean = frontier_df.dropna(subset=list(required)).sort_values("CVaR")
    if clean.empty:
        return empty_figure("No feasible frontier points are available.", title=title)
    figure = go.Figure(
        go.Scatter(
            x=clean["CVaR"],
            y=clean["expected_return"],
            mode="lines+markers",
            name="Feasible CVaR frontier",
            line={"color": COLORS["primary"]},
            marker={"color": COLORS["danger"], "size": 8},
            customdata=clean.index,
            hovertemplate=(
                "CVaR: %{x:.2%}<br>Expected return: %{y:.2%}"
                "<br>Point: %{customdata}<extra></extra>"
            ),
        )
    )
    apply_risk_theme(
        figure,
        title=title,
        x_title="CVaR loss magnitude",
        y_title="Expected return",
        meta=_meta(provenance, chart="cvar_efficient_frontier"),
    )
    figure.update_xaxes(tickformat=".1%")
    figure.update_yaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_allocation_comparison(
    weights_dict: dict,
    output_path: str | None = None,
    title: str = "Portfolio Allocation Comparison",
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot supplied allocations across named portfolios."""
    if not weights_dict:
        return empty_figure("No portfolio allocations are available.", title=title)
    assets = list(
        dict.fromkeys(
            str(asset) for series in weights_dict.values() for asset in series.index
        )
    )
    figure = go.Figure()
    for position, (label, series) in enumerate(weights_dict.items()):
        figure.add_trace(
            go.Bar(
                x=assets,
                y=[float(series.get(asset, 0.0)) for asset in assets],
                name=str(label),
                marker_color=PALETTE[position % len(PALETTE)],
                texttemplate="%{y:.1%}",
                hovertemplate=(
                    "%{x}<br>Weight: %{y:.2%}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(barmode="group")
    apply_risk_theme(
        figure,
        title=title,
        x_title="Asset",
        y_title="Portfolio weight",
        meta=_meta(provenance, chart="allocation_comparison"),
    )
    figure.update_yaxes(tickformat=".0%")
    _save_if_requested(figure, output_path)
    return figure


def plot_model_comparison_backtest(
    comparison_df: pd.DataFrame,
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot precomputed actual and expected breach counts by model."""
    required = {"method", "actual_breaches", "expected_breaches"}
    valid = comparison_df.copy()
    if "error" in valid:
        valid = valid[valid["error"].isna()]
    title = "Historical Backtest — Actual vs Expected Breaches"
    if valid.empty or not required.issubset(valid.columns):
        return empty_figure("No valid model comparison is available.", title=title)
    labels = valid["method"].astype(str).str.replace("_", " ").str.title()
    colors = [
        _TRAFFIC_LIGHT_EDGE.get(str(status), COLORS["neutral"])
        for status in valid.get("traffic_light", pd.Series("", index=valid.index))
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=labels,
            y=valid["actual_breaches"],
            name="Actual breaches",
            marker={"color": colors},
            hovertemplate="%{x}<br>Actual: %{y}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=valid["expected_breaches"],
            name="Expected breaches",
            marker_color=COLORS["neutral"],
            hovertemplate="%{x}<br>Expected: %{y:.2f}<extra></extra>",
        )
    )
    figure.update_layout(barmode="group")
    apply_risk_theme(
        figure,
        title=title,
        x_title="VaR model",
        y_title="Breach count",
        meta=_meta(provenance, chart="backtest_model_comparison"),
    )
    _save_if_requested(figure, output_path)
    return figure


def plot_rolling_breach_rate(
    rolling_breach_rate: pd.Series,
    expected_breach_rate: float,
    method: str,
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot supplied rolling breach rates against the expected rate."""
    clean = _clean_series(rolling_breach_rate, name="breach_rate")
    title = f"Rolling Breach Rate — {method.replace('_', ' ').title()}"
    if clean.empty:
        return empty_figure("No rolling breach rate is available.", title=title)
    figure = go.Figure(
        go.Scatter(
            x=clean.index,
            y=clean,
            mode="lines",
            name="Rolling breach rate",
            line={"color": COLORS["primary"]},
            hovertemplate="%{x|%Y-%m-%d}<br>Rate: %{y:.2%}<extra></extra>",
        )
    )
    figure.add_hline(
        y=expected_breach_rate,
        line_color=COLORS["danger"],
        line_dash="dash",
        annotation_text=f"Expected {expected_breach_rate:.2%}",
    )
    apply_risk_theme(
        figure,
        title=title,
        x_title="Date (UTC)",
        y_title="Breach rate",
        hovermode="x unified",
        meta=_meta(
            provenance,
            chart="rolling_breach_rate",
            expected_breach_rate=expected_breach_rate,
            method=method,
        ),
    )
    figure.update_yaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_asset_return_distributions(
    asset_returns: pd.DataFrame,
    horizon_days: int = 1,
    confidence_level: float = 0.95,
    return_method: str = "simple",
    output_path: str | None = None,
    risk_levels: Mapping[str, Mapping[str, float]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot chart-ready, horizon-matched returns for each asset.

    ``asset_returns`` must already represent the requested horizon. Optional
    ``risk_levels`` supplies precomputed signed-loss ``var``/``cvar`` values.
    """
    del return_method  # public-call compatibility; no calculation occurs here
    assets = list(asset_returns.columns)
    title = (
        "Historical Asset Return Distributions"
        if horizon_days == 1
        else f"Historical Asset {horizon_days}-day Return Distributions"
    )
    if asset_returns.empty or not assets:
        return empty_figure("No asset returns are available.", title=title)
    columns = min(3, len(assets))
    rows = int(np.ceil(len(assets) / columns))
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=assets)
    for position, asset in enumerate(assets):
        row, column = divmod(position, columns)
        clean = pd.to_numeric(asset_returns[asset], errors="coerce").dropna()
        figure.add_trace(
            go.Histogram(
                x=clean,
                nbinsx=40,
                name=str(asset),
                marker_color=PALETTE[position % len(PALETTE)],
                showlegend=False,
                hovertemplate=(
                    "Return: %{x:.2%}<br>Count: %{y}<extra>" + str(asset) + "</extra>"
                ),
            ),
            row=row + 1,
            col=column + 1,
        )
        levels = (risk_levels or {}).get(str(asset), {})
        for name, color, dash in (
            ("var", COLORS["danger"], "dash"),
            ("cvar", COLORS["purple"], "dot"),
        ):
            if name in levels:
                figure.add_vline(
                    x=loss_value_to_return_threshold(float(levels[name])),
                    line_color=color,
                    line_dash=dash,
                    row=row + 1,
                    col=column + 1,
                )
    apply_risk_theme(
        figure,
        title=title,
        x_title="Historical return",
        y_title="Observation count",
        meta=_meta(
            provenance,
            chart="asset_return_distributions",
            horizon_days=horizon_days,
            confidence_level=confidence_level,
        ),
        height=max(420, 330 * rows),
    )
    figure.update_xaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_qq_vs_normal(
    returns: pd.Series,
    title: str = "Historical Returns — QQ Plot vs Normal",
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot sample quantiles against Normal theoretical quantiles."""
    clean = _clean_series(returns, name="return")
    if len(clean) < 3:
        return empty_figure("At least three returns are required.", title=title)
    theoretical, ordered = stats.probplot(clean.to_numpy(), dist="norm", fit=False)
    slope, intercept = np.polyfit(theoretical, ordered, 1)
    reference = slope * np.asarray(theoretical) + intercept
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=theoretical,
            y=ordered,
            mode="markers",
            name="Observed sample quantiles",
            marker={"color": COLORS["primary"], "size": 6, "opacity": 0.65},
            hovertemplate=(
                "Normal quantile: %{x:.3f}<br>Sample return: %{y:.2%}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=theoretical,
            y=reference,
            mode="lines",
            name="Normal reference",
            line={"color": COLORS["danger"]},
            hoverinfo="skip",
        )
    )
    apply_risk_theme(
        figure,
        title=title,
        x_title="Theoretical Normal quantile",
        y_title="Historical sample return",
        meta=_meta(provenance, chart="qq_plot"),
    )
    figure.update_yaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_asset_cumulative_returns(
    asset_returns: pd.DataFrame,
    title: str = "Historical Asset Cumulative Returns",
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot historical asset growth computed from supplied simple returns."""
    if asset_returns.empty:
        return empty_figure("No asset returns are available.", title=title)
    cumulative = (1.0 + asset_returns).cumprod() - 1.0
    figure = go.Figure()
    for position, asset in enumerate(cumulative.columns):
        figure.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative[asset],
                mode="lines",
                name=str(asset),
                line={"color": PALETTE[position % len(PALETTE)]},
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    apply_risk_theme(
        figure,
        title=title,
        x_title="Date (UTC)",
        y_title="Cumulative return",
        hovermode="x unified",
        meta=_meta(provenance, chart="asset_cumulative_returns"),
    )
    figure.update_yaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_asset_drawdowns(
    asset_drawdowns: pd.DataFrame,
    title: str = "Historical Asset Drawdowns",
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot precomputed per-asset drawdowns."""
    if asset_drawdowns.empty:
        return empty_figure("No asset drawdowns are available.", title=title)
    figure = go.Figure()
    for position, asset in enumerate(asset_drawdowns.columns):
        figure.add_trace(
            go.Scatter(
                x=asset_drawdowns.index,
                y=asset_drawdowns[asset],
                mode="lines",
                name=str(asset),
                line={"color": PALETTE[position % len(PALETTE)]},
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    apply_risk_theme(
        figure,
        title=title,
        x_title="Date (UTC)",
        y_title="Drawdown",
        hovermode="x unified",
        meta=_meta(provenance, chart="asset_drawdowns"),
    )
    figure.update_yaxes(tickformat=".1%")
    _save_if_requested(figure, output_path)
    return figure


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Historical Asset Return Correlation Matrix",
    output_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot an annotated correlation matrix with fixed [-1, 1] scale."""
    if corr_matrix.empty:
        return empty_figure("No correlation matrix is available.", title=title)
    figure = go.Figure(
        go.Heatmap(
            z=corr_matrix.to_numpy(dtype=float),
            x=corr_matrix.columns.astype(str),
            y=corr_matrix.index.astype(str),
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu_r",
            colorbar={"title": "Correlation"},
            text=corr_matrix.to_numpy(dtype=float),
            texttemplate="%{text:.2f}",
            hovertemplate=("%{y} / %{x}<br>Correlation: %{z:.3f}<extra></extra>"),
        )
    )
    apply_risk_theme(
        figure,
        title=title,
        x_title="Asset",
        y_title="Asset",
        meta=_meta(provenance, chart="correlation_heatmap"),
        height=max(430, 68 * len(corr_matrix) + 180),
    )
    _save_if_requested(figure, output_path)
    return figure


def plot_rolling_average_correlation(
    rolling_corr: pd.Series,
    title: str = "Rolling Average Pairwise Correlation",
    output_path: str | None = None,
    method: str | None = None,
    window: int | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Plot supplied rolling correlation with explicit method and window."""
    clean = _clean_series(rolling_corr, name="correlation")
    context = ""
    if method or window:
        context = f" — {(method or 'unspecified').title()}"
        if window:
            context += f", {window}-observation window"
    full_title = title + context
    if clean.empty:
        return empty_figure("No rolling correlation is available.", title=full_title)
    figure = go.Figure(
        go.Scatter(
            x=clean.index,
            y=clean,
            mode="lines",
            name="Average pairwise correlation",
            line={"color": COLORS["primary"]},
            hovertemplate=("%{x|%Y-%m-%d}<br>Correlation: %{y:.3f}<extra></extra>"),
        )
    )
    apply_risk_theme(
        figure,
        title=full_title,
        x_title="Date (UTC)",
        y_title="Average pairwise correlation",
        hovermode="x unified",
        meta=_meta(
            provenance,
            chart="rolling_average_correlation",
            method=method,
            window=window,
        ),
    )
    figure.update_yaxes(range=[-1, 1], tickformat=".2f")
    _save_if_requested(figure, output_path)
    return figure


__all__ = [
    "plot_allocation_comparison",
    "plot_asset_cumulative_returns",
    "plot_asset_drawdowns",
    "plot_asset_return_distributions",
    "plot_breach_timeline",
    "plot_correlation_heatmap",
    "plot_cumulative_returns",
    "plot_cvar_efficient_frontier",
    "plot_drawdown",
    "plot_mc_loss_distribution",
    "plot_mc_portfolio_paths",
    "plot_model_comparison_backtest",
    "plot_normal_vs_student_t_distribution",
    "plot_optimized_weights",
    "plot_portfolio_comparison",
    "plot_qq_vs_normal",
    "plot_return_distribution_with_var_cvar",
    "plot_rolling_average_correlation",
    "plot_rolling_breach_rate",
    "plot_var_backtest",
    "plot_var_cvar_method_comparison",
]
