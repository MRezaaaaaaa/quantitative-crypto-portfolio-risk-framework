"""Matplotlib plotting helpers for the risk engine.

Charts are styled to be clean and professional enough for portfolio /
LinkedIn presentation. Only matplotlib is used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .risk_conventions import loss_value_to_return_threshold


_BG_COLOR = "#f5f5f5"
_PRIMARY = "#1f3b73"
_DANGER = "#c8102e"
_GRID = "#cfcfcf"


def _styled_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(_BG_COLOR)
    ax.grid(True, color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#888")


def _save_if_requested(fig: plt.Figure, output_path: str | None) -> None:
    if output_path is None:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())


def plot_return_distribution_with_var_cvar(
    returns: pd.Series,
    var_value: float,
    cvar_value: float,
    title: str = "Return Distribution with VaR and CVaR",
    confidence_level: float = 0.95,
    output_path: str | None = None,
    xlabel: str = "Daily Return",
    extra_var_lines: dict[str, float] | None = None,
) -> plt.Figure:
    """Histogram of returns with VaR threshold and CVaR tail shaded.

    Parameters
    ----------
    returns : pandas.Series
        Daily (or horizon-matched) portfolio returns.
    var_value : float
        Signed decimal loss value (e.g. 0.042 for a 4.2% loss).
    cvar_value : float
        Signed decimal loss value; expected to be >= ``var_value`` in loss
        space.
    title : str, optional
    confidence_level : float, optional
        Used in annotations.
    output_path : str | None, optional
        If provided, the figure is saved as PNG.
    xlabel : str, optional
        X-axis label (e.g. ``"7-day Return"`` for a horizon-matched chart).
    extra_var_lines : dict[str, float] | None, optional
        Optional ``{label: signed_loss}`` entries drawn as additional
        vertical VaR lines with distinct dashed styles (used by the
        "Show all VaR lines" comparison).

    Returns
    -------
    matplotlib.figure.Figure
    """
    clean = returns.dropna().values
    var_threshold = loss_value_to_return_threshold(var_value)
    cvar_threshold = loss_value_to_return_threshold(cvar_value)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _styled_axes(ax)

    counts, bins, _ = ax.hist(
        clean,
        bins=60,
        color=_PRIMARY,
        edgecolor="white",
        alpha=0.85,
        label="Return distribution",
    )

    mu = float(np.mean(clean))
    sigma = float(np.std(clean, ddof=1))
    if sigma > 0:
        x = np.linspace(clean.min(), clean.max(), 400)
        pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
        bin_width = bins[1] - bins[0]
        scaled_pdf = pdf * len(clean) * bin_width
        ax.plot(x, scaled_pdf, color="#333", linewidth=1.5, label="Normal fit")

    tail_mask = clean <= var_threshold
    if tail_mask.any():
        ax.hist(
            clean[tail_mask],
            bins=bins,
            color=_DANGER,
            alpha=0.55,
            label=f"CVaR tail (worst {(1 - confidence_level) * 100:.0f}%)",
        )

    ax.axvline(
        var_threshold,
        color=_DANGER,
        linestyle="--",
        linewidth=1.8,
        label=f"VaR {confidence_level * 100:.0f}% = {var_value * 100:.2f}%",
    )
    ax.axvline(
        cvar_threshold,
        color="#7a0017",
        linestyle=":",
        linewidth=1.8,
        label=f"CVaR {confidence_level * 100:.0f}% = {cvar_value * 100:.2f}%",
    )

    if extra_var_lines:
        _extra_styles = [
            ("#1b7837", (0, (3, 1, 1, 1))),
            ("#762a83", (0, (5, 2))),
            ("#e08214", (0, (1, 1))),
            ("#2166ac", (0, (4, 1, 1, 1, 1, 1))),
        ]
        for i, (lbl, val) in enumerate(extra_var_lines.items()):
            color, dash = _extra_styles[i % len(_extra_styles)]
            ax.axvline(
                -float(val),
                color=color,
                linestyle=dash,
                linewidth=1.4,
                alpha=0.9,
                label=f"{lbl} = {float(val) * 100:.2f}%",
            )

    y_top = counts.max() if len(counts) else 1.0
    ax.annotate(
        f"VaR = {var_value * 100:.2f}%",
        xy=(var_threshold, y_top * 0.85),
        xytext=(var_threshold - abs(var_threshold) * 0.6, y_top * 0.85),
        color=_DANGER,
        fontsize=9,
        ha="right",
    )
    ax.annotate(
        f"CVaR = {cvar_value * 100:.2f}%",
        xy=(cvar_threshold, y_top * 0.65),
        xytext=(cvar_threshold - abs(cvar_threshold) * 0.6, y_top * 0.65),
        color="#7a0017",
        fontsize=9,
        ha="right",
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_cumulative_returns(
    portfolio_returns: pd.Series,
    title: str = "Portfolio Cumulative Returns",
    output_path: str | None = None,
) -> plt.Figure:
    """Cumulative return curve over time.

    Includes a horizontal zero line for reference.
    """
    cumulative = (1.0 + portfolio_returns).cumprod() - 1.0

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    _styled_axes(ax)

    ax.plot(cumulative.index, cumulative.values * 100.0, color=_PRIMARY, linewidth=1.6)
    ax.axhline(0, color="#444", linewidth=0.8, linestyle="-")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_drawdown(
    portfolio_returns: pd.Series,
    title: str = "Portfolio Drawdown",
    output_path: str | None = None,
) -> plt.Figure:
    """Drawdown over time as a filled area chart.

    The area below zero is filled in red to emphasize losses.
    """
    cumulative = (1.0 + portfolio_returns).cumprod() - 1.0
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / (1.0 + rolling_max)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    _styled_axes(ax)

    dd_pct = drawdown.values * 100.0
    ax.fill_between(
        drawdown.index, dd_pct, 0, where=dd_pct < 0, color=_DANGER, alpha=0.45
    )
    ax.plot(drawdown.index, dd_pct, color=_DANGER, linewidth=1.0)
    ax.axhline(0, color="#444", linewidth=0.8, linestyle="-")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


# ─── VaR backtesting charts ───────────────────────────────────────────────


_TRAFFIC_LIGHT_EDGE = {
    "Green": "#27ae60",
    "Yellow": "#f39c12",
    "Red": _DANGER,
}


def plot_var_backtest(
    backtest_df: pd.DataFrame,
    method: str,
    confidence_level: float,
    output_path: str | None = None,
) -> plt.Figure:
    """Plot rolling VaR backtest: actual returns vs negative VaR forecast line.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of :func:`var_cvar_crypto_risk.backtesting.rolling_var_forecast`.
        Must contain columns ``actual_return``, ``var_forecast``, ``breach``
        and a DatetimeIndex.
    method : str
        Name of the VaR method (e.g. ``"historical"``).
    confidence_level : float
        Confidence level used to generate the forecast (e.g. ``0.95``).
    output_path : str | None, optional
        If provided, the figure is saved as PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    _styled_axes(ax)

    dates = backtest_df.index
    actual = backtest_df["actual_return"].to_numpy(dtype=float)
    var_forecast = backtest_df["var_forecast"].to_numpy(dtype=float)
    breach_mask = backtest_df["breach"].to_numpy(dtype=bool)
    breach_count = int(breach_mask.sum())

    ax.plot(
        dates,
        actual,
        color="#555",
        linewidth=1.0,
        label="Actual Return",
    )
    ax.plot(
        dates,
        -var_forecast,
        color=_DANGER,
        linestyle="--",
        linewidth=1.4,
        label=f"−VaR {confidence_level * 100:.0f}% ({method})",
    )

    if breach_count > 0:
        ax.scatter(
            dates[breach_mask],
            actual[breach_mask],
            marker="x",
            s=30,
            color=_DANGER,
            zorder=5,
            label=f"Breach ({breach_count} total)",
        )

    ax.axhline(0.0, color="#444", linewidth=0.6)

    method_label = method.replace("_", " ").title()
    ax.set_title(
        f"VaR Backtest — {method_label} | {confidence_level * 100:.0f}% Confidence",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_breach_timeline(
    backtest_df: pd.DataFrame,
    method: str,
    output_path: str | None = None,
) -> plt.Figure:
    """Visualise breach events on a timeline to detect clustering.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of :func:`var_cvar_crypto_risk.backtesting.rolling_var_forecast`.
    method : str
        Name of the VaR method (used in the title only).
    output_path : str | None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 2.5), facecolor="white")
    _styled_axes(ax)

    breach_dates = backtest_df.index[
        backtest_df["breach"].to_numpy(dtype=bool)
    ]
    breach_count = int(len(breach_dates))

    ax.axhline(1.0, color="#999", linewidth=0.5)

    if breach_count > 0:
        ax.vlines(
            breach_dates,
            ymin=0.0,
            ymax=2.0,
            color=_DANGER,
            linewidth=0.8,
            alpha=0.7,
        )

    ax.set_xlim(backtest_df.index.min(), backtest_df.index.max())
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks([])
    ax.set_ylabel("")

    method_label = method.replace("_", " ").title()
    ax.set_title(
        f"Breach Timeline — {method_label}", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.annotate(
        f"Total breaches: {breach_count}",
        xy=(0.99, 0.92),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        color=_DANGER,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


# ─── Monte Carlo charts ───────────────────────────────────────────────────


def plot_mc_loss_distribution(
    scenario_returns: pd.Series,
    var_value: float,
    cvar_value: float,
    title: str = "Monte Carlo Return Distribution",
    output_path: str | None = None,
) -> plt.Figure:
    """Histogram of Monte Carlo scenario returns with VaR and CVaR markers.

    Parameters
    ----------
    scenario_returns : pd.Series
        Simulated portfolio scenario returns (positive = gain, negative = loss).
    var_value : float
        Signed decimal loss value to mark as the VaR threshold.
    cvar_value : float
        Signed decimal loss value to mark as the CVaR (>= ``var_value`` in
        loss space).
    title : str, optional
    output_path : str | None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    values = scenario_returns.to_numpy(dtype=float)
    var_threshold = loss_value_to_return_threshold(var_value)
    cvar_threshold = loss_value_to_return_threshold(cvar_value)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _styled_axes(ax)

    counts, bins, _ = ax.hist(
        values,
        bins=60,
        color=_PRIMARY,
        edgecolor="white",
        alpha=0.85,
        label="MC scenario returns",
    )

    tail_mask = values <= var_threshold
    if tail_mask.any():
        ax.hist(
            values[tail_mask],
            bins=bins,
            color=_DANGER,
            alpha=0.55,
            label="Tail (worse than VaR)",
        )

    ax.axvline(
        var_threshold,
        color=_DANGER,
        linestyle="--",
        linewidth=1.8,
        label=f"VaR = {var_value * 100:.2f}%",
    )
    ax.axvline(
        cvar_threshold,
        color="#7a0017",
        linestyle=":",
        linewidth=1.8,
        label=f"CVaR = {cvar_value * 100:.2f}%",
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Scenario Return")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_mc_portfolio_paths(
    paths: pd.DataFrame,
    output_path: str | None = None,
    max_paths_to_plot: int = 100,
    title: str = "Monte Carlo Portfolio Value Paths",
) -> plt.Figure:
    """Plot simulated portfolio value paths, with the mean path overlaid.

    Parameters
    ----------
    paths : pd.DataFrame
        Output of :func:`var_cvar_crypto_risk.monte_carlo.simulate_portfolio_paths`.
        Shape ``(horizon_days + 1) x n_paths``.
    output_path : str | None, optional
    max_paths_to_plot : int, optional
        Plot at most this many random paths for readability.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_paths = paths.shape[1]
    plot_n = min(max_paths_to_plot, n_paths)
    if plot_n < n_paths:
        rng = np.random.default_rng(seed=0)
        cols = rng.choice(n_paths, size=plot_n, replace=False)
        sampled = paths.iloc[:, cols]
    else:
        sampled = paths

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _styled_axes(ax)

    ax.plot(
        sampled.index,
        sampled.values,
        color=_PRIMARY,
        alpha=0.25,
        linewidth=0.8,
    )
    mean_path = paths.mean(axis=1)
    ax.plot(
        mean_path.index,
        mean_path.values,
        color="#1a1a1a",
        linewidth=1.8,
        label="Mean path",
    )

    initial_value = float(paths.iloc[0, 0])
    ax.axhline(
        initial_value,
        color="#888",
        linestyle="--",
        linewidth=0.8,
        label=f"Initial value (${initial_value:,.0f})",
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_normal_vs_student_t_distribution(
    normal_returns: pd.Series,
    student_t_returns: pd.Series,
    output_path: str | None = None,
    title: str = "Normal vs Student-t Monte Carlo — Portfolio Return Distribution",
) -> plt.Figure:
    """Overlay histograms of Normal and Student-t scenario returns.

    Parameters
    ----------
    normal_returns : pd.Series
    student_t_returns : pd.Series
    output_path : str | None, optional
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _styled_axes(ax)

    combined = np.concatenate(
        [
            normal_returns.to_numpy(dtype=float),
            student_t_returns.to_numpy(dtype=float),
        ]
    )
    bins = np.linspace(np.min(combined), np.max(combined), 80)

    ax.hist(
        normal_returns.to_numpy(dtype=float),
        bins=bins,
        color=_PRIMARY,
        alpha=0.55,
        label="Normal MC",
        edgecolor="white",
    )
    ax.hist(
        student_t_returns.to_numpy(dtype=float),
        bins=bins,
        color=_DANGER,
        alpha=0.45,
        label="Student-t MC",
        edgecolor="white",
    )

    ax.axvline(0.0, color="#444", linewidth=0.6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Scenario Return")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_var_cvar_method_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | None = None,
    title: str = "VaR & CVaR by Method",
) -> plt.Figure:
    """Grouped bar chart of VaR and CVaR by risk method.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of
        :func:`var_cvar_crypto_risk.monte_carlo.compare_all_risk_methods`.
        Must have columns ``Method``, ``VaR``, ``CVaR``.
    output_path : str | None, optional
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    methods = comparison_df["Method"].astype(str).tolist()
    var_vals = comparison_df["VaR"].to_numpy(dtype=float) * 100.0
    cvar_vals = comparison_df["CVaR"].to_numpy(dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    _styled_axes(ax)

    x = np.arange(len(methods))
    bar_width = 0.38

    var_bars = ax.bar(
        x - bar_width / 2,
        np.where(np.isfinite(var_vals), var_vals, 0.0),
        width=bar_width,
        color=_PRIMARY,
        label="VaR",
    )
    cvar_bars = ax.bar(
        x + bar_width / 2,
        np.where(np.isfinite(cvar_vals), cvar_vals, 0.0),
        width=bar_width,
        color=_DANGER,
        label="CVaR",
    )

    for bar, raw in zip(var_bars, var_vals):
        if not np.isfinite(raw):
            continue
        ax.annotate(
            f"{raw:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#222",
        )
    for bar, raw in zip(cvar_bars, cvar_vals):
        if not np.isfinite(raw):
            ax.annotate(
                "N/A",
                xy=(bar.get_x() + bar.get_width() / 2.0, 0.0),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#888",
            )
            continue
        ax.annotate(
            f"{raw:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Loss (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


# ─── CVaR optimization charts ─────────────────────────────────────────────


def plot_optimized_weights(
    weights: pd.Series,
    title: str = "Optimized Portfolio Weights",
    output_path: str | None = None,
) -> plt.Figure:
    """Bar chart of optimized weights (descending).

    Parameters
    ----------
    weights : pd.Series
        Indexed by asset, values are weights (may include CASH).
    title : str, optional
    output_path : str | None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    sorted_w = weights.sort_values(ascending=False)
    labels = [str(x) for x in sorted_w.index]
    values = sorted_w.to_numpy(dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    _styled_axes(ax)

    colors = [_DANGER if v < 0 else _PRIMARY for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="white")

    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 3 if v >= 0 else -12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#222",
        )

    ax.axhline(0.0, color="#444", linewidth=0.6)
    ax.set_ylabel("Weight (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_portfolio_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | None = None,
    title: str = "Current vs Optimized — Risk Comparison",
) -> plt.Figure:
    """Grouped bar chart comparing Expected Return, Volatility, VaR, CVaR
    across portfolios.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of
        :func:`var_cvar_crypto_risk.optimization.compare_current_vs_optimized`.
        Must include the ``Portfolio`` column and at least one of
        ``Expected Return``, ``Volatility``, ``VaR``, ``CVaR``.
    output_path : str | None, optional
    title : str, optional
    """
    metric_cols = [
        c
        for c in ("Expected Return", "Volatility", "VaR", "CVaR")
        if c in comparison_df.columns
    ]
    if not metric_cols:
        raise ValueError(
            "comparison_df must contain at least one of "
            "Expected Return / Volatility / VaR / CVaR."
        )

    portfolios = comparison_df["Portfolio"].astype(str).tolist()
    n_portfolios = len(portfolios)
    n_metrics = len(metric_cols)

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    _styled_axes(ax)

    x = np.arange(n_metrics)
    bar_width = 0.8 / max(n_portfolios, 1)

    palette = [
        "#1f3b73",
        "#c8102e",
        "#27ae60",
        "#f39c12",
        "#7a0017",
        "#5b5b5b",
    ]

    for i, label in enumerate(portfolios):
        row = comparison_df.iloc[i]
        values = [float(row[m]) * 100.0 for m in metric_cols]
        offset = (i - (n_portfolios - 1) / 2.0) * bar_width
        color = palette[i % len(palette)]
        bars = ax.bar(
            x + offset,
            np.where(np.isfinite(values), values, 0.0),
            width=bar_width,
            color=color,
            label=label,
            edgecolor="white",
        )
        for bar, v in zip(bars, values):
            if not np.isfinite(v):
                continue
            ax.annotate(
                f"{v:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols)
    ax.set_ylabel("Value (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_cvar_efficient_frontier(
    frontier_df: pd.DataFrame,
    output_path: str | None = None,
    title: str = "CVaR Efficient Frontier",
) -> plt.Figure:
    """Plot expected return vs CVaR for each feasible frontier point.

    Parameters
    ----------
    frontier_df : pd.DataFrame
        Output of
        :func:`var_cvar_crypto_risk.optimization.generate_cvar_efficient_frontier`.
        Must contain ``expected_return`` and ``CVaR`` columns.
    output_path : str | None, optional
    title : str, optional
    """
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    _styled_axes(ax)

    if (
        frontier_df.empty
        or "expected_return" not in frontier_df.columns
        or "CVaR" not in frontier_df.columns
    ):
        ax.text(
            0.5,
            0.5,
            "No feasible frontier points.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        _save_if_requested(fig, output_path)
        return fig

    df = frontier_df.copy()
    df = df.dropna(subset=["expected_return", "CVaR"])
    df = df.sort_values("CVaR")

    cvar_vals = df["CVaR"].to_numpy(dtype=float) * 100.0
    ret_vals = df["expected_return"].to_numpy(dtype=float) * 100.0

    ax.plot(cvar_vals, ret_vals, color=_PRIMARY, linewidth=1.4, alpha=0.85)
    ax.scatter(cvar_vals, ret_vals, color=_DANGER, s=35, zorder=3)

    if len(cvar_vals):
        # Highlight min-CVaR point.
        idx_min = int(np.argmin(cvar_vals))
        ax.scatter(
            cvar_vals[idx_min],
            ret_vals[idx_min],
            color="#27ae60",
            s=110,
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
            label="Minimum CVaR",
        )
        idx_max = int(np.argmax(ret_vals))
        ax.scatter(
            cvar_vals[idx_max],
            ret_vals[idx_max],
            color="#f39c12",
            s=110,
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
            label="Maximum Return",
        )

    ax.set_xlabel("CVaR (%)")
    ax.set_ylabel("Expected Return (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_allocation_comparison(
    weights_dict: dict,
    output_path: str | None = None,
    title: str = "Portfolio Allocation Comparison",
) -> plt.Figure:
    """Grouped bar chart of weights across multiple portfolios.

    Parameters
    ----------
    weights_dict : dict[str, pd.Series]
        ``{"Current": w_current, "Min CVaR": w_min, ...}``. All series
        should share the same asset index; missing assets are filled with 0.
    output_path : str | None, optional
    title : str, optional
    """
    if not weights_dict:
        raise ValueError("weights_dict must contain at least one portfolio.")

    all_assets: list[str] = []
    for series in weights_dict.values():
        for asset in series.index:
            if asset not in all_assets:
                all_assets.append(str(asset))

    portfolios = list(weights_dict.keys())
    n_portfolios = len(portfolios)
    n_assets = len(all_assets)

    fig, ax = plt.subplots(
        figsize=(max(8.0, 0.9 * n_assets + 3.0), 5.5), facecolor="white"
    )
    _styled_axes(ax)

    x = np.arange(n_assets)
    bar_width = 0.8 / max(n_portfolios, 1)
    palette = [
        "#1f3b73",
        "#c8102e",
        "#27ae60",
        "#f39c12",
        "#7a0017",
        "#5b5b5b",
    ]

    for i, label in enumerate(portfolios):
        series = weights_dict[label]
        values = [float(series.get(asset, 0.0)) * 100.0 for asset in all_assets]
        offset = (i - (n_portfolios - 1) / 2.0) * bar_width
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=palette[i % len(palette)],
            label=label,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_assets, rotation=15, ha="right")
    ax.set_ylabel("Weight (%)")
    ax.axhline(0.0, color="#444", linewidth=0.6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_model_comparison_backtest(
    comparison_df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure:
    """Grouped bar chart: actual vs expected breaches per VaR model.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of
        :func:`var_cvar_crypto_risk.backtesting.compare_var_models_backtest`.
        Rows where ``error`` is not None/NaN are skipped.
    output_path : str | None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if "error" in comparison_df.columns:
        valid = comparison_df[
            comparison_df["error"].isna() | (comparison_df["error"].isnull())
        ]
        if len(valid) == 0:
            valid = comparison_df[comparison_df["error"].apply(
                lambda v: v is None or (isinstance(v, float) and np.isnan(v))
            )]
    else:
        valid = comparison_df

    methods = valid["method"].astype(str).tolist()
    method_labels = [m.replace("_", " ").title() for m in methods]
    actual = valid["actual_breaches"].to_numpy(dtype=float)
    expected = valid["expected_breaches"].to_numpy(dtype=float)
    traffic_lights = valid["traffic_light"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    _styled_axes(ax)

    x = np.arange(len(methods))
    bar_width = 0.38

    actual_bars = ax.bar(
        x - bar_width / 2,
        actual,
        width=bar_width,
        color=_PRIMARY,
        label="Actual Breaches",
    )
    expected_bars = ax.bar(
        x + bar_width / 2,
        expected,
        width=bar_width,
        color="#a0a0a0",
        label="Expected Breaches",
    )

    for bar, status in zip(actual_bars, traffic_lights):
        edge = _TRAFFIC_LIGHT_EDGE.get(status, "#888")
        bar.set_edgecolor(edge)
        bar.set_linewidth(2)

    for bar in list(actual_bars) + list(expected_bars):
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel("Number of Breaches")
    ax.set_title(
        "Model Comparison — Actual vs Expected Breaches",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


# ─── Breach-rate, distribution, asset-level, and correlation charts ──────


def plot_rolling_breach_rate(
    rolling_breach_rate: pd.Series,
    expected_breach_rate: float,
    method: str,
    output_path: str | None = None,
) -> plt.Figure:
    """Rolling breach rate over time vs the expected breach rate."""
    confidence = 1.0 - float(expected_breach_rate)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    _styled_axes(ax)

    ax.plot(
        rolling_breach_rate.index,
        rolling_breach_rate.values * 100.0,
        color=_PRIMARY,
        linewidth=1.6,
        label="Rolling breach rate",
    )
    ax.axhline(
        expected_breach_rate * 100.0,
        color=_DANGER,
        linestyle="--",
        linewidth=1.6,
        label=f"Expected = {expected_breach_rate * 100:.2f}%",
    )

    method_label = str(method).replace("_", " ").title()
    ax.set_title(
        f"Rolling Breach Rate — {method_label} ({confidence * 100:.0f}% confidence)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Breach rate (%)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_asset_return_distributions(
    asset_returns: pd.DataFrame,
    horizon_days: int = 1,
    confidence_level: float = 0.95,
    return_method: str = "simple",
    output_path: str | None = None,
) -> plt.Figure:
    """Small-multiples return histograms per asset with historical VaR/CVaR."""
    from .returns import calculate_horizon_returns

    assets = list(asset_returns.columns)
    n = len(assets)
    ncols = min(3, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    alpha = 1.0 - float(confidence_level)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), facecolor="white"
    )
    axes = np.atleast_1d(axes).ravel()

    horizon_label = "Daily" if horizon_days == 1 else f"{horizon_days}-day"
    for i, asset in enumerate(assets):
        ax = axes[i]
        _styled_axes(ax)
        series = asset_returns[asset].dropna()
        if horizon_days > 1:
            series = calculate_horizon_returns(
                series, horizon_days=horizon_days, method=return_method
            )
        clean = series.to_numpy(dtype=float)
        ax.hist(clean, bins=40, color=_PRIMARY, edgecolor="white", alpha=0.85)
        if clean.size:
            var_threshold = float(np.quantile(clean, alpha))
            tail = clean[clean <= var_threshold]
            cvar_threshold = float(tail.mean()) if tail.size else var_threshold
            ax.axvline(
                var_threshold, color=_DANGER, linestyle="--", linewidth=1.5,
                label=(
                    f"VaR {confidence_level * 100:.0f}%: "
                    f"{-var_threshold * 100:.1f}%"
                ),
            )
            ax.axvline(
                cvar_threshold, color="#7a0017", linestyle=":", linewidth=1.5,
                label=(
                    f"CVaR {confidence_level * 100:.0f}%: "
                    f"{-cvar_threshold * 100:.1f}%"
                ),
            )
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.set_title(asset, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{horizon_label} Return")
        ax.set_ylabel("Frequency")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Asset-level {horizon_label} Return Distributions",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_qq_vs_normal(
    returns: pd.Series,
    title: str = "QQ Plot vs Normal",
    output_path: str | None = None,
) -> plt.Figure:
    """Quantile-quantile plot of returns against a Normal reference."""
    clean = returns.dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    _styled_axes(ax)

    stats.probplot(clean, dist="norm", plot=ax)
    # Restyle the points and reference line probplot draws.
    if len(ax.get_lines()) >= 2:
        ax.get_lines()[0].set(marker="o", markersize=3, color=_PRIMARY, alpha=0.6)
        ax.get_lines()[1].set(color=_DANGER, linewidth=1.8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Theoretical Quantiles (Normal)")
    ax.set_ylabel("Sample Quantiles")
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_tail_zoom_distribution(
    returns: pd.Series,
    var_value: float,
    cvar_value: float,
    confidence_level: float = 0.95,
    output_path: str | None = None,
    tail_quantile: float = 0.15,
) -> plt.Figure:
    """Zoom on the left (loss) tail of the return distribution."""
    clean = returns.dropna().to_numpy(dtype=float)
    threshold = float(np.quantile(clean, tail_quantile)) if clean.size else 0.0
    tail = clean[clean <= threshold]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _styled_axes(ax)

    if tail.size:
        ax.hist(tail, bins=30, color=_DANGER, edgecolor="white", alpha=0.7)
    ax.axvline(
        loss_value_to_return_threshold(var_value),
        color=_PRIMARY,
        linestyle="--",
        linewidth=1.8,
        label=f"VaR {confidence_level * 100:.0f}% = {var_value * 100:.2f}%",
    )
    ax.axvline(
        loss_value_to_return_threshold(cvar_value),
        color="#7a0017",
        linestyle=":",
        linewidth=1.8,
        label=f"CVaR {confidence_level * 100:.0f}% = {cvar_value * 100:.2f}%",
    )

    ax.set_title(
        f"Left-Tail Zoom (worst {tail_quantile * 100:.0f}%) — "
        f"{confidence_level * 100:.0f}% confidence",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Return (loss tail)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_asset_cumulative_returns(
    asset_returns: pd.DataFrame,
    title: str = "Asset Cumulative Returns",
    output_path: str | None = None,
) -> plt.Figure:
    """Per-asset cumulative return curves (growth from 0%)."""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    _styled_axes(ax)

    cumulative = (1.0 + asset_returns).cumprod() - 1.0
    for col in cumulative.columns:
        ax.plot(cumulative.index, cumulative[col].values * 100.0, linewidth=1.5, label=str(col))
    ax.axhline(0, color="#444", linewidth=0.8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_asset_drawdowns(
    asset_drawdowns: pd.DataFrame,
    title: str = "Asset Drawdowns",
    output_path: str | None = None,
) -> plt.Figure:
    """Per-asset drawdown curves over time."""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    _styled_axes(ax)

    for col in asset_drawdowns.columns:
        ax.plot(
            asset_drawdowns.index,
            asset_drawdowns[col].values * 100.0,
            linewidth=1.3,
            label=str(col),
        )
    ax.axhline(0, color="#444", linewidth=0.8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Asset Return Correlation Matrix",
    output_path: str | None = None,
) -> plt.Figure:
    """Correlation heatmap with in-cell value annotations (matplotlib only)."""
    labels = list(corr_matrix.columns)
    data = corr_matrix.to_numpy(dtype=float)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(0.9 * n + 3, 0.9 * n + 2.5), facecolor="white")
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            value = data[i, j]
            text_color = "white" if abs(value) > 0.6 else "#222"
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center",
                color=text_color, fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def plot_rolling_average_correlation(
    rolling_corr: pd.Series,
    title: str = "Rolling Average Pairwise Correlation",
    output_path: str | None = None,
) -> plt.Figure:
    """Average pairwise correlation through time."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    _styled_axes(ax)

    ax.plot(rolling_corr.index, rolling_corr.values, color=_PRIMARY, linewidth=1.6)
    ax.axhline(0, color="#444", linewidth=0.8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average pairwise correlation")
    ax.set_ylim(-1.0, 1.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig
