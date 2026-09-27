"""Shared, Streamlit-independent Plotly styling for Risk Lab figures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go


COLORS = {
    "primary": "#4C78A8",
    "secondary": "#72B7B2",
    "danger": "#E45756",
    "warning": "#F2CF5B",
    "success": "#54A24B",
    "purple": "#B279A2",
    "orange": "#F58518",
    "neutral": "#A0AEC0",
    "dark": "#E2E8F0",
}

PALETTE = [
    COLORS["primary"],
    COLORS["danger"],
    COLORS["success"],
    COLORS["orange"],
    COLORS["purple"],
    COLORS["secondary"],
    COLORS["warning"],
    COLORS["neutral"],
]


def apply_risk_theme(
    figure: go.Figure,
    *,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    meta: Mapping[str, Any] | None = None,
    hovermode: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Apply one accessible, dark-theme-compatible visual contract."""
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        font={"family": "Inter, Arial, sans-serif", "size": 13},
        margin={"l": 64, "r": 28, "t": 70, "b": 58},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode=hovermode,
        meta=dict(meta or {}),
        height=height,
    )
    figure.update_xaxes(
        title_text=x_title,
        showgrid=True,
        gridcolor="rgba(160,174,192,0.18)",
        zeroline=True,
        zerolinecolor="rgba(226,232,240,0.45)",
        automargin=True,
    )
    figure.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor="rgba(160,174,192,0.18)",
        zeroline=True,
        zerolinecolor="rgba(226,232,240,0.45)",
        automargin=True,
    )
    return figure


def empty_figure(
    message: str,
    *,
    title: str,
    meta: Mapping[str, Any] | None = None,
) -> go.Figure:
    """Return a useful Plotly empty state rather than an ambiguous blank chart."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": COLORS["neutral"]},
    )
    return apply_risk_theme(figure, title=title, meta=meta, height=360)


__all__ = ["COLORS", "PALETTE", "apply_risk_theme", "empty_figure"]
