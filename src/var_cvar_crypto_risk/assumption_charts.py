"""Pure Plotly audit charts for completed robust-assumption outputs.

The builders in this module consume the transparency table produced by the
Robust Assumptions Engine. They never estimate returns, apply manual views, or
alter the vector passed downstream to optimization.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .assumptions import AssumptionConfig
from .plotly_theme import COLORS, apply_risk_theme


EXPECTED_RETURN_COMPARISONS: dict[str, str] = {
    "median": "Median",
    "trimmed_mean": "Trimmed Mean",
    "winsorized_mean": "Winsorized Mean",
    "shrinkage_to_zero": "Shrinkage Estimate",
    "manual_view": "Manual View",
    "final_expected_return": "Final E[r]",
}

EXPECTED_RETURN_VIEW_MODES: dict[str, str] = {
    "all_estimators": "All Estimators",
    "pairwise": "Pairwise Comparison",
}
DEFAULT_EXPECTED_RETURN_VIEW_MODE = "all_estimators"

EXPECTED_RETURN_ESTIMATORS: dict[str, str] = {
    "mean": "Raw Historical Mean",
    "median": "Median",
    "trimmed_mean": "Trimmed Mean",
    "winsorized_mean": "Winsorized Mean",
    "shrinkage_to_zero": "Shrinkage Estimate",
    "manual_view": "Manual View",
    "final_expected_return": "Final E[r]",
}

_MARKER_CONTRACT: dict[str, dict[str, Any]] = {
    "mean": {
        "symbol": "circle-open",
        "size": 15,
        "color": COLORS["neutral"],
        "line": {"color": COLORS["neutral"], "width": 3},
    },
    "median": {
        "symbol": "diamond",
        "size": 10,
        "color": COLORS["orange"],
        "line": {"color": COLORS["dark"], "width": 1},
    },
    "trimmed_mean": {
        "symbol": "square",
        "size": 10,
        "color": COLORS["primary"],
        "line": {"color": COLORS["dark"], "width": 1},
    },
    "winsorized_mean": {
        "symbol": "triangle-up",
        "size": 11,
        "color": COLORS["purple"],
        "line": {"color": COLORS["dark"], "width": 1},
    },
    "shrinkage_to_zero": {
        "symbol": "cross",
        "size": 11,
        "color": COLORS["success"],
        "line": {"color": COLORS["success"], "width": 2},
    },
    "manual_view": {
        "symbol": "hexagon",
        "size": 11,
        "color": COLORS["warning"],
        "line": {"color": COLORS["dark"], "width": 1},
    },
    "final_expected_return": {
        "symbol": "star",
        "size": 16,
        "color": COLORS["danger"],
        "line": {"color": COLORS["dark"], "width": 1.5},
    },
}

_VERTICAL_OFFSETS = {
    "mean": -0.27,
    "median": -0.18,
    "trimmed_mean": -0.09,
    "winsorized_mean": 0.0,
    "shrinkage_to_zero": 0.09,
    "manual_view": 0.18,
    "final_expected_return": 0.27,
}

_REQUIRED_COLUMNS = {
    "mean",
    "median",
    "trimmed_mean",
    "winsorized_mean",
    "shrinkage_to_zero",
    "manual_view",
    "final_expected_return",
}

_ESTIMATOR_LABELS = {
    "mean": "Historical Mean",
    "median": "Median",
    "trimmed_mean": "Trimmed Mean",
    "winsorized_mean": "Winsorized Mean",
    "shrinkage_to_zero": "Shrinkage Estimate",
    "zero": "Zero",
}


def _format_return(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.4%}"


def _active_estimator_label(config: AssumptionConfig) -> str:
    label = _ESTIMATOR_LABELS.get(
        config.expected_return_method,
        str(config.expected_return_method).replace("_", " ").title(),
    )
    if config.manual_views:
        label += f" + Manual View blend ({config.view_blend_weight:.0%})"
    return label


def _estimator_configurations(config: AssumptionConfig) -> dict[str, str]:
    return {
        "mean": "Arithmetic mean of the historical scenario sample",
        "median": "Median of the historical scenario sample",
        "trimmed_mean": f"{config.trim_proportion:.1%} removed from each tail",
        "winsorized_mean": (
            f"Clipped at the {config.winsor_proportion:.1%} and "
            f"{1.0 - config.winsor_proportion:.1%} quantiles"
        ),
        "shrinkage_to_zero": (
            f"{config.shrinkage_weight:.1%} historical mean + "
            f"{1.0 - config.shrinkage_weight:.1%} zero prior"
        ),
        "manual_view": (
            f"Point view; {config.view_blend_weight:.1%} blend weight is applied "
            "only when constructing Final E[r]"
        ),
        "final_expected_return": (
            "Exact downstream model input after the active estimator and any "
            "manual-view blend"
        ),
    }


def _estimator_values(row: pd.Series) -> dict[str, float | None]:
    return {
        column: float(row[column]) if pd.notna(row[column]) else None
        for column in EXPECTED_RETURN_ESTIMATORS
    }


def _estimator_dispersion(row: pd.Series) -> float:
    values = [value for value in _estimator_values(row).values() if value is not None]
    return float(max(values) - min(values))


def _ordered_assets_by_dispersion(
    table: pd.DataFrame,
    *,
    asset_order: Sequence[str] | None,
    sort_by_dispersion: bool,
) -> list[str]:
    available = [str(asset) for asset in table.index]
    if asset_order is None:
        ordered = available
    else:
        requested = [str(asset) for asset in asset_order]
        ordered = [asset for asset in requested if asset in table.index]
        ordered.extend(asset for asset in available if asset not in ordered)
    if not sort_by_dispersion:
        return ordered
    positions = {asset: position for position, asset in enumerate(ordered)}
    return sorted(
        ordered,
        key=lambda asset: (-_estimator_dispersion(table.loc[asset]), positions[asset]),
    )


def _validate_chart_inputs(
    assumption_table: pd.DataFrame, horizon_days: int
) -> tuple[pd.DataFrame, int]:
    if not isinstance(assumption_table, pd.DataFrame) or assumption_table.empty:
        raise ValueError("assumption_table must be a non-empty DataFrame")
    missing = sorted(_REQUIRED_COLUMNS.difference(assumption_table.columns))
    if missing:
        raise ValueError(f"assumption_table is missing columns: {missing}")
    horizon = int(horizon_days)
    if horizon < 1:
        raise ValueError("horizon_days must be at least 1")
    if assumption_table.index.has_duplicates:
        raise ValueError("assumption_table index must contain unique assets")
    table = assumption_table.copy(deep=True)
    table.index = table.index.map(str)
    return table, horizon


def _ordered_assets(
    table: pd.DataFrame,
    *,
    asset_order: Sequence[str] | None,
    comparison_column: str,
    sort_by_adjustment: bool,
) -> list[str]:
    available = [str(asset) for asset in table.index]
    if asset_order is None:
        ordered = available
    else:
        requested = [str(asset) for asset in asset_order]
        ordered = [asset for asset in requested if asset in table.index]
        ordered.extend(asset for asset in available if asset not in ordered)
    if not sort_by_adjustment:
        return ordered

    magnitudes: dict[str, float] = {}
    for asset in ordered:
        comparison = table.at[asset, comparison_column]
        magnitudes[asset] = (
            abs(float(comparison) - float(table.at[asset, "mean"]))
            if pd.notna(comparison)
            else -np.inf
        )
    return sorted(ordered, key=lambda asset: magnitudes[asset], reverse=True)


def build_expected_return_dumbbell(
    assumption_table: pd.DataFrame,
    config: AssumptionConfig,
    *,
    comparison: str = "final_expected_return",
    horizon_days: int = 1,
    asset_order: Sequence[str] | None = None,
    sort_by_adjustment: bool = False,
    sort_by_dispersion: bool = False,
) -> go.Figure:
    """Compare raw mean returns with one completed robust estimate.

    ``assumption_table`` must be the exact table returned by
    :func:`build_assumption_table`. Values are read only; no estimator is
    recomputed in this function.
    """
    table, horizon = _validate_chart_inputs(assumption_table, horizon_days)
    if comparison not in EXPECTED_RETURN_COMPARISONS:
        raise ValueError(
            f"Unsupported comparison '{comparison}'. Choose from "
            f"{list(EXPECTED_RETURN_COMPARISONS)}."
        )
    comparison_label = EXPECTED_RETURN_COMPARISONS[comparison]
    if sort_by_dispersion:
        assets = _ordered_assets_by_dispersion(
            table,
            asset_order=asset_order,
            sort_by_dispersion=True,
        )
    else:
        assets = _ordered_assets(
            table,
            asset_order=asset_order,
            comparison_column=comparison,
            sort_by_adjustment=sort_by_adjustment,
        )

    active_estimator = _active_estimator_label(config)
    trim_config = f"{config.trim_proportion:.1%} from each tail"
    winsor_config = (
        f"{config.winsor_proportion:.1%} / "
        f"{1.0 - config.winsor_proportion:.1%} quantile limits"
    )
    shrinkage_config = (
        f"{config.shrinkage_weight:.1%} × historical mean + "
        f"{1.0 - config.shrinkage_weight:.1%} × zero prior"
    )

    raw_points: list[float] = []
    comparison_points: list[float | None] = []
    connector_x: list[float | None] = []
    connector_y: list[str | None] = []
    hover_rows: list[list[str]] = []
    audit_rows: list[dict[str, Any]] = []
    equal_assets: list[str] = []
    unavailable_assets: list[str] = []

    for asset in assets:
        row = table.loc[asset]
        raw = float(row["mean"])
        selected_raw = row[comparison]
        selected = float(selected_raw) if pd.notna(selected_raw) else None
        difference = selected - raw if selected is not None else None
        raw_percent = raw * 100.0
        selected_percent = selected * 100.0 if selected is not None else None
        raw_points.append(raw_percent)
        comparison_points.append(selected_percent)

        if selected_percent is None:
            connector_x.extend([None, None, None])
            connector_y.extend([asset, asset, None])
            unavailable_assets.append(asset)
        else:
            connector_x.extend([raw_percent, selected_percent, None])
            connector_y.extend([asset, asset, None])
            if np.isclose(raw, selected, rtol=0.0, atol=1e-14):
                equal_assets.append(asset)

        hover_rows.append(
            [
                asset,
                _format_return(raw),
                _format_return(selected),
                (
                    f"{abs(difference) * 100.0:.4f} percentage points"
                    if difference is not None
                    else "N/A"
                ),
                f"{difference * 10_000.0:+.2f} bps"
                if difference is not None
                else "N/A",
                _format_return(row["median"]),
                _format_return(row["trimmed_mean"]),
                trim_config,
                _format_return(row["winsorized_mean"]),
                winsor_config,
                _format_return(row["shrinkage_to_zero"]),
                shrinkage_config,
                _format_return(row["manual_view"]),
                _format_return(row["final_expected_return"]),
                active_estimator,
            ]
        )
        audit_rows.append(
            {
                "asset": asset,
                "raw_historical_mean": raw,
                "comparison_value": selected,
                "absolute_difference": abs(difference)
                if difference is not None
                else None,
                "difference_basis_points": difference * 10_000.0
                if difference is not None
                else None,
                "manual_view_available": bool(pd.notna(row["manual_view"])),
                "final_expected_return": float(row["final_expected_return"]),
            }
        )

    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "Raw historical mean: %{customdata[1]}<br>"
        f"{comparison_label}: %{{customdata[2]}}<br>"
        "Absolute difference: %{customdata[3]}<br>"
        "Difference: %{customdata[4]}<br>"
        "Median: %{customdata[5]}<br>"
        "Trimmed mean: %{customdata[6]}<br>"
        "Trim used: %{customdata[7]}<br>"
        "Winsorized mean: %{customdata[8]}<br>"
        "Winsorization limits: %{customdata[9]}<br>"
        "Shrinkage estimate: %{customdata[10]}<br>"
        "Shrinkage configuration: %{customdata[11]}<br>"
        "Manual view: %{customdata[12]}<br>"
        "Final E[r]: %{customdata[13]}<br>"
        "Active estimator: %{customdata[14]}<extra></extra>"
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=connector_x,
            y=connector_y,
            mode="lines",
            name="Adjustment",
            line={"color": COLORS["neutral"], "width": 3},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=raw_points,
            y=assets,
            mode="markers",
            name="Raw Historical Mean",
            marker={
                "symbol": "circle-open",
                "size": 16,
                "color": COLORS["neutral"],
                "line": {"color": COLORS["neutral"], "width": 3},
            },
            customdata=hover_rows,
            hovertemplate=hover_template,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=comparison_points,
            y=assets,
            mode="markers",
            name=comparison_label,
            marker={
                "symbol": "diamond",
                "size": 10,
                "color": COLORS["orange"],
                "line": {"color": COLORS["dark"], "width": 1},
            },
            customdata=hover_rows,
            hovertemplate=hover_template,
        )
    )

    numeric_points = [0.0, *raw_points]
    numeric_points.extend(value for value in comparison_points if value is not None)
    largest = max(abs(value) for value in numeric_points)
    padding = max(0.05, largest * 0.18)
    x_range = [-(largest + padding), largest + padding]

    def _annotation_layout(x_value: float) -> dict[str, Any]:
        """Keep equality/unavailable labels inside the visible x-range."""
        range_width = x_range[1] - x_range[0]
        near_right_edge = x_value > x_range[0] + 0.72 * range_width
        return {
            "xanchor": "right" if near_right_edge else "left",
            "xshift": -16 if near_right_edge else 16,
        }

    for asset in equal_assets:
        x_value = float(table.at[asset, "mean"]) * 100.0
        figure.add_annotation(
            x=x_value,
            y=asset,
            text="No adjustment",
            showarrow=False,
            font={"size": 11, "color": COLORS["neutral"]},
            **_annotation_layout(x_value),
        )
    for asset in unavailable_assets:
        x_value = float(table.at[asset, "mean"]) * 100.0
        figure.add_annotation(
            x=x_value,
            y=asset,
            text=f"{comparison_label}: N/A",
            showarrow=False,
            font={"size": 11, "color": COLORS["warning"]},
            **_annotation_layout(x_value),
        )

    horizon_label = f"{horizon}-Day"
    endpoint_title = (
        "Final Robust Assumption"
        if comparison == "final_expected_return"
        else comparison_label
    )
    apply_risk_theme(
        figure,
        title=f"Historical Mean vs {endpoint_title} — {horizon_label} Return",
        x_title=f"{horizon_label} Return Assumption (%)",
        y_title="Asset",
        height=max(390, 68 * len(assets) + 190),
        meta={
            "view_mode": "pairwise",
            "comparison_column": comparison,
            "comparison_label": comparison_label,
            "horizon_days": horizon,
            "sort_by_adjustment": bool(sort_by_adjustment),
            "sort_by_dispersion": bool(sort_by_dispersion),
            "asset_order": assets,
            "audit_rows": audit_rows,
            "active_expected_return_estimator": active_estimator,
            "estimator_values_by_asset": {
                asset: _estimator_values(table.loc[asset]) for asset in assets
            },
            "final_expected_return_by_asset": {
                asset: float(table.at[asset, "final_expected_return"])
                for asset in assets
            },
            "estimator_dispersion_by_asset": {
                asset: _estimator_dispersion(table.loc[asset]) for asset in assets
            },
            "trim_proportion": float(config.trim_proportion),
            "winsorization_proportion": float(config.winsor_proportion),
            "shrinkage_weight": float(config.shrinkage_weight),
            "manual_view_blend_weight": float(config.view_blend_weight),
        },
    )
    figure.add_vline(
        x=0.0,
        line={"color": COLORS["dark"], "width": 2, "dash": "dash"},
    )
    figure.update_xaxes(
        range=x_range,
        ticksuffix="%",
        tickformat=".2f",
        zeroline=False,
    )
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(assets)),
    )
    return figure


def build_expected_return_estimator_comparison(
    assumption_table: pd.DataFrame,
    config: AssumptionConfig,
    *,
    horizon_days: int = 1,
    asset_order: Sequence[str] | None = None,
    sort_by_dispersion: bool = False,
) -> go.Figure:
    """Display every completed return estimate without recalculating it.

    The input is the exact transparency table returned by the assumptions
    engine. Decimal returns are converted to percentage points only for the
    Plotly x-axis. Small deterministic y-offsets keep coincident estimates
    inspectable while leaving every x-value unchanged.
    """
    table, horizon = _validate_chart_inputs(assumption_table, horizon_days)
    assets = _ordered_assets_by_dispersion(
        table,
        asset_order=asset_order,
        sort_by_dispersion=sort_by_dispersion,
    )
    active_estimator = _active_estimator_label(config)
    configurations = _estimator_configurations(config)
    estimator_values = {asset: _estimator_values(table.loc[asset]) for asset in assets}
    dispersions = {asset: _estimator_dispersion(table.loc[asset]) for asset in assets}
    differences_bps = {
        asset: {
            column: (
                None
                if value is None
                else (value - estimator_values[asset]["mean"]) * 10_000.0
            )
            for column, value in estimator_values[asset].items()
        }
        for asset in assets
    }
    base_y = {asset: float(position) for position, asset in enumerate(assets)}

    figure = go.Figure()
    connector_x: list[float | None] = []
    connector_y: list[float | None] = []
    for asset in assets:
        available = [
            value for value in estimator_values[asset].values() if value is not None
        ]
        connector_x.extend([min(available) * 100.0, max(available) * 100.0, None])
        connector_y.extend([base_y[asset], base_y[asset], None])
    figure.add_trace(
        go.Scatter(
            x=connector_x,
            y=connector_y,
            mode="lines",
            name="Estimator Range",
            line={"color": COLORS["neutral"], "width": 2},
            opacity=0.48,
            hoverinfo="skip",
            showlegend=False,
        )
    )

    for column, label in EXPECTED_RETURN_ESTIMATORS.items():
        x_values: list[float] = []
        y_values: list[float] = []
        customdata: list[list[Any]] = []
        for asset in assets:
            value = estimator_values[asset][column]
            if value is None:
                continue
            raw_mean = float(estimator_values[asset]["mean"])
            difference = value - raw_mean
            overlaps = [
                EXPECTED_RETURN_ESTIMATORS[other]
                for other, other_value in estimator_values[asset].items()
                if other != column
                and other_value is not None
                and np.isclose(value, other_value, rtol=0.0, atol=1e-14)
            ]
            x_values.append(value * 100.0)
            y_values.append(base_y[asset] + _VERTICAL_OFFSETS[column])
            customdata.append(
                [
                    asset,
                    label,
                    _format_return(value),
                    f"{difference * 100.0:+.4f} percentage points",
                    f"{difference * 10_000.0:+.2f} bps",
                    active_estimator,
                    "Yes — downstream model input"
                    if column == "final_expected_return"
                    else "No",
                    f"{horizon}-day",
                    configurations[column],
                    _format_return(estimator_values[asset]["manual_view"]),
                    ", ".join(overlaps) if overlaps else "None",
                ]
            )
        if not x_values:
            continue
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=label,
                marker=dict(_MARKER_CONTRACT[column]),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Estimator: %{customdata[1]}<br>"
                    "Value: %{customdata[2]}<br>"
                    "Difference from raw mean: %{customdata[3]}<br>"
                    "Difference from raw mean: %{customdata[4]}<br>"
                    "Active estimator: %{customdata[5]}<br>"
                    "Final E[r]: %{customdata[6]}<br>"
                    "Horizon: %{customdata[7]}<br>"
                    "Configuration: %{customdata[8]}<br>"
                    "Manual View: %{customdata[9]}<br>"
                    "Same x-value as: %{customdata[10]}<extra></extra>"
                ),
            )
        )

    numeric_points = [0.0]
    numeric_points.extend(
        value * 100.0
        for values in estimator_values.values()
        for value in values.values()
        if value is not None
    )
    minimum = min(numeric_points)
    maximum = max(numeric_points)
    span = maximum - minimum
    padding = max(0.05, span * 0.12, max(abs(minimum), abs(maximum)) * 0.08)

    marker_meta = {
        column: {
            "label": EXPECTED_RETURN_ESTIMATORS[column],
            "symbol": contract["symbol"],
            "color": contract["color"],
            "vertical_offset": _VERTICAL_OFFSETS[column],
        }
        for column, contract in _MARKER_CONTRACT.items()
    }
    apply_risk_theme(
        figure,
        title=f"Return Estimator Comparison — {horizon}-Day Assumptions",
        x_title=f"{horizon}-Day Return Assumption (%)",
        y_title="Asset",
        height=max(520, 86 * len(assets) + 270),
        meta={
            "view_mode": "all_estimators",
            "asset_order": assets,
            "horizon_days": horizon,
            "active_expected_return_estimator": active_estimator,
            "estimator_values_by_asset": estimator_values,
            "final_expected_return_by_asset": {
                asset: float(table.at[asset, "final_expected_return"])
                for asset in assets
            },
            "estimator_dispersion_by_asset": dispersions,
            "estimator_differences_basis_points_by_asset": differences_bps,
            "manual_view_by_asset": {
                asset: (
                    "N/A"
                    if estimator_values[asset]["manual_view"] is None
                    else estimator_values[asset]["manual_view"]
                )
                for asset in assets
            },
            "trim_proportion": float(config.trim_proportion),
            "winsorization_proportion": float(config.winsor_proportion),
            "shrinkage_weight": float(config.shrinkage_weight),
            "manual_view_blend_weight": float(config.view_blend_weight),
            "marker_contract": marker_meta,
            "overlap_handling": "deterministic vertical offsets; x-values unchanged",
            "sort_by_dispersion": bool(sort_by_dispersion),
        },
    )
    # The six-to-seven item legend wraps at Streamlit's normal content width.
    # Reserve a dedicated header band so neither legend row can collide with
    # the chart title or the first asset row.
    figure.update_layout(
        margin={"l": 64, "r": 28, "t": 150, "b": 58},
        title={
            "text": f"Return Estimator Comparison — {horizon}-Day Assumptions",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
        },
        legend={
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "top",
            "font": {"size": 12},
        },
    )
    figure.add_vline(
        x=0.0,
        line={"color": COLORS["dark"], "width": 2, "dash": "dash"},
    )
    figure.update_xaxes(
        range=[minimum - padding, maximum + padding],
        ticksuffix="%",
        tickformat=".2f",
        zeroline=False,
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=[base_y[asset] for asset in assets],
        ticktext=assets,
        range=[len(assets) - 0.55, -0.55],
        showgrid=False,
        zeroline=False,
    )
    return figure


__all__ = [
    "DEFAULT_EXPECTED_RETURN_VIEW_MODE",
    "EXPECTED_RETURN_COMPARISONS",
    "EXPECTED_RETURN_ESTIMATORS",
    "EXPECTED_RETURN_VIEW_MODES",
    "build_expected_return_dumbbell",
    "build_expected_return_estimator_comparison",
]
