"""Contract tests for Robust Assumptions audit charts."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from var_cvar_crypto_risk import assumption_charts
from var_cvar_crypto_risk.assumption_charts import (
    DEFAULT_EXPECTED_RETURN_VIEW_MODE,
    EXPECTED_RETURN_ESTIMATORS,
    EXPECTED_RETURN_VIEW_MODES,
    build_expected_return_dumbbell,
    build_expected_return_estimator_comparison,
)
from var_cvar_crypto_risk.assumptions import (
    AssumptionConfig,
    build_assumption_table,
)


@pytest.fixture
def returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BTC": [0.04, -0.01, 0.02, 0.03, -0.02],
            "ETH": [-0.03, -0.02, 0.01, -0.01, 0.00],
            "SOL": [0.10, -0.08, 0.04, -0.03, 0.02],
        }
    )


@pytest.fixture
def robust_config() -> AssumptionConfig:
    return AssumptionConfig(
        expected_return_method="trimmed_mean",
        trim_proportion=0.10,
        winsor_proportion=0.05,
        shrinkage_weight=0.40,
        manual_views={"ETH": -0.005},
        view_blend_weight=0.50,
    )


def _trace(figure: go.Figure, name: str) -> go.Scatter:
    return next(trace for trace in figure.data if trace.name == name)


def _trace_assets(trace: go.Scatter) -> list[str]:
    return [str(row[0]) for row in trace.customdata]


def test_all_estimators_is_the_default_view_mode() -> None:
    assert DEFAULT_EXPECTED_RETURN_VIEW_MODE == "all_estimators"
    assert list(EXPECTED_RETURN_VIEW_MODES)[0] == DEFAULT_EXPECTED_RETURN_VIEW_MODE
    assert EXPECTED_RETURN_VIEW_MODES[DEFAULT_EXPECTED_RETURN_VIEW_MODE] == (
        "All Estimators"
    )


def test_all_available_estimators_appear_for_each_asset(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    figure = build_expected_return_estimator_comparison(table, robust_config)

    assert isinstance(figure, go.Figure)
    for column, label in EXPECTED_RETURN_ESTIMATORS.items():
        trace = _trace(figure, label)
        expected_assets = list(table.index[table[column].notna()])
        assert _trace_assets(trace) == expected_assets


def test_all_estimator_marker_values_match_transparency_table(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    figure = build_expected_return_estimator_comparison(table, robust_config)

    for column, label in EXPECTED_RETURN_ESTIMATORS.items():
        trace = _trace(figure, label)
        for asset, value in zip(_trace_assets(trace), trace.x):
            assert value / 100.0 == pytest.approx(table.at[asset, column])


def test_all_estimator_final_values_match_downstream_output(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    downstream = robust_config.final_expected_returns(returns)
    figure = build_expected_return_estimator_comparison(table, robust_config)
    final = _trace(figure, "Final E[r]")

    np.testing.assert_allclose(
        np.asarray(final.x, dtype=float) / 100.0,
        downstream.reindex(_trace_assets(final)).to_numpy(),
    )


def test_all_estimator_missing_manual_view_is_omitted_not_zero(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    figure = build_expected_return_estimator_comparison(table, robust_config)
    manual = _trace(figure, "Manual View")

    assert _trace_assets(manual) == ["ETH"]
    assert list(manual.x) == pytest.approx([-0.5])
    assert figure.layout.meta["manual_view_by_asset"]["BTC"] == "N/A"
    assert figure.layout.meta["estimator_values_by_asset"]["BTC"]["manual_view"] is None


def test_all_estimator_view_is_immutable_and_pairwise_still_works(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    original = table.copy(deep=True)
    all_estimators = build_expected_return_estimator_comparison(table, robust_config)
    pairwise = build_expected_return_dumbbell(
        table, robust_config, comparison="winsorized_mean"
    )

    pd.testing.assert_frame_equal(table, original)
    assert all_estimators.layout.meta["view_mode"] == "all_estimators"
    assert pairwise.layout.meta["view_mode"] == "pairwise"
    np.testing.assert_allclose(
        _trace(pairwise, "Winsorized Mean").x,
        table["winsorized_mean"].to_numpy() * 100.0,
    )


def test_estimator_dispersion_sorting_uses_full_available_range(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    table.loc["BTC", list(EXPECTED_RETURN_ESTIMATORS)] = [
        0.00,
        0.10,
        0.01,
        0.02,
        0.03,
        np.nan,
        0.00,
    ]
    table.loc["ETH", list(EXPECTED_RETURN_ESTIMATORS)] = [
        0.00,
        0.01,
        0.02,
        0.01,
        0.00,
        0.01,
        0.00,
    ]
    table.loc["SOL", list(EXPECTED_RETURN_ESTIMATORS)] = [
        0.00,
        0.04,
        -0.02,
        0.01,
        0.00,
        np.nan,
        0.00,
    ]
    original = table.copy(deep=True)
    figure = build_expected_return_estimator_comparison(
        table,
        robust_config,
        asset_order=["SOL", "ETH", "BTC"],
        sort_by_dispersion=True,
    )

    assert figure.layout.meta["asset_order"] == ["BTC", "SOL", "ETH"]
    assert figure.layout.meta["estimator_dispersion_by_asset"] == pytest.approx(
        {"BTC": 0.10, "SOL": 0.06, "ETH": 0.02}
    )
    pd.testing.assert_frame_equal(table, original)


def test_negative_zero_and_positive_estimates_render_exactly(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    table.loc[:, "median"] = [-0.01, 0.0, 0.01]
    figure = build_expected_return_estimator_comparison(table, robust_config)
    median = _trace(figure, "Median")

    np.testing.assert_allclose(median.x, [-1.0, 0.0, 1.0])
    assert any(shape.x0 == 0.0 and shape.x1 == 0.0 for shape in figure.layout.shapes)


def test_identical_values_remain_visible_through_vertical_offsets(
    returns: pd.DataFrame,
) -> None:
    config = AssumptionConfig(expected_return_method="mean")
    table = build_assumption_table(returns, config)
    figure = build_expected_return_estimator_comparison(table, config)
    raw = _trace(figure, "Raw Historical Mean")
    final = _trace(figure, "Final E[r]")
    raw_position = _trace_assets(raw).index("BTC")
    final_position = _trace_assets(final).index("BTC")

    assert raw.x[raw_position] == pytest.approx(final.x[final_position])
    assert raw.y[raw_position] != final.y[final_position]
    assert raw.marker.symbol == "circle-open"
    assert final.marker.symbol == "star"
    assert "Raw Historical Mean" in final.customdata[final_position][10]


def test_all_estimator_basis_point_differences_are_exact(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    figure = build_expected_return_estimator_comparison(table, robust_config)
    audit = figure.layout.meta["estimator_differences_basis_points_by_asset"]

    expected = (table.at["SOL", "trimmed_mean"] - table.at["SOL", "mean"]) * 10_000
    assert audit["SOL"]["trimmed_mean"] == pytest.approx(expected)


def test_all_estimator_metadata_contains_complete_audit(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    figure = build_expected_return_estimator_comparison(
        table, robust_config, horizon_days=7
    )
    meta = figure.layout.meta

    required = {
        "asset_order",
        "view_mode",
        "horizon_days",
        "active_expected_return_estimator",
        "estimator_values_by_asset",
        "final_expected_return_by_asset",
        "estimator_dispersion_by_asset",
        "trim_proportion",
        "winsorization_proportion",
        "shrinkage_weight",
        "manual_view_blend_weight",
        "marker_contract",
    }
    assert required.issubset(meta)
    assert meta["horizon_days"] == 7
    assert meta["final_expected_return_by_asset"] == pytest.approx(
        table["final_expected_return"].to_dict()
    )
    assert figure.layout.title.text == (
        "Return Estimator Comparison — 7-Day Assumptions"
    )


def test_chart_building_does_not_change_optimizer_expected_returns(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    before = robust_config.final_expected_returns(returns).copy()
    build_expected_return_estimator_comparison(table, robust_config)
    after = robust_config.final_expected_returns(returns)

    pd.testing.assert_series_equal(after, before)


def test_returns_plotly_figure_with_both_endpoints(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    figure = build_expected_return_dumbbell(table, robust_config)

    assert isinstance(figure, go.Figure)
    raw = _trace(figure, "Raw Historical Mean")
    final = _trace(figure, "Final E[r]")
    assert list(raw.y) == list(table.index)
    assert list(final.y) == list(table.index)
    assert len(raw.x) == len(final.x) == len(table)


def test_connector_endpoints_match_underlying_table(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    figure = build_expected_return_dumbbell(
        table, robust_config, comparison="trimmed_mean"
    )
    connector = _trace(figure, "Adjustment")

    for position, asset in enumerate(table.index):
        start, end, separator = connector.x[position * 3 : position * 3 + 3]
        assert start == pytest.approx(table.loc[asset, "mean"] * 100.0)
        assert end == pytest.approx(table.loc[asset, "trimmed_mean"] * 100.0)
        assert separator is None


def test_final_endpoint_is_exact_downstream_engine_output(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    downstream = robust_config.final_expected_returns(returns)
    figure = build_expected_return_dumbbell(table, robust_config)
    final = _trace(figure, "Final E[r]")

    np.testing.assert_allclose(
        np.asarray(final.x, dtype=float) / 100.0,
        downstream.reindex(final.y).to_numpy(),
    )


def test_basis_point_difference_is_correct(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    figure = build_expected_return_dumbbell(table, robust_config)
    audit = {row["asset"]: row for row in figure.layout.meta["audit_rows"]}

    expected = (
        table.loc["ETH", "final_expected_return"] - table.loc["ETH", "mean"]
    ) * 10_000.0
    assert audit["ETH"]["difference_basis_points"] == pytest.approx(expected)


def test_equal_endpoints_remain_visible(returns: pd.DataFrame) -> None:
    config = AssumptionConfig(expected_return_method="mean")
    table = build_assumption_table(returns, config)
    figure = build_expected_return_dumbbell(table, config)

    raw = _trace(figure, "Raw Historical Mean")
    final = _trace(figure, "Final E[r]")
    np.testing.assert_allclose(raw.x, final.x)
    assert raw.marker.symbol == "circle-open"
    assert final.marker.symbol == "diamond"
    assert sum(
        note.text == "No adjustment" for note in figure.layout.annotations
    ) == len(table)


def test_equal_endpoint_labels_stay_inside_plot(returns: pd.DataFrame) -> None:
    config = AssumptionConfig(expected_return_method="mean")
    table = build_assumption_table(returns, config)
    figure = build_expected_return_dumbbell(table, config)
    notes = {note.y: note for note in figure.layout.annotations}
    rightmost_asset = str(table["mean"].idxmax())
    leftmost_asset = str(table["mean"].idxmin())

    assert notes[rightmost_asset].xanchor == "right"
    assert notes[rightmost_asset].xshift < 0
    assert notes[leftmost_asset].xanchor == "left"
    assert notes[leftmost_asset].xshift > 0


def test_negative_estimates_and_zero_reference_are_preserved(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    figure = build_expected_return_dumbbell(table, robust_config, comparison="median")
    comparison = _trace(figure, "Median")

    assert comparison.x[list(comparison.y).index("ETH")] < 0.0
    assert any(shape.x0 == 0.0 and shape.x1 == 0.0 for shape in figure.layout.shapes)


def test_missing_manual_view_is_unavailable_not_zero(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    figure = build_expected_return_dumbbell(
        table, robust_config, comparison="manual_view"
    )
    manual = _trace(figure, "Manual View")
    btc_position = list(manual.y).index("BTC")
    eth_position = list(manual.y).index("ETH")

    assert manual.x[btc_position] is None
    assert manual.x[eth_position] == pytest.approx(-0.5)
    assert manual.customdata[btc_position][12] == "N/A"
    audit = {row["asset"]: row for row in figure.layout.meta["audit_rows"]}
    assert audit["BTC"]["comparison_value"] is None
    assert audit["BTC"]["manual_view_available"] is False


def test_selector_changes_only_displayed_endpoint(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    original = table.copy(deep=True)
    median = build_expected_return_dumbbell(table, robust_config, comparison="median")
    winsorized = build_expected_return_dumbbell(
        table, robust_config, comparison="winsorized_mean"
    )

    pd.testing.assert_frame_equal(table, original)
    np.testing.assert_allclose(
        _trace(median, "Median").x,
        table["median"].to_numpy() * 100.0,
    )
    np.testing.assert_allclose(
        _trace(winsorized, "Winsorized Mean").x,
        table["winsorized_mean"].to_numpy() * 100.0,
    )
    assert median.layout.meta["comparison_column"] == "median"
    assert winsorized.layout.meta["comparison_column"] == "winsorized_mean"


def test_sorting_is_explicit_and_default_preserves_portfolio_order(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    order = ["SOL", "BTC", "ETH"]
    default = build_expected_return_dumbbell(table, robust_config, asset_order=order)
    sorted_figure = build_expected_return_dumbbell(
        table,
        robust_config,
        asset_order=order,
        sort_by_adjustment=True,
    )

    assert list(_trace(default, "Raw Historical Mean").y) == order
    assert sorted_figure.layout.meta["sort_by_adjustment"] is True


def test_title_and_axis_use_percentage_points(
    returns: pd.DataFrame, robust_config: AssumptionConfig
) -> None:
    table = build_assumption_table(returns, robust_config)
    figure = build_expected_return_dumbbell(table, robust_config)

    assert (
        figure.layout.title.text
        == "Historical Mean vs Final Robust Assumption — 1-Day Return"
    )
    assert figure.layout.xaxis.title.text == "1-Day Return Assumption (%)"
    assert figure.layout.xaxis.ticksuffix == "%"


def test_chart_module_has_no_streamlit_dependency() -> None:
    source = inspect.getsource(assumption_charts)
    assert "import streamlit" not in source
    assert "st." not in source
