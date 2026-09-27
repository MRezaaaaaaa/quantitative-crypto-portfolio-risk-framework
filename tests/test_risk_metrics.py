"""Tests for the risk metrics module."""

from __future__ import annotations

import pandas as pd
import pytest

from var_cvar_crypto_risk.risk_metrics import (
    calculate_asset_drawdowns,
    calculate_drawdown,
    calculate_max_drawdown,
    generate_risk_summary,
)


def test_generate_risk_summary_returns_dataframe(long_returns: pd.Series) -> None:
    summary = generate_risk_summary(
        portfolio_returns=long_returns,
        confidence_level=0.95,
        initial_capital=100_000.0,
        var_methods=["historical", "gaussian", "cornish_fisher"],
        cvar_methods=["historical", "gaussian"],
    )
    assert isinstance(summary, pd.DataFrame)
    assert {"Metric", "Value", "Unit"}.issubset(summary.columns)


def test_summary_contains_historical_var(long_returns: pd.Series) -> None:
    summary = generate_risk_summary(
        portfolio_returns=long_returns,
        confidence_level=0.95,
        initial_capital=100_000.0,
        var_methods=["historical"],
        cvar_methods=["historical"],
    )
    assert summary["Metric"].str.contains("Historical VaR").any()


def test_summary_contains_gaussian_cvar(long_returns: pd.Series) -> None:
    summary = generate_risk_summary(
        portfolio_returns=long_returns,
        confidence_level=0.95,
        initial_capital=100_000.0,
        var_methods=["gaussian"],
        cvar_methods=["gaussian"],
    )
    assert summary["Metric"].str.contains("Gaussian CVaR").any()


def test_money_var_equals_pct_times_capital(long_returns: pd.Series) -> None:
    initial_capital = 100_000.0
    summary = generate_risk_summary(
        portfolio_returns=long_returns,
        confidence_level=0.95,
        initial_capital=initial_capital,
        var_methods=["historical"],
        cvar_methods=["historical"],
    )
    pct_row = summary[summary["Metric"].str.contains("Historical VaR 95%", regex=False)]
    money_row = summary[
        summary["Metric"].str.contains("Historical Money VaR 95%", regex=False)
    ]
    assert not pct_row.empty
    assert not money_row.empty
    pct_value = float(pct_row["Value"].iloc[0]) / 100.0
    money_value = float(money_row["Value"].iloc[0])
    assert abs(money_value - pct_value * initial_capital) < 0.01


def test_current_risk_base_is_explicit_and_drives_money_risk(
    long_returns: pd.Series,
) -> None:
    summary = generate_risk_summary(
        portfolio_returns=long_returns,
        confidence_level=0.95,
        initial_capital=100_000.0,
        var_methods=["historical"],
        cvar_methods=["historical"],
        risk_base_value=123_456.0,
        risk_base_date="2026-09-21",
        risk_base_type="current_net_nav",
    )
    pct = (
        float(summary.loc[summary["Metric"] == "Historical VaR 95%", "Value"].iloc[0])
        / 100.0
    )
    money = float(
        summary.loc[summary["Metric"] == "Historical Money VaR 95%", "Value"].iloc[0]
    )
    assert money == pytest.approx(pct * 123_456.0)
    assert summary["Risk Base Value"].eq(123_456.0).all()
    assert summary["Risk Base Date"].eq("2026-09-21").all()
    assert summary["Risk Base Type"].eq("current_net_nav").all()


def test_explicit_risk_base_requires_a_type(long_returns: pd.Series) -> None:
    with pytest.raises(ValueError, match="risk_base_type"):
        generate_risk_summary(
            portfolio_returns=long_returns,
            confidence_level=0.95,
            initial_capital=100_000.0,
            var_methods=["historical"],
            cvar_methods=["historical"],
            risk_base_value=110_000.0,
        )


def test_log_return_money_metrics_are_labeled_linearized(
    long_returns: pd.Series,
) -> None:
    summary = generate_risk_summary(
        portfolio_returns=long_returns,
        confidence_level=0.95,
        initial_capital=100_000.0,
        var_methods=["historical"],
        cvar_methods=["historical"],
        return_method="log",
    )
    money_rows = summary[summary["Metric"].str.contains("Money")]
    assert not money_rows.empty
    assert set(money_rows["Unit"]) == {"USD (linearized)"}


def test_risk_summary_rejects_unknown_return_method(
    long_returns: pd.Series,
) -> None:
    with pytest.raises(ValueError, match="return_method"):
        generate_risk_summary(
            portfolio_returns=long_returns,
            confidence_level=0.95,
            initial_capital=100_000.0,
            var_methods=["historical"],
            cvar_methods=["historical"],
            return_method="arithmetic",
        )


def test_max_drawdown_negative_for_volatile_series(long_returns: pd.Series) -> None:
    dd = calculate_max_drawdown(long_returns)
    assert dd <= 0
    assert isinstance(dd, float)


def test_max_drawdown_zero_for_all_positive_returns() -> None:
    rets = pd.Series(
        [0.01, 0.02, 0.005, 0.03, 0.015],
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    dd = calculate_max_drawdown(rets)
    assert dd >= -1e-12


def test_drawdown_uses_launch_wealth_as_the_initial_peak() -> None:
    returns = pd.Series(
        [-0.20, 0.125], index=pd.date_range("2024-01-01", periods=2, freq="D")
    )
    result = calculate_drawdown(returns)

    assert list(result.index) == list(returns.index)
    assert result["drawdown"].iloc[0] == pytest.approx(-0.20)
    assert result["drawdown"].iloc[1] == pytest.approx(-0.10)
    assert calculate_max_drawdown(returns) == pytest.approx(-0.20)


@pytest.mark.parametrize(
    ("returns", "expected"),
    [
        ([0.10, -0.10], [0.0, -0.10]),
        ([-0.20, 0.25], [-0.20, 0.0]),
        ([0.10, -0.20, 0.25, -0.10], [0.0, -0.20, 0.0, -0.10]),
    ],
)
def test_drawdown_handles_gain_recovery_and_multiple_episodes(returns, expected):
    series = pd.Series(returns, index=pd.date_range("2024-01-01", periods=len(returns)))
    result = calculate_drawdown(series)
    assert list(result["drawdown"]) == pytest.approx(expected)


def test_drawdown_preserves_missing_observations_without_zero_filling() -> None:
    returns = pd.Series(
        [-0.10, float("nan"), 0.05],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    result = calculate_drawdown(returns)

    assert result["drawdown"].iloc[0] == pytest.approx(-0.10)
    assert pd.isna(result["drawdown"].iloc[1])
    assert result["drawdown"].iloc[2] == pytest.approx(-0.055)


def test_return_drawdown_matches_equivalent_explicit_wealth_path() -> None:
    returns = pd.Series(
        [-0.20, 0.125, 0.20, -0.10],
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    wealth = (1.0 + returns).cumprod()
    explicit_peak = (
        pd.concat([pd.Series([1.0]), wealth.reset_index(drop=True)], ignore_index=True)
        .cummax()
        .iloc[1:]
    )
    explicit_peak.index = returns.index
    expected = wealth / explicit_peak - 1.0

    pd.testing.assert_series_equal(
        calculate_drawdown(returns)["drawdown"], expected, check_names=False
    )


# ── Asset-level drawdowns ──────────────────────────────────────────────────


def test_calculate_asset_drawdowns_columns_and_sign(
    sample_returns: pd.DataFrame,
) -> None:
    dd = calculate_asset_drawdowns(sample_returns)
    assert list(dd.columns) == list(sample_returns.columns)
    assert dd.shape == sample_returns.shape
    # drawdowns are always <= 0 (within fp tolerance)
    assert (dd.to_numpy() <= 1e-9).all()


def test_asset_drawdown_includes_each_assets_first_period_loss() -> None:
    returns = pd.DataFrame(
        {"BTC": [-0.20, 0.125], "ETH": [0.10, -0.10]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    result = calculate_asset_drawdowns(returns)

    assert result.loc[returns.index[0], "BTC"] == pytest.approx(-0.20)
    assert result.loc[returns.index[1], "BTC"] == pytest.approx(-0.10)
    assert result.loc[returns.index[0], "ETH"] == pytest.approx(0.0)
    assert result.loc[returns.index[1], "ETH"] == pytest.approx(-0.10)
