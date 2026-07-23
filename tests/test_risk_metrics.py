"""Tests for the risk metrics module."""

from __future__ import annotations

import pandas as pd
import pytest

from var_cvar_crypto_risk.risk_metrics import (
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


# ── Asset-level drawdowns ──────────────────────────────────────────────────


def test_calculate_asset_drawdowns_columns_and_sign(sample_returns: pd.DataFrame) -> None:
    from var_cvar_crypto_risk.risk_metrics import calculate_asset_drawdowns

    dd = calculate_asset_drawdowns(sample_returns)
    assert list(dd.columns) == list(sample_returns.columns)
    assert dd.shape == sample_returns.shape
    # drawdowns are always <= 0 (within fp tolerance)
    assert (dd.to_numpy() <= 1e-9).all()
