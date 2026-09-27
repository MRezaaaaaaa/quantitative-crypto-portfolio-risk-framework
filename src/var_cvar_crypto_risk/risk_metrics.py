"""Portfolio-level risk metrics and summary table."""

from __future__ import annotations

import math

import pandas as pd
from scipy import stats

from .cvar_models import calculate_cvar, return_cvar_to_money_cvar
from .returns import annualize_return, annualize_volatility
from .var_models import calculate_var, return_var_to_money_var


def calculate_drawdown(portfolio_returns: pd.Series) -> pd.DataFrame:
    """Calculate running drawdown from a portfolio return series.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - ``cumulative_return``: ``(1 + r).cumprod() - 1``
        - ``rolling_max``: running maximum cumulative return, including launch
          wealth ``1.0`` as a zero-return peak
        - ``drawdown``: ``wealth / running_peak_wealth - 1``
    """
    wealth = (1.0 + portfolio_returns).cumprod()
    cumulative = wealth - 1.0
    running_peak_wealth = wealth.cummax().clip(lower=1.0)
    rolling_max = running_peak_wealth - 1.0
    drawdown = wealth / running_peak_wealth - 1.0
    return pd.DataFrame(
        {
            "cumulative_return": cumulative,
            "rolling_max": rolling_max,
            "drawdown": drawdown,
        }
    )


def calculate_max_drawdown(portfolio_returns: pd.Series) -> float:
    """Return the maximum drawdown as a negative float (or ``0.0`` if none)."""
    dd = calculate_drawdown(portfolio_returns)["drawdown"]
    if dd.empty:
        return 0.0
    return float(dd.min())


def calculate_asset_drawdowns(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Per-asset drawdown series.

    Applies :func:`calculate_drawdown` to each column independently.

    Returns
    -------
    pandas.DataFrame
        One column per asset (same names as ``asset_returns``), each holding
        the running drawdown (values in ``[-1, 0]``), indexed like the input.
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise ValueError("asset_returns must be a pd.DataFrame.")
    drawdowns = {
        col: calculate_drawdown(asset_returns[col])["drawdown"]
        for col in asset_returns.columns
    }
    return pd.DataFrame(drawdowns, index=asset_returns.index)


def calculate_distribution_stats(
    returns: pd.Series,
    periods_per_year: int = 365,
) -> dict:
    """Return descriptive statistics relevant to risk analysis.

    Returns
    -------
    dict
        Keys: ``mean_daily``, ``annualized_return``, ``daily_volatility``,
        ``annualized_volatility``, ``skewness``, ``excess_kurtosis``,
        ``min_return``, ``max_return``, ``sharpe_ratio``.
    """
    clean = returns.dropna()
    mean_daily = float(clean.mean())
    daily_vol = float(clean.std(ddof=1))
    ann_return = annualize_return(clean, periods_per_year)
    ann_vol = annualize_volatility(clean, periods_per_year)
    skewness = float(stats.skew(clean.values, bias=False)) if len(clean) >= 3 else 0.0
    excess_kurt = (
        float(stats.kurtosis(clean.values, fisher=True, bias=False))
        if len(clean) >= 4
        else 0.0
    )
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    return {
        "mean_daily": mean_daily,
        "annualized_return": ann_return,
        "daily_volatility": daily_vol,
        "annualized_volatility": ann_vol,
        "skewness": skewness,
        "excess_kurtosis": excess_kurt,
        "min_return": float(clean.min()),
        "max_return": float(clean.max()),
        "sharpe_ratio": float(sharpe),
    }


_METHOD_LABELS = {
    "historical": "Historical",
    "gaussian": "Gaussian",
    "cornish_fisher": "Cornish-Fisher",
}


def _format_method_label(method: str) -> str:
    return _METHOD_LABELS.get(method, method.replace("_", " ").title())


def generate_risk_summary(
    portfolio_returns: pd.Series,
    confidence_level: float,
    initial_capital: float,
    var_methods: list[str],
    cvar_methods: list[str],
    periods_per_year: int = 365,
    return_method: str = "simple",
    risk_base_value: float | None = None,
    risk_base_date: object | None = None,
    risk_base_type: str | None = None,
) -> pd.DataFrame:
    """Generate the legacy low-level risk summary table.

    This API is retained for backward compatibility and focused estimator use.
    It is not the authoritative application or publication Risk Summary; those
    workflows use ``HistoricalRiskSummary`` from an explicit portfolio path.

    Returns
    -------
    pandas.DataFrame
        Columns ``["Metric", "Value", "Unit"]``. ``Value`` is the raw
        numeric value (already converted to percentage points or USD as
        appropriate), and ``Unit`` describes its interpretation. Monetary
        values are labeled as linearized when ``return_method="log"``.
    """
    if return_method not in {"simple", "log"}:
        raise ValueError(
            f"return_method must be 'simple' or 'log', got {return_method}."
        )
    money_base = initial_capital if risk_base_value is None else float(risk_base_value)
    if not math.isfinite(money_base) or money_base <= 0.0:
        raise ValueError("risk_base_value must be finite and positive")
    if risk_base_value is not None and not str(risk_base_type or "").strip():
        raise ValueError("risk_base_type is required when risk_base_value is supplied")
    stats_dict = calculate_distribution_stats(
        portfolio_returns, periods_per_year=periods_per_year
    )
    max_dd = calculate_max_drawdown(portfolio_returns)
    confidence_pct = confidence_level * 100.0
    money_unit = "USD" if return_method == "simple" else "USD (linearized)"

    rows: list[dict] = [
        {
            "Metric": "Mean Daily Return",
            "Value": stats_dict["mean_daily"] * 100.0,
            "Unit": "%",
        },
        {
            "Metric": "Annualized Return",
            "Value": stats_dict["annualized_return"] * 100.0,
            "Unit": "%",
        },
        {
            "Metric": "Daily Volatility",
            "Value": stats_dict["daily_volatility"] * 100.0,
            "Unit": "%",
        },
        {
            "Metric": "Annualized Volatility",
            "Value": stats_dict["annualized_volatility"] * 100.0,
            "Unit": "%",
        },
        {"Metric": "Skewness", "Value": stats_dict["skewness"], "Unit": ""},
        {
            "Metric": "Excess Kurtosis",
            "Value": stats_dict["excess_kurtosis"],
            "Unit": "",
        },
        {
            "Metric": "Min Daily Return",
            "Value": stats_dict["min_return"] * 100.0,
            "Unit": "%",
        },
        {
            "Metric": "Max Daily Return",
            "Value": stats_dict["max_return"] * 100.0,
            "Unit": "%",
        },
        {"Metric": "Max Drawdown", "Value": max_dd * 100.0, "Unit": "%"},
        {
            "Metric": "Sharpe Ratio (annualized)",
            "Value": stats_dict["sharpe_ratio"],
            "Unit": "",
        },
    ]

    for method in var_methods:
        var_pct = calculate_var(portfolio_returns, method, confidence_level)
        money_var = return_var_to_money_var(var_pct, money_base)
        label = _format_method_label(method)
        rows.append(
            {
                "Metric": f"{label} VaR {confidence_pct:.0f}%",
                "Value": var_pct * 100.0,
                "Unit": "%",
            }
        )
        rows.append(
            {
                "Metric": f"{label} Money VaR {confidence_pct:.0f}%",
                "Value": money_var,
                "Unit": money_unit,
            }
        )

    for method in cvar_methods:
        cvar_pct = calculate_cvar(portfolio_returns, method, confidence_level)
        money_cvar = return_cvar_to_money_cvar(cvar_pct, money_base)
        label = _format_method_label(method)
        rows.append(
            {
                "Metric": f"{label} CVaR {confidence_pct:.0f}%",
                "Value": cvar_pct * 100.0,
                "Unit": "%",
            }
        )
        rows.append(
            {
                "Metric": f"{label} Money CVaR {confidence_pct:.0f}%",
                "Value": money_cvar,
                "Unit": money_unit,
            }
        )

    result = pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])
    if risk_base_value is not None:
        result["Risk Base Value"] = money_base
        result["Risk Base Date"] = risk_base_date
        result["Risk Base Type"] = str(risk_base_type)
    return result
