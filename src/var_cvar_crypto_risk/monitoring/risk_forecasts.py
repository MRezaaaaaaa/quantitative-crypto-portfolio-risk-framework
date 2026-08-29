"""Pure origin-safe risk forecast construction and outcome evaluation.

This module never queries persistence and rejects a price frame that extends
past the forecast origin.  The caller owns sequential information revelation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd

from ..cvar_models import calculate_cvar
from ..var_models import calculate_var
from .domain import (
    DailyPortfolioState,
    DailyRiskForecast,
    DomainValidationError,
    ForecastEvaluationStatus,
)
from .hashing import sha256_fingerprint
from .prices import NormalizedPriceData, fingerprint_price_slice
from .recipes import CashPolicy, RiskMonitoringRecipe


def _aggregate_simple_returns(values: np.ndarray, horizon_days: int) -> np.ndarray:
    """Reuse the project's compounded Simple-return horizon convention."""
    if horizon_days == 1:
        return values.copy()
    count = len(values) - horizon_days + 1
    if count < 1:
        return np.asarray([], dtype=float)
    return np.asarray(
        [
            float(np.prod(1.0 + values[index : index + horizon_days]) - 1.0)
            for index in range(count)
        ],
        dtype=float,
    )


def _insufficient_forecast(
    *,
    state: DailyPortfolioState,
    recipe: RiskMonitoringRecipe,
    target_date,
    model_version: str,
    input_data_hash: str,
    reason: str,
    created_at: datetime | None,
) -> DailyRiskForecast:
    return DailyRiskForecast.create(
        experiment_id=state.experiment_id,
        origin_date=state.state_date,
        target_date=target_date,
        horizon_days=recipe.horizon_days,
        evaluation_mode=recipe.evaluation_mode,
        estimation_window=recipe.estimation_window,
        var_method=recipe.var_method,
        cvar_method=recipe.cvar_method,
        confidence_level=recipe.confidence_level,
        horizon_construction="rolling_compounded_simple_returns",
        convention_version=recipe.convention_version,
        model_version=model_version,
        portfolio_definition="current_drifted_weights",
        input_max_date=state.state_date,
        input_data_hash=input_data_hash,
        evaluation_status=ForecastEvaluationStatus.INSUFFICIENT_WINDOW,
        forecast_metadata={
            "reason": reason,
            "future_frame_used": False,
            "return_method": "simple",
        },
        **({"created_at": created_at} if created_at is not None else {}),
    )


def build_origin_safe_forecast(
    *,
    normalized: NormalizedPriceData,
    state: DailyPortfolioState,
    recipe: RiskMonitoringRecipe,
    cash_policy: CashPolicy,
    model_version: str,
    created_at: datetime | None = None,
) -> DailyRiskForecast:
    """Build current-exposure VaR/CVaR using data available at one origin.

    ``estimation_window`` counts daily return observations.  For a multi-day
    horizon those observations are converted to overlapping compounded horizon
    returns, matching the existing backtesting convention without square-root
    scaling.  Non-overlapping mode controls forecast scheduling, not the
    estimator sample construction.
    """
    if not state.finalized or state.nav is None:
        raise DomainValidationError("risk forecast origin requires a finalized state")
    origin = state.state_date
    origin_key = pd.Timestamp(origin)
    if normalized.prices.empty or normalized.prices.index.max() > origin_key:
        raise DomainValidationError(
            "forecast price frame must end at or before the origin date"
        )
    if origin_key not in normalized.prices.index:
        raise DomainValidationError("forecast origin is absent from the price frame")

    market_states = tuple(item for item in state.asset_states if not item.is_cash)
    cash_states = tuple(item for item in state.asset_states if item.is_cash)
    if len(cash_states) > 1 or bool(cash_states) != cash_policy.enabled:
        raise DomainValidationError("forecast state does not match frozen cash policy")
    if any(item.current_weight is None for item in state.asset_states):
        raise DomainValidationError("forecast requires complete current weights")
    current_weights = {
        item.asset: float(item.current_weight) for item in state.asset_states
    }
    if not math.isclose(
        math.fsum(current_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise DomainValidationError("forecast current weights must sum to one")
    assets = tuple(item.asset for item in market_states)
    missing_columns = [asset for asset in assets if asset not in normalized.prices]
    if missing_columns:
        raise DomainValidationError(
            "forecast source lacks current-exposure assets: "
            + ", ".join(missing_columns)
        )

    required_prices = recipe.estimation_window + 1
    visible = normalized.prices.loc[:origin_key, list(assets)].tail(required_prices)
    visible_hash = fingerprint_price_slice(
        visible,
        source=normalized.source,
        quote_currency=normalized.quote_currency,
    )
    input_hash = sha256_fingerprint(
        {
            "visible_price_hash": visible_hash,
            "current_weights": current_weights,
            "risk_recipe": recipe.to_dict(),
            "origin_date": origin,
        }
    )
    target_date = origin + timedelta(days=recipe.horizon_days)
    if len(visible) < required_prices:
        return _insufficient_forecast(
            state=state,
            recipe=recipe,
            target_date=target_date,
            model_version=model_version,
            input_data_hash=input_hash,
            reason=(
                f"requires {required_prices} price observations; found {len(visible)}"
            ),
            created_at=created_at,
        )
    if visible.isna().any().any():
        return _insufficient_forecast(
            state=state,
            recipe=recipe,
            target_date=target_date,
            model_version=model_version,
            input_data_hash=input_hash,
            reason="estimation window contains explicit missing prices",
            created_at=created_at,
        )
    gaps = visible.index.to_series().diff().dropna().dt.days
    if (gaps != 1).any():
        return _insufficient_forecast(
            state=state,
            recipe=recipe,
            target_date=target_date,
            model_version=model_version,
            input_data_hash=input_hash,
            reason="estimation window is not consecutive daily crypto data",
            created_at=created_at,
        )

    daily_asset_returns = visible.divide(visible.shift(1)).subtract(1.0).iloc[1:]
    portfolio_returns = pd.Series(0.0, index=daily_asset_returns.index, dtype=float)
    for asset in assets:
        portfolio_returns = portfolio_returns.add(
            daily_asset_returns[asset] * current_weights[asset], fill_value=0.0
        )
    if cash_states:
        cash_symbol = cash_states[0].asset
        portfolio_returns = portfolio_returns.add(
            current_weights[cash_symbol] * cash_policy.horizon_return(1)
        )
    if (
        portfolio_returns.isna().any()
        or not np.isfinite(portfolio_returns.to_numpy(dtype=float)).all()
    ):
        raise DomainValidationError("forecast estimator returns are non-finite")
    estimator_values = _aggregate_simple_returns(
        portfolio_returns.to_numpy(dtype=float), recipe.horizon_days
    )
    if len(estimator_values) < 2:
        return _insufficient_forecast(
            state=state,
            recipe=recipe,
            target_date=target_date,
            model_version=model_version,
            input_data_hash=input_hash,
            reason="fewer than two horizon-return estimator observations",
            created_at=created_at,
        )
    estimator_returns = pd.Series(estimator_values, dtype=float)
    try:
        forecast_var = calculate_var(
            estimator_returns, recipe.var_method, recipe.confidence_level
        )
        forecast_cvar = calculate_cvar(
            estimator_returns, recipe.cvar_method, recipe.confidence_level
        )
    except ValueError as exc:
        raise DomainValidationError(f"risk forecast method failed: {exc}") from exc
    forecast_volatility = float(estimator_returns.std(ddof=1))
    return DailyRiskForecast.create(
        experiment_id=state.experiment_id,
        origin_date=origin,
        target_date=target_date,
        horizon_days=recipe.horizon_days,
        evaluation_mode=recipe.evaluation_mode,
        estimation_window=recipe.estimation_window,
        var_method=recipe.var_method,
        cvar_method=recipe.cvar_method,
        confidence_level=recipe.confidence_level,
        horizon_construction="rolling_compounded_simple_returns",
        convention_version=recipe.convention_version,
        model_version=model_version,
        portfolio_definition="current_drifted_weights",
        input_max_date=visible.index.max().date(),
        input_data_hash=input_hash,
        evaluation_status=ForecastEvaluationStatus.PENDING,
        forecast_var=float(forecast_var),
        forecast_cvar=float(forecast_cvar),
        forecast_volatility=forecast_volatility,
        forecast_metadata={
            "current_weights": current_weights,
            "input_min_date": visible.index.min().date().isoformat(),
            "input_max_date": visible.index.max().date().isoformat(),
            "horizon_sample_count": len(estimator_returns),
            "future_frame_used": False,
            "return_method": "simple",
            "overlapping_estimator_samples": recipe.horizon_days > 1,
            "independence_limitation": recipe.horizon_days > 1,
        },
        **({"created_at": created_at} if created_at is not None else {}),
    )


def evaluate_matured_forecast(
    forecast: DailyRiskForecast,
    *,
    origin_state: DailyPortfolioState,
    target_state: DailyPortfolioState,
    evaluated_at: datetime | None = None,
) -> DailyRiskForecast:
    """Evaluate one pending forecast with a matching realized NAV outcome."""
    if origin_state.experiment_id != forecast.experiment_id or (
        target_state.experiment_id != forecast.experiment_id
    ):
        raise DomainValidationError("forecast outcome states belong to another experiment")
    if origin_state.state_date != forecast.origin_date:
        raise DomainValidationError("forecast origin state date does not match")
    if target_state.state_date != forecast.target_date:
        raise DomainValidationError("forecast target state date does not match")
    if (
        not origin_state.finalized
        or not target_state.finalized
        or origin_state.nav is None
        or target_state.nav is None
    ):
        raise DomainValidationError("forecast outcome requires complete finalized NAVs")
    realized_return = target_state.nav / origin_state.nav - 1.0
    return forecast.evaluate(-realized_return, at=evaluated_at)


__all__ = ["build_origin_safe_forecast", "evaluate_matured_forecast"]
