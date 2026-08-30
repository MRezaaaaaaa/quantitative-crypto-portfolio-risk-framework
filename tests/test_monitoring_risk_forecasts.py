"""Origin-safe VaR/CVaR forecast contract tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from uuid import uuid4

import pandas as pd
import pytest

from var_cvar_crypto_risk.monitoring.domain import (
    DailyAssetState,
    DailyPortfolioState,
    DataQualityStatus,
    DomainValidationError,
    ForecastEvaluationStatus,
)
from var_cvar_crypto_risk.monitoring.prices import normalize_monitoring_prices
from var_cvar_crypto_risk.monitoring.recipes import CashPolicy, RiskMonitoringRecipe
from var_cvar_crypto_risk.monitoring.risk_forecasts import (
    build_origin_safe_forecast,
    evaluate_matured_forecast,
)


NOW = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)


def _state(state_date: date, *, nav: float = 1_000.0) -> DailyPortfolioState:
    experiment_id = uuid4()
    return DailyPortfolioState(
        experiment_id=experiment_id,
        state_date=state_date,
        data_quality_status=DataQualityStatus.COMPLETE,
        calculation_version="test-v1",
        finalized=True,
        nav=nav,
        base_100_nav=100.0,
        cash_value=0.0,
        daily_return=0.0,
        cumulative_return=0.0,
        running_peak=max(nav, 1_000.0),
        drawdown=min(nav / max(nav, 1_000.0) - 1.0, 0.0),
        maximum_drawdown=min(nav / max(nav, 1_000.0) - 1.0, 0.0),
        total_drift=0.0,
        return_interval_days=0,
        asset_states=(
            DailyAssetState(
                experiment_id=experiment_id,
                state_date=state_date,
                asset="BTC",
                quantity=10.0,
                target_weight=1.0,
                price=100.0,
                market_value=nav,
                current_weight=1.0,
                drift_percentage_points=0.0,
            ),
        ),
    )


def _normalized(periods: int = 10):
    frame = pd.DataFrame(
        {"BTC": [100.0 + index + (index % 3) for index in range(periods)]},
        index=pd.date_range("2026-01-01", periods=periods, freq="D"),
    )
    return normalize_monitoring_prices(frame, source="fixture", retrieved_at=NOW)


def test_forecast_uses_only_origin_visible_current_exposure_data() -> None:
    state = _state(date(2026, 1, 10))
    recipe = RiskMonitoringRecipe(estimation_window=5, horizon_days=1)
    forecast = build_origin_safe_forecast(
        normalized=_normalized(),
        state=state,
        recipe=recipe,
        cash_policy=CashPolicy(enabled=False),
        model_version="risk-v1",
        created_at=NOW,
    )
    assert forecast.evaluation_status is ForecastEvaluationStatus.PENDING
    assert forecast.input_max_date == state.state_date
    assert forecast.target_date == date(2026, 1, 11)
    assert forecast.forecast_metadata["current_weights"] == {"BTC": 1.0}
    assert forecast.forecast_metadata["future_frame_used"] is False


def test_forecast_rejects_a_frame_that_extends_past_origin() -> None:
    with pytest.raises(DomainValidationError, match="must end"):
        build_origin_safe_forecast(
            normalized=_normalized(),
            state=_state(date(2026, 1, 9)),
            recipe=RiskMonitoringRecipe(estimation_window=5),
            cash_policy=CashPolicy(enabled=False),
            model_version="risk-v1",
        )


def test_insufficient_window_is_explicit_and_has_no_fabricated_estimate() -> None:
    state = _state(date(2026, 1, 3))
    forecast = build_origin_safe_forecast(
        normalized=_normalized(periods=3),
        state=state,
        recipe=RiskMonitoringRecipe(estimation_window=5),
        cash_policy=CashPolicy(enabled=False),
        model_version="risk-v1",
    )
    assert forecast.evaluation_status is ForecastEvaluationStatus.INSUFFICIENT_WINDOW
    assert forecast.forecast_var is None
    assert "requires 6" in forecast.forecast_metadata["reason"]


def test_matured_outcome_uses_var_only_for_breach_semantics() -> None:
    origin = _state(date(2026, 1, 10), nav=1_000.0)
    forecast = build_origin_safe_forecast(
        normalized=_normalized(),
        state=origin,
        recipe=RiskMonitoringRecipe(estimation_window=5),
        cash_policy=CashPolicy(enabled=False),
        model_version="risk-v1",
    )
    target = replace(
        _state(date(2026, 1, 11), nav=900.0),
        experiment_id=origin.experiment_id,
        asset_states=(
            replace(
                _state(date(2026, 1, 11)).asset_states[0],
                experiment_id=origin.experiment_id,
            ),
        ),
    )
    evaluated = evaluate_matured_forecast(
        forecast,
        origin_state=origin,
        target_state=target,
        evaluated_at=NOW,
    )
    assert evaluated.realized_horizon_loss == pytest.approx(0.10)
    assert evaluated.var_breach is (0.10 > float(forecast.forecast_var))
    assert evaluated.evaluation_status is ForecastEvaluationStatus.EVALUATED
