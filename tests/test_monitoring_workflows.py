"""Creation-workflow gates for Live Forward monitoring."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from var_cvar_crypto_risk.monitoring.database import (
    create_monitoring_engine,
    create_session_factory,
)
from var_cvar_crypto_risk.monitoring.domain import (
    ExperimentMode,
    ExperimentStatus,
    ForecastEvaluationStatus,
)
from var_cvar_crypto_risk.monitoring.models import Base
from var_cvar_crypto_risk.monitoring.prices import normalize_monitoring_prices
from var_cvar_crypto_risk.monitoring.recipes import (
    OptimizationRecipe,
    RiskMonitoringRecipe,
    ScenarioRecipe,
    SourceRecipe,
)
from var_cvar_crypto_risk.monitoring.repository import SqlAlchemyUnitOfWork
from var_cvar_crypto_risk.monitoring.workflows import ExperimentCreationWorkflow


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BTC": [100, 101, 99, 102, 101, 104, 103, 105, 104, 106, 107, 108],
            "ETH": [50, 51, 50, 52, 51, 53, 52, 54, 53, 55, 56, 57],
        },
        index=pd.date_range("2026-01-01", periods=12, freq="D"),
        dtype=float,
    )


def test_live_creation_freezes_cutoff_and_persists_only_launch_state(
    tmp_path: Path,
) -> None:
    engine = create_monitoring_engine(
        f"sqlite+pysqlite:///{(tmp_path / 'monitoring.db').as_posix()}"
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)

    def uow_factory():
        return SqlAlchemyUnitOfWork(session_factory)

    recipe = OptimizationRecipe(
        scenario=ScenarioRecipe(source="historical", horizon_days=1),
        risk=RiskMonitoringRecipe(horizon_days=1, estimation_window=4),
        source=SourceRecipe(
            provider="fixture",
            symbol_mapping={"BTC": "bitcoin", "ETH": "ethereum"},
            refreshable=True,
        ),
    )
    normalized = normalize_monitoring_prices(
        _frame(),
        source="fixture",
        retrieved_at=datetime(2026, 1, 12, 12, tzinfo=timezone.utc),
    )
    try:
        result = ExperimentCreationWorkflow(uow_factory).create(
            name="live point-in-time gate",
            mode=ExperimentMode.LIVE_FORWARD,
            base_currency="USD",
            initial_capital=100_000.0,
            recipe=recipe,
            normalized=normalized,
            universe=("BTC", "ETH"),
            training_start=date(2026, 1, 1),
            training_end=date(2026, 1, 10),
            optimization_as_of=date(2026, 1, 10),
            launch_date=date(2026, 1, 11),
            historical_evaluation_end=None,
            live_tracking_end=date(2026, 2, 28),
            benchmark_symbol="BTC",
            package_version="1.0.0",
            code_version="batch6-test",
            calculation_version="valuation-v1",
        )
        assert result.experiment.status is ExperimentStatus.ACTIVE
        assert result.historical_replay is None
        with uow_factory() as uow:
            snapshot = uow.snapshots.get_for_experiment(
                result.experiment.experiment_id
            )
            states = uow.valuations.list(result.experiment.experiment_id)
            forecasts = uow.forecasts.list(result.experiment.experiment_id)
            prices = uow.prices.list(source="fixture")
            events = uow.events.list(result.experiment.experiment_id)
        assert snapshot is not None
        input_dates = snapshot.assumptions["input_dates"]
        assert date.fromisoformat(input_dates["solver_input_max_date"]) <= date(
            2026, 1, 10
        )
        assert [state.state_date for state in states] == [date(2026, 1, 11)]
        assert states[0].daily_return == 0.0
        assert len(forecasts) == 1
        assert forecasts[0].evaluation_status is ForecastEvaluationStatus.PENDING
        assert max(item.observation_date for item in prices) == date(2026, 1, 11)
        assert events[-1].event_type == "live_forward_initialized"
        assert events[-1].event_metadata["session_optimizer_reused"] is False
    finally:
        engine.dispose()
