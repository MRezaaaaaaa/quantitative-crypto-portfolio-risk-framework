"""Offline Live/Hybrid append, idempotency, rollback, and CLI tests."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from var_cvar_crypto_risk.monitoring.cli import main as cli_main
from var_cvar_crypto_risk.monitoring.database import (
    create_monitoring_engine,
    create_session_factory,
)
from var_cvar_crypto_risk.monitoring.domain import (
    DomainValidationError,
    ExperimentMode,
    ExperimentStatus,
    ForecastEvaluationStatus,
    MonitoringRunStatus,
    PriceDataStatus,
    PriceObservation,
)
from var_cvar_crypto_risk.monitoring.historical_replay import HistoricalReplayService
from var_cvar_crypto_risk.monitoring.live_update import (
    LiveMonitoringService,
    LiveUpdateResult,
)
from var_cvar_crypto_risk.monitoring.models import Base
from var_cvar_crypto_risk.monitoring.prices import normalize_monitoring_prices
from var_cvar_crypto_risk.monitoring.providers import (
    PriceFetchRequest,
    PriceProviderRegistry,
    ProviderPriceBatch,
)
from var_cvar_crypto_risk.monitoring.recipes import (
    OptimizationRecipe,
    RiskMonitoringRecipe,
    ScenarioRecipe,
    SourceRecipe,
)
from var_cvar_crypto_risk.monitoring.repository import (
    PersistenceCounts,
    SqlAlchemyUnitOfWork,
)
from var_cvar_crypto_risk.monitoring.services import ExperimentRegistry


HISTORICAL_RETRIEVED = datetime(2026, 1, 16, 18, tzinfo=timezone.utc)
UPDATE_AS_OF = datetime(2026, 1, 21, 12, tzinfo=timezone.utc)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=24, freq="D")
    return pd.DataFrame(
        {
            "BTC": [100.0 + index + (index % 3) for index in range(24)],
            "ETH": [50.0 + index * 0.7 + (index % 4) for index in range(24)],
        },
        index=dates,
    )


def _recipe(*, fallback: bool = False) -> OptimizationRecipe:
    metadata = (
        {
            "fallback_provider": "backup",
            "fallback_symbol_mapping": {"BTC": "BTC-X", "ETH": "ETH-X"},
        }
        if fallback
        else {}
    )
    return OptimizationRecipe(
        scenario=ScenarioRecipe(source="historical", horizon_days=1, random_seed=8),
        risk=RiskMonitoringRecipe(
            horizon_days=1,
            estimation_window=4,
            evaluation_mode="overlapping",
        ),
        source=SourceRecipe(
            provider="fixture",
            symbol_mapping={"BTC": "bitcoin", "ETH": "ethereum"},
            refreshable=True,
            metadata=metadata,
        ),
    )


class FakeProvider:
    provider_name = "fixture"

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        actual_source: str = "fixture-live",
        complete_through: date = date(2026, 1, 24),
    ) -> None:
        self.frame = frame
        self.actual_source = actual_source
        self.complete_through = complete_through
        self.calls: list[PriceFetchRequest] = []

    def fetch(self, request: PriceFetchRequest) -> ProviderPriceBatch:
        self.calls.append(request)
        return ProviderPriceBatch(
            prices=self.frame,
            actual_source=self.actual_source,
            quote_currency=request.quote_currency,
            retrieved_at=request.requested_at,
            complete_through=self.complete_through,
            metadata={"requested_provider": self.provider_name},
        )


class FailingProvider:
    provider_name = "fixture"

    def fetch(self, request: PriceFetchRequest) -> ProviderPriceBatch:
        raise RuntimeError("private-token-must-never-be-persisted")


class BackupProvider(FakeProvider):
    provider_name = "backup"


def _setup_hybrid(root: Path, *, recipe: OptimizationRecipe | None = None):
    selected_recipe = recipe or _recipe()
    engine = create_monitoring_engine(
        f"sqlite+pysqlite:///{(root / 'monitoring.db').as_posix()}"
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)

    def uow_factory():
        return SqlAlchemyUnitOfWork(session_factory)

    experiment = ExperimentRegistry(uow_factory).create(
        name="hybrid monitoring",
        mode=ExperimentMode.HYBRID,
        base_currency="USD",
        initial_capital=100_000.0,
        benchmark_symbol="BTC",
        recipe=selected_recipe,
        training_start=date(2026, 1, 1),
        training_end=date(2026, 1, 10),
        optimization_as_of=date(2026, 1, 10),
        launch_date=date(2026, 1, 11),
        historical_evaluation_end=date(2026, 1, 16),
        live_tracking_end=date(2026, 1, 22),
    )
    historical = normalize_monitoring_prices(
        _frame(), source="fixture", retrieved_at=HISTORICAL_RETRIEVED
    )
    HistoricalReplayService(uow_factory).run(
        experiment_id=experiment.experiment_id,
        normalized=historical,
        universe=["BTC", "ETH"],
        recipe=selected_recipe,
        package_version="1.0.0",
        code_version="batch5-test",
        calculation_version="valuation-v1",
    )
    return engine, uow_factory, experiment, selected_recipe


def test_live_update_appends_complete_dates_excludes_partial_day_and_is_idempotent(
    tmp_path: Path,
) -> None:
    engine, uow_factory, experiment, _ = _setup_hybrid(tmp_path)
    provider = FakeProvider(_frame())
    service = LiveMonitoringService(
        uow_factory, PriceProviderRegistry([provider])
    )
    try:
        first = service.update_experiment(
            experiment.experiment_id,
            requested_cutoff=date(2026, 1, 24),
            as_of=UPDATE_AS_OF,
            code_version="batch5-test",
            calculation_version="valuation-v1",
        )
        assert first.actual_cutoff == date(2026, 1, 20)
        assert first.processed_dates == 4
        assert first.actual_source == "fixture-live"
        assert provider.calls[0].start_date == date(2026, 1, 13)
        assert provider.calls[0].end_date == date(2026, 1, 20)
        with uow_factory() as uow:
            states = uow.valuations.list(experiment.experiment_id)
            forecasts = uow.forecasts.list(experiment.experiment_id)
            runs = uow.runs.list(experiment.experiment_id)
            prices = uow.prices.list(
                symbols=["BTC", "ETH"], source="fixture-live"
            )
        assert states[-1].state_date == date(2026, 1, 20)
        assert all(state.state_date <= date(2026, 1, 20) for state in states)
        assert all(item.observation_date <= date(2026, 1, 20) for item in prices)
        live_forecasts = [item for item in forecasts if item.origin_date >= date(2026, 1, 17)]
        assert [item.evaluation_status for item in live_forecasts] == [
            ForecastEvaluationStatus.EVALUATED,
            ForecastEvaluationStatus.EVALUATED,
            ForecastEvaluationStatus.EVALUATED,
            ForecastEvaluationStatus.PENDING,
        ]
        assert runs[-1].status is MonitoringRunStatus.COMPLETED
        assert runs[-1].run_metadata["actual_source"] == "fixture-live"

        second = service.update_experiment(
            experiment.experiment_id,
            requested_cutoff=date(2026, 1, 24),
            as_of=UPDATE_AS_OF,
            code_version="batch5-test",
            calculation_version="valuation-v1",
        )
        assert second.processed_dates == 0
        assert len(provider.calls) == 1
        with uow_factory() as uow:
            assert len(uow.valuations.list(experiment.experiment_id)) == len(states)
            assert len(uow.forecasts.list(experiment.experiment_id)) == len(forecasts)
            assert len(uow.runs.list(experiment.experiment_id)) == 2
    finally:
        engine.dispose()


def test_internal_missing_asset_creates_incomplete_state_and_pending_outcome(
    tmp_path: Path,
) -> None:
    engine, uow_factory, experiment, _ = _setup_hybrid(tmp_path)
    frame = _frame()
    frame.loc["2026-01-18", "ETH"] = float("nan")
    service = LiveMonitoringService(
        uow_factory, PriceProviderRegistry([FakeProvider(frame)])
    )
    try:
        result = service.update_experiment(
            experiment.experiment_id,
            as_of=UPDATE_AS_OF,
            code_version="batch5-test",
            calculation_version="valuation-v1",
        )
        assert result.actual_cutoff == date(2026, 1, 20)
        assert result.incomplete_dates == 1
        with uow_factory() as uow:
            missing = uow.valuations.get(experiment.experiment_id, date(2026, 1, 18))
            forecasts = uow.forecasts.list(experiment.experiment_id)
        assert missing is not None and missing.finalized is False
        target = next(item for item in forecasts if item.target_date == date(2026, 1, 18))
        assert target.evaluation_status is ForecastEvaluationStatus.PENDING
    finally:
        engine.dispose()


def test_provider_revision_cannot_overwrite_a_finalized_portfolio_date(
    tmp_path: Path,
) -> None:
    engine, uow_factory, experiment, _ = _setup_hybrid(tmp_path)
    service = LiveMonitoringService(
        uow_factory, PriceProviderRegistry([FakeProvider(_frame())])
    )
    try:
        service.update_experiment(
            experiment.experiment_id,
            as_of=UPDATE_AS_OF,
            code_version="batch5-test",
            calculation_version="valuation-v1",
        )
        with uow_factory() as uow:
            original = uow.valuations.get(
                experiment.experiment_id, date(2026, 1, 17)
            )
        assert original is not None and original.finalized

        revised = _frame()
        revised.loc["2026-01-17", ["BTC", "ETH"]] *= 5.0
        LiveMonitoringService(
            uow_factory, PriceProviderRegistry([FakeProvider(revised)])
        ).update_experiment(
            experiment.experiment_id,
            as_of=datetime(2026, 1, 22, 12, tzinfo=timezone.utc),
            code_version="batch5-test",
            calculation_version="valuation-v1",
        )
        with uow_factory() as uow:
            restored = uow.valuations.get(
                experiment.experiment_id, date(2026, 1, 17)
            )
        assert restored == original
    finally:
        engine.dispose()


def test_failed_financial_transaction_rolls_back_and_retry_gets_new_run(
    tmp_path: Path,
) -> None:
    engine, uow_factory, experiment, _ = _setup_hybrid(tmp_path)
    correct = _frame()
    conflicting = correct.copy()
    conflicting.loc["2026-01-17", "ETH"] += 10.0
    with uow_factory() as uow:
        uow.prices.add_many(
            [
                PriceObservation(
                    symbol="ETH",
                    observation_date=date(2026, 1, 17),
                    price=float(correct.loc["2026-01-17", "ETH"]),
                    quote_currency="USD",
                    source="fixture-live",
                    retrieved_at=UPDATE_AS_OF,
                    data_status=PriceDataStatus.COMPLETE,
                )
            ]
        )
        uow.commit()
    failed_service = LiveMonitoringService(
        uow_factory, PriceProviderRegistry([FakeProvider(conflicting)])
    )
    try:
        with pytest.raises(Exception):
            failed_service.update_experiment(
                experiment.experiment_id,
                as_of=UPDATE_AS_OF,
                code_version="batch5-test",
                calculation_version="valuation-v1",
            )
        with uow_factory() as uow:
            assert uow.valuations.get(
                experiment.experiment_id, date(2026, 1, 17)
            ) is None
            btc = uow.prices.list(
                symbols=["BTC"],
                start=date(2026, 1, 17),
                end=date(2026, 1, 17),
                source="fixture-live",
            )
            failed_runs = uow.runs.list(experiment.experiment_id)
        assert btc == []
        assert failed_runs[-1].status is MonitoringRunStatus.FAILED
        assert "private" not in (failed_runs[-1].error_summary or "")

        retry = LiveMonitoringService(
            uow_factory, PriceProviderRegistry([FakeProvider(correct)])
        ).update_experiment(
            experiment.experiment_id,
            as_of=UPDATE_AS_OF,
            code_version="batch5-test",
            calculation_version="valuation-v1",
        )
        assert retry.processed_dates == 4
        with uow_factory() as uow:
            runs = uow.runs.list(experiment.experiment_id)
            events = uow.events.list(experiment.experiment_id)
        assert len(runs) == 2
        assert runs[0].run_id != runs[1].run_id
        assert any(item.event_type == "monitoring_retry_started" for item in events)
    finally:
        engine.dispose()


def test_declared_fallback_records_actual_source_without_primary_error_text(
    tmp_path: Path,
) -> None:
    recipe = _recipe(fallback=True)
    engine, uow_factory, experiment, _ = _setup_hybrid(tmp_path, recipe=recipe)
    backup = BackupProvider(_frame(), actual_source="backup-live")
    service = LiveMonitoringService(
        uow_factory, PriceProviderRegistry([FailingProvider(), backup])
    )
    try:
        result = service.update_experiment(
            experiment.experiment_id,
            as_of=UPDATE_AS_OF,
            code_version="batch5-test",
            calculation_version="valuation-v1",
        )
        assert result.actual_source == "backup-live"
        assert result.warning_count >= 1
        with uow_factory() as uow:
            run = uow.runs.list(experiment.experiment_id)[-1]
        serialized = str(run.run_metadata)
        assert run.run_metadata["fallback_used"] is True
        assert "private-token" not in serialized
    finally:
        engine.dispose()


def test_live_end_transitions_experiment_to_completed(tmp_path: Path) -> None:
    engine, uow_factory, experiment, _ = _setup_hybrid(tmp_path)
    service = LiveMonitoringService(
        uow_factory, PriceProviderRegistry([FakeProvider(_frame())])
    )
    try:
        result = service.update_experiment(
            experiment.experiment_id,
            requested_cutoff=date(2026, 1, 22),
            as_of=datetime(2026, 1, 23, 12, tzinfo=timezone.utc),
            code_version="batch5-test",
            calculation_version="valuation-v1",
        )
        assert result.actual_cutoff == date(2026, 1, 22)
        assert result.final_status is ExperimentStatus.COMPLETED
    finally:
        engine.dispose()


def test_update_all_active_isolates_one_experiment_failure(
    tmp_path: Path, monkeypatch
) -> None:
    engine, uow_factory, first, recipe = _setup_hybrid(tmp_path)
    registry = ExperimentRegistry(uow_factory)
    second = registry.create(
        name="second active",
        mode=ExperimentMode.LIVE_FORWARD,
        base_currency="USD",
        initial_capital=50_000.0,
        recipe=recipe,
        training_start=date(2026, 1, 1),
        training_end=date(2026, 1, 10),
        optimization_as_of=date(2026, 1, 10),
        launch_date=date(2026, 1, 11),
    )
    registry.transition(second.experiment_id, ExperimentStatus.ACTIVE)
    service = LiveMonitoringService(
        uow_factory, PriceProviderRegistry([FakeProvider(_frame())])
    )

    def fake_update(experiment_id, **_kwargs):
        if experiment_id == second.experiment_id:
            raise RuntimeError("isolated failure")
        return LiveUpdateResult(
            experiment_id=first.experiment_id,
            run_id=first.experiment_id,
            final_status=ExperimentStatus.ACTIVE,
            requested_cutoff=date(2026, 1, 20),
            actual_cutoff=date(2026, 1, 20),
            actual_source="fixture-live",
            processed_dates=1,
            price_counts=PersistenceCounts(),
            state_counts=PersistenceCounts(),
            forecast_counts=PersistenceCounts(),
            evaluated_forecasts=0,
            incomplete_dates=0,
            warning_count=0,
        )

    monkeypatch.setattr(service, "update_experiment", fake_update)
    try:
        result = service.update_all_active(
            as_of=UPDATE_AS_OF,
            code_version="batch5-test",
            calculation_version="valuation-v1",
        )
        assert [item.experiment_id for item in result.completed] == [first.experiment_id]
        assert result.failed_experiment_ids == (second.experiment_id,)
    finally:
        engine.dispose()


def test_live_update_rejects_an_as_of_time_in_the_future(tmp_path: Path) -> None:
    engine, uow_factory, experiment, _ = _setup_hybrid(tmp_path)
    service = LiveMonitoringService(
        uow_factory, PriceProviderRegistry([FakeProvider(_frame())])
    )
    try:
        with pytest.raises(DomainValidationError, match="future"):
            service.update_experiment(
                experiment.experiment_id,
                as_of=datetime(2099, 1, 1, tzinfo=timezone.utc),
                code_version="batch5-test",
                calculation_version="valuation-v1",
            )
        with uow_factory() as uow:
            assert uow.runs.list(experiment.experiment_id) == []
    finally:
        engine.dispose()


def test_one_shot_cli_handles_empty_migrated_database(
    tmp_path: Path, capsys
) -> None:
    database_path = tmp_path / "cli.db"
    engine = create_monitoring_engine(
        f"sqlite+pysqlite:///{database_path.as_posix()}"
    )
    Base.metadata.create_all(engine)
    engine.dispose()
    exit_code = cli_main(
        [
            "--all-active",
            "--database-url",
            f"sqlite+pysqlite:///{database_path.as_posix()}",
        ]
    )
    assert exit_code == 0
    assert '"completed": []' in capsys.readouterr().out
