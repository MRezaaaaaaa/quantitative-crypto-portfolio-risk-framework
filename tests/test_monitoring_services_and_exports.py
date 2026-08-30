"""Registry, atomic persistence, idempotency, and export tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from var_cvar_crypto_risk.monitoring.database import (
    create_monitoring_engine,
    create_session_factory,
)
from var_cvar_crypto_risk.monitoring.domain import (
    DataQualityStatus,
    DomainValidationError,
    ExperimentMode,
    ExperimentStatus,
    ImmutableRecordError,
    OptimizationSnapshot,
    RecordNotFoundError,
    SnapshotAllocation,
)
from var_cvar_crypto_risk.monitoring.exports import export_experiment_bundle
from var_cvar_crypto_risk.monitoring.models import Base
from var_cvar_crypto_risk.monitoring.prices import normalize_monitoring_prices
from var_cvar_crypto_risk.monitoring.recipes import (
    CashPolicy,
    OptimizationRecipe,
    SourceRecipe,
)
from var_cvar_crypto_risk.monitoring.repository import SqlAlchemyUnitOfWork
from var_cvar_crypto_risk.monitoring.services import (
    ExperimentRegistry,
    MonitoringPersistenceService,
)
from var_cvar_crypto_risk.monitoring.valuation import value_fixed_holdings


NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


@pytest.fixture
def persistence(tmp_path: Path):
    engine = create_monitoring_engine(
        f"sqlite+pysqlite:///{(tmp_path / 'monitoring.db').as_posix()}"
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)

    def uow_factory():
        return SqlAlchemyUnitOfWork(session_factory)

    try:
        yield engine, session_factory, uow_factory
    finally:
        engine.dispose()


def _recipe(*, refreshable: bool = False) -> OptimizationRecipe:
    return OptimizationRecipe(
        source=SourceRecipe(
            provider="fixture",
            symbol_mapping={"BTC": "bitcoin"},
            refreshable=refreshable,
        )
    )


def _create_historical(registry: ExperimentRegistry):
    return registry.create(
        name="registry experiment",
        mode=ExperimentMode.HISTORICAL_OOS,
        base_currency="USD",
        initial_capital=1_000.0,
        recipe=_recipe(),
        training_start=date(2025, 1, 1),
        training_end=date(2025, 12, 31),
        optimization_as_of=date(2025, 12, 31),
        launch_date=date(2026, 1, 1),
        historical_evaluation_end=date(2026, 1, 3),
    )


def _snapshot(experiment_id) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        snapshot_id=uuid4(),
        experiment_id=experiment_id,
        package_version="1.0.0",
        code_version="abc123",
        objective="min_cvar",
        solver="CLARABEL",
        solver_status="optimal",
        source_data_hash="a" * 64,
        assumption_recipe_hash="b" * 64,
        assumptions={},
        constraints={"cash": CashPolicy(enabled=False).to_dict()},
        launch_forecast={"cvar": 0.05},
        scenario_metadata={"source": "historical"},
        return_policy={"wealth": "simple"},
        loss_convention={"name": "signed_loss_space"},
        residual_validation={"passed": True},
        allocations=(
            SnapshotAllocation(
                asset="BTC",
                asset_type="crypto",
                target_weight=1.0,
                launch_price=100.0,
                initial_value=1_000.0,
                quantity=10.0,
            ),
        ),
    ).activate(at=NOW)


def _normalized():
    return normalize_monitoring_prices(
        pd.DataFrame(
            {"BTC": [100.0, 110.0, 90.0]},
            index=pd.date_range("2026-01-01", periods=3, freq="D"),
        ),
        source="fixture",
        retrieved_at=NOW,
    )


def _states(experiment, snapshot):
    return value_fixed_holdings(
        experiment=experiment,
        snapshot=snapshot,
        normalized=_normalized(),
        cash_policy=CashPolicy(enabled=False),
        calculation_version="valuation-v1",
    )


def test_registry_creates_queries_transitions_and_archives_with_events(
    persistence,
) -> None:
    _engine, _session_factory, uow_factory = persistence
    registry = ExperimentRegistry(uow_factory)
    experiment = _create_historical(registry)
    assert registry.get(experiment.experiment_id) == experiment
    assert registry.list() == [experiment]
    active = registry.transition(experiment.experiment_id, ExperimentStatus.ACTIVE)
    assert active.status is ExperimentStatus.ACTIVE
    archived = registry.archive(experiment.experiment_id)
    assert archived.status is ExperimentStatus.ARCHIVED
    assert registry.list() == []
    with uow_factory() as uow:
        events = uow.events.list(experiment.experiment_id)
    assert [event.event_type for event in events] == [
        "created",
        "status_transition",
        "status_transition",
    ]


def test_live_registry_requires_refreshable_provider(persistence) -> None:
    _engine, _session_factory, uow_factory = persistence
    registry = ExperimentRegistry(uow_factory)
    with pytest.raises(DomainValidationError, match="refreshable"):
        registry.create(
            name="invalid live",
            mode=ExperimentMode.LIVE_FORWARD,
            base_currency="USD",
            initial_capital=1_000.0,
            recipe=_recipe(refreshable=False),
            training_start=date(2025, 1, 1),
            training_end=date(2025, 12, 31),
            optimization_as_of=date(2025, 12, 31),
            launch_date=date(2026, 1, 1),
        )


def test_snapshot_and_daily_persistence_are_idempotent(persistence) -> None:
    _engine, _session_factory, uow_factory = persistence
    registry = ExperimentRegistry(uow_factory)
    experiment = _create_historical(registry)
    snapshot = _snapshot(experiment.experiment_id)
    assert registry.save_snapshot(snapshot) == "inserted"
    assert registry.save_snapshot(snapshot) == "skipped"
    service = MonitoringPersistenceService(uow_factory)
    normalized = _normalized()
    states = _states(experiment, snapshot)
    first = service.persist(observations=normalized.observations(), states=states)
    second = service.persist(observations=normalized.observations(), states=states)
    assert first["prices"].inserted == 3
    assert first["states"].inserted == 3
    assert second["prices"].skipped == 3
    assert second["states"].skipped == 3
    with uow_factory() as uow:
        restored = uow.valuations.list(experiment.experiment_id)
    assert [item.nav for item in restored] == [1_000.0, 1_100.0, 900.0]


def test_finalized_state_cannot_be_overwritten(persistence) -> None:
    _engine, _session_factory, uow_factory = persistence
    registry = ExperimentRegistry(uow_factory)
    experiment = _create_historical(registry)
    snapshot = _snapshot(experiment.experiment_id)
    registry.save_snapshot(snapshot)
    state = _states(experiment, snapshot)[0]
    service = MonitoringPersistenceService(uow_factory)
    service.persist(observations=_normalized().observations(), states=(state,))
    conflicting = replace(
        state,
        nav=1_001.0,
        base_100_nav=100.1,
        cumulative_return=0.001,
    )
    with pytest.raises(ImmutableRecordError, match="cannot be overwritten"):
        service.persist(observations=(), states=(conflicting,))


def test_incomplete_state_can_be_completed_once(persistence) -> None:
    _engine, _session_factory, uow_factory = persistence
    registry = ExperimentRegistry(uow_factory)
    experiment = _create_historical(registry)
    snapshot = _snapshot(experiment.experiment_id)
    registry.save_snapshot(snapshot)
    complete = _states(experiment, snapshot)[0]
    incomplete_assets = tuple(
        replace(
            item,
            market_value=None,
            current_weight=None,
            drift_percentage_points=None,
        )
        for item in complete.asset_states
    )
    incomplete = replace(
        complete,
        data_quality_status=DataQualityStatus.INCOMPLETE,
        finalized=False,
        asset_states=incomplete_assets,
        nav=None,
        base_100_nav=None,
        cash_value=None,
        daily_return=None,
        cumulative_return=None,
        running_peak=None,
        drawdown=None,
        maximum_drawdown=None,
        total_drift=None,
        return_interval_days=None,
        quality_metadata={"missing_assets": ["BTC"]},
    )
    service = MonitoringPersistenceService(uow_factory)
    first = service.persist(observations=(), states=(incomplete,))
    second = service.persist(observations=(), states=(complete,))
    assert first["states"].inserted == 1
    assert second["states"].updated == 1
    with uow_factory() as uow:
        restored = uow.valuations.get(experiment.experiment_id, complete.state_date)
    assert restored is not None and restored.finalized is True


def test_atomic_failure_rolls_back_prices_when_state_experiment_is_missing(
    persistence,
) -> None:
    _engine, _session_factory, uow_factory = persistence
    normalized = _normalized()
    registry = ExperimentRegistry(uow_factory)
    experiment = _create_historical(registry)
    state = _states(experiment, _snapshot(experiment.experiment_id))[0]
    orphan_id = uuid4()
    orphan = replace(
        state,
        experiment_id=orphan_id,
        asset_states=tuple(
            replace(item, experiment_id=orphan_id) for item in state.asset_states
        ),
    )
    with pytest.raises((RecordNotFoundError, DomainValidationError)):
        MonitoringPersistenceService(uow_factory).persist(
            observations=normalized.observations(), states=(orphan,)
        )
    with uow_factory() as uow:
        assert uow.prices.list() == []


def test_export_bundle_is_secret_free_and_uses_relative_manifest_entries(
    persistence, tmp_path: Path
) -> None:
    _engine, _session_factory, uow_factory = persistence
    registry = ExperimentRegistry(uow_factory)
    experiment = _create_historical(registry)
    snapshot = _snapshot(experiment.experiment_id)
    registry.save_snapshot(snapshot)
    normalized = _normalized()
    MonitoringPersistenceService(uow_factory).persist(
        observations=normalized.observations(),
        states=_states(experiment, snapshot),
    )
    destination = tmp_path / "bundle"
    manifest_path = export_experiment_bundle(
        uow_factory=uow_factory,
        experiment_id=experiment.experiment_id,
        output_directory=destination,
        generated_at=NOW,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["privacy"] == "private_by_default"
    assert manifest["contains_secrets"] is False
    assert manifest["counts"]["portfolio_states"] == 3
    assert set(manifest["files"]) == {
        "daily_asset_states.csv",
        "daily_portfolio_states.csv",
        "experiment.json",
        "experiment_events.csv",
        "optimization_snapshot.json",
        "price_observations.csv",
        "snapshot_allocations.csv",
    }
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in destination.iterdir()
    ).lower()
    assert "/users/" not in combined
    assert "password" not in combined
    assert "api_key" not in combined
