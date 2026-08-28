"""Transactional SQLAlchemy repository tests using only temporary SQLite."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from var_cvar_crypto_risk.monitoring.database import (
    create_monitoring_engine,
    create_session_factory,
)
from var_cvar_crypto_risk.monitoring.domain import (
    DuplicateRecordError,
    Experiment,
    ExperimentMode,
    ExperimentStatus,
    ImmutableRecordError,
    OptimizationSnapshot,
    RecordNotFoundError,
    SnapshotAllocation,
)
from var_cvar_crypto_risk.monitoring.models import (
    Base,
    ExperimentEventModel,
    ExperimentModel,
    OptimizationSnapshotModel,
    SnapshotAllocationModel,
)
from var_cvar_crypto_risk.monitoring.repository import SqlAlchemyUnitOfWork


def _experiment(name: str = "Portfolio experiment") -> Experiment:
    return Experiment.create(
        name=name,
        mode=ExperimentMode.LIVE_FORWARD,
        base_currency="USD",
        initial_capital=100_000.0,
        source_metadata={"provider": "synthetic"},
    )


def _snapshot(experiment_id, *, activated: bool = False) -> OptimizationSnapshot:
    snapshot = OptimizationSnapshot(
        snapshot_id=uuid4(),
        experiment_id=experiment_id,
        package_version="1.0.0",
        code_version="abc123",
        objective="min_cvar",
        solver="CLARABEL",
        solver_status="optimal",
        source_data_hash="a" * 64,
        assumption_recipe_hash="b" * 64,
        assumptions={"confidence": 0.95},
        constraints={"long_only": True},
        launch_forecast={"var": 0.05, "cvar": 0.08},
        scenario_metadata={"source": "historical"},
        return_policy={"wealth": "simple"},
        loss_convention={"losses": "positive"},
        residual_validation={"passed": True},
        allocations=(
            SnapshotAllocation(
                asset="BTC",
                asset_type="crypto",
                target_weight=1.0,
                launch_price=50_000.0,
                initial_value=100_000.0,
                quantity=2.0,
            ),
        ),
    )
    return snapshot.activate() if activated else snapshot


@pytest.fixture
def database(tmp_path: Path):
    path = tmp_path / "repository.db"
    engine = create_monitoring_engine(f"sqlite+pysqlite:///{path.as_posix()}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    try:
        yield path, engine, factory
    finally:
        engine.dispose()


def test_repository_crud_allows_duplicate_names_but_not_duplicate_ids(
    database,
) -> None:
    _path, _engine, factory = database
    first = _experiment("same name")
    second = _experiment("same name")
    with SqlAlchemyUnitOfWork(factory) as uow:
        uow.experiments.add(first)
        uow.experiments.add(second)
        uow.commit()
    with SqlAlchemyUnitOfWork(factory) as uow:
        assert [item.name for item in uow.experiments.list()] == [
            "same name",
            "same name",
        ]
        assert uow.experiments.get(first.experiment_id) == first
    with pytest.raises(DuplicateRecordError):
        with SqlAlchemyUnitOfWork(factory) as uow:
            uow.experiments.add(first)


def test_unit_of_work_without_commit_rolls_back(database) -> None:
    _path, _engine, factory = database
    experiment = _experiment("rollback")
    with SqlAlchemyUnitOfWork(factory) as uow:
        uow.experiments.add(experiment)
    with SqlAlchemyUnitOfWork(factory) as uow:
        assert uow.experiments.get(experiment.experiment_id) is None


def test_unit_of_work_exception_rolls_back(database) -> None:
    _path, _engine, factory = database
    experiment = _experiment("exception rollback")
    with pytest.raises(RuntimeError, match="stop"):
        with SqlAlchemyUnitOfWork(factory) as uow:
            uow.experiments.add(experiment)
            raise RuntimeError("stop")
    with SqlAlchemyUnitOfWork(factory) as uow:
        assert uow.experiments.get(experiment.experiment_id) is None


def test_archive_keeps_history_and_writes_transition_event(database) -> None:
    _path, _engine, factory = database
    experiment = _experiment("archive")
    at = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)
    with SqlAlchemyUnitOfWork(factory) as uow:
        uow.experiments.add(experiment)
        uow.commit()
    with SqlAlchemyUnitOfWork(factory) as uow:
        archived = uow.experiments.archive(experiment.experiment_id, at=at)
        uow.commit()
    assert archived.status is ExperimentStatus.ARCHIVED
    with SqlAlchemyUnitOfWork(factory) as uow:
        assert uow.experiments.list() == []
        assert uow.experiments.list(include_archived=True)[0].archived_at == at
        event_count = uow.session.scalar(
            select(func.count()).select_from(ExperimentEventModel)
        )
        assert event_count == 2


def test_persistence_survives_engine_restart(tmp_path: Path) -> None:
    path = tmp_path / "restart.db"
    url = f"sqlite+pysqlite:///{path.as_posix()}"
    experiment = _experiment("restart")
    first_engine = create_monitoring_engine(url)
    Base.metadata.create_all(first_engine)
    with SqlAlchemyUnitOfWork(create_session_factory(first_engine)) as uow:
        uow.experiments.add(experiment)
        uow.commit()
    first_engine.dispose()
    second_engine = create_monitoring_engine(url)
    try:
        with SqlAlchemyUnitOfWork(create_session_factory(second_engine)) as uow:
            restored = uow.experiments.get(experiment.experiment_id)
            assert restored == experiment
    finally:
        second_engine.dispose()


def test_repository_rejects_snapshot_without_experiment(database) -> None:
    _path, _engine, factory = database
    snapshot = _snapshot(uuid4())
    with pytest.raises(RecordNotFoundError):
        with SqlAlchemyUnitOfWork(factory) as uow:
            uow.snapshots.add(snapshot)


def test_database_foreign_key_rejects_orphan_snapshot(database) -> None:
    _path, _engine, factory = database
    session = factory()
    try:
        snapshot = _snapshot(uuid4())
        session.add(
            OptimizationSnapshotModel(
                snapshot_id=str(snapshot.snapshot_id),
                experiment_id=str(snapshot.experiment_id),
                package_version=snapshot.package_version,
                code_version=snapshot.code_version,
                objective=snapshot.objective,
                solver=snapshot.solver,
                solver_status=snapshot.solver_status,
                assumptions_json={},
                constraints_json={},
                launch_forecast_json={},
                scenario_metadata_json={},
                return_policy_json={},
                loss_convention_json={},
                residual_validation_json={"passed": True},
                source_data_hash=snapshot.source_data_hash,
                assumption_recipe_hash=snapshot.assumption_recipe_hash,
                created_at=snapshot.created_at,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()


def test_one_snapshot_per_experiment_and_round_trip(database) -> None:
    _path, _engine, factory = database
    experiment = _experiment("snapshot")
    snapshot = _snapshot(experiment.experiment_id)
    with SqlAlchemyUnitOfWork(factory) as uow:
        uow.experiments.add(experiment)
        uow.snapshots.add(snapshot)
        uow.commit()
    with SqlAlchemyUnitOfWork(factory) as uow:
        assert uow.snapshots.get_for_experiment(experiment.experiment_id) == snapshot
    with pytest.raises(DuplicateRecordError):
        with SqlAlchemyUnitOfWork(factory) as uow:
            uow.snapshots.add(_snapshot(experiment.experiment_id))


def test_repository_activation_freezes_snapshot_and_allocations(database) -> None:
    _path, _engine, factory = database
    experiment = _experiment("immutable")
    snapshot = _snapshot(experiment.experiment_id)
    activated_at = datetime(2026, 7, 2, 12, tzinfo=timezone.utc)
    with SqlAlchemyUnitOfWork(factory) as uow:
        uow.experiments.add(experiment)
        uow.snapshots.add(snapshot)
        uow.commit()
    with SqlAlchemyUnitOfWork(factory) as uow:
        activated = uow.snapshots.activate(
            experiment.experiment_id, at=activated_at
        )
        uow.commit()
    assert activated.activated_at == activated_at
    session = factory()
    try:
        model = session.get(OptimizationSnapshotModel, str(snapshot.snapshot_id))
        model.objective = "max_return"
        with pytest.raises(ImmutableRecordError):
            session.commit()
        session.rollback()
        allocation = session.get(
            SnapshotAllocationModel, (str(snapshot.snapshot_id), "BTC")
        )
        allocation.target_weight = 0.5
        with pytest.raises(ImmutableRecordError):
            session.commit()
    finally:
        session.rollback()
        session.close()


def test_activated_snapshot_can_be_inserted_once_but_not_deleted(database) -> None:
    _path, _engine, factory = database
    experiment = _experiment("pre-activated")
    snapshot = _snapshot(experiment.experiment_id, activated=True)
    with SqlAlchemyUnitOfWork(factory) as uow:
        uow.experiments.add(experiment)
        uow.snapshots.add(snapshot)
        uow.commit()
    session = factory()
    try:
        model = session.get(OptimizationSnapshotModel, str(snapshot.snapshot_id))
        session.delete(model)
        with pytest.raises(ImmutableRecordError):
            session.commit()
    finally:
        session.rollback()
        session.close()


def test_database_check_constraints_reject_invalid_mode(database) -> None:
    _path, _engine, factory = database
    session = factory()
    try:
        now = datetime.now(timezone.utc)
        session.add(
            ExperimentModel(
                experiment_id=str(uuid4()),
                name="invalid mode",
                mode="not-a-mode",
                status="draft",
                base_currency="USD",
                initial_capital=100.0,
                source_metadata_json={},
                schema_version="1",
                created_at=now,
                updated_at=now,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()
