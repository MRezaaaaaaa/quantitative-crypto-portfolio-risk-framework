"""Repository protocols and SQLAlchemy adapters for monitoring aggregates."""

from __future__ import annotations

from datetime import datetime, timezone
from types import TracebackType
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from .domain import (
    DuplicateRecordError,
    Experiment,
    ExperimentMode,
    ExperimentStatus,
    OptimizationSnapshot,
    RecordNotFoundError,
    SnapshotAllocation,
    ensure_utc,
    utc_now,
)
from .models import (
    ExperimentEventModel,
    ExperimentModel,
    OptimizationSnapshotModel,
    SnapshotAllocationModel,
)


def _database_timestamp(value: datetime | None) -> datetime | None:
    """Interpret SQLite-naive values as UTC while retaining aware values."""
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class ExperimentRepository(Protocol):
    """Persistence contract for experiment aggregates."""

    def add(self, experiment: Experiment) -> None: ...

    def get(self, experiment_id: UUID) -> Experiment | None: ...

    def list(self, *, include_archived: bool = False) -> list[Experiment]: ...

    def transition(
        self,
        experiment_id: UUID,
        target: ExperimentStatus,
        *,
        at: datetime | None = None,
    ) -> Experiment: ...

    def archive(
        self, experiment_id: UUID, *, at: datetime | None = None
    ) -> Experiment: ...


class SnapshotRepository(Protocol):
    """Persistence contract for immutable optimization snapshots."""

    def add(self, snapshot: OptimizationSnapshot) -> None: ...

    def get_for_experiment(
        self, experiment_id: UUID
    ) -> OptimizationSnapshot | None: ...

    def activate(
        self, experiment_id: UUID, *, at: datetime | None = None
    ) -> OptimizationSnapshot: ...


class UnitOfWork(Protocol):
    """Transaction boundary shared by monitoring services."""

    experiments: ExperimentRepository
    snapshots: SnapshotRepository

    def __enter__(self) -> UnitOfWork: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


def _experiment_to_model(experiment: Experiment) -> ExperimentModel:
    return ExperimentModel(
        experiment_id=str(experiment.experiment_id),
        name=experiment.name,
        description=experiment.description,
        mode=experiment.mode.value,
        status=experiment.status.value,
        base_currency=experiment.base_currency,
        initial_capital=experiment.initial_capital,
        benchmark_symbol=experiment.benchmark_symbol,
        training_start=experiment.training_start,
        training_end=experiment.training_end,
        optimization_as_of=experiment.optimization_as_of,
        launch_date=experiment.launch_date,
        historical_evaluation_end=experiment.historical_evaluation_end,
        live_tracking_end=experiment.live_tracking_end,
        source_metadata_json=dict(experiment.source_metadata),
        schema_version=experiment.schema_version,
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
        archived_at=experiment.archived_at,
    )


def _experiment_from_model(model: ExperimentModel) -> Experiment:
    return Experiment(
        experiment_id=UUID(model.experiment_id),
        name=model.name,
        description=model.description,
        mode=ExperimentMode(model.mode),
        status=ExperimentStatus(model.status),
        base_currency=model.base_currency,
        initial_capital=model.initial_capital,
        benchmark_symbol=model.benchmark_symbol,
        training_start=model.training_start,
        training_end=model.training_end,
        optimization_as_of=model.optimization_as_of,
        launch_date=model.launch_date,
        historical_evaluation_end=model.historical_evaluation_end,
        live_tracking_end=model.live_tracking_end,
        source_metadata=model.source_metadata_json,
        schema_version=model.schema_version,
        created_at=_database_timestamp(model.created_at),
        updated_at=_database_timestamp(model.updated_at),
        archived_at=_database_timestamp(model.archived_at),
    )


def _snapshot_to_model(snapshot: OptimizationSnapshot) -> OptimizationSnapshotModel:
    model = OptimizationSnapshotModel(
        snapshot_id=str(snapshot.snapshot_id),
        experiment_id=str(snapshot.experiment_id),
        package_version=snapshot.package_version,
        code_version=snapshot.code_version,
        objective=snapshot.objective,
        solver=snapshot.solver,
        solver_status=snapshot.solver_status,
        assumptions_json=dict(snapshot.assumptions),
        constraints_json=dict(snapshot.constraints),
        launch_forecast_json=dict(snapshot.launch_forecast),
        scenario_metadata_json=dict(snapshot.scenario_metadata),
        return_policy_json=dict(snapshot.return_policy),
        loss_convention_json=dict(snapshot.loss_convention),
        residual_validation_json=dict(snapshot.residual_validation),
        source_data_hash=snapshot.source_data_hash,
        assumption_recipe_hash=snapshot.assumption_recipe_hash,
        created_at=snapshot.created_at,
        activated_at=snapshot.activated_at,
    )
    model.allocations = [
        SnapshotAllocationModel(
            snapshot_id=str(snapshot.snapshot_id),
            asset=item.asset,
            asset_type=item.asset_type,
            target_weight=item.target_weight,
            launch_price=item.launch_price,
            initial_value=item.initial_value,
            quantity=item.quantity,
            is_cash=item.is_cash,
        )
        for item in snapshot.allocations
    ]
    return model


def _snapshot_from_model(model: OptimizationSnapshotModel) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        snapshot_id=UUID(model.snapshot_id),
        experiment_id=UUID(model.experiment_id),
        package_version=model.package_version,
        code_version=model.code_version,
        objective=model.objective,
        solver=model.solver,
        solver_status=model.solver_status,
        source_data_hash=model.source_data_hash,
        assumption_recipe_hash=model.assumption_recipe_hash,
        assumptions=model.assumptions_json,
        constraints=model.constraints_json,
        launch_forecast=model.launch_forecast_json,
        scenario_metadata=model.scenario_metadata_json,
        return_policy=model.return_policy_json,
        loss_convention=model.loss_convention_json,
        residual_validation=model.residual_validation_json,
        allocations=tuple(
            SnapshotAllocation(
                asset=item.asset,
                asset_type=item.asset_type,
                target_weight=item.target_weight,
                launch_price=item.launch_price,
                initial_value=item.initial_value,
                quantity=item.quantity,
                is_cash=item.is_cash,
            )
            for item in sorted(model.allocations, key=lambda item: item.asset)
        ),
        created_at=_database_timestamp(model.created_at),
        activated_at=_database_timestamp(model.activated_at),
    )


def _flush_or_duplicate(session: Session, message: str) -> None:
    try:
        session.flush()
    except IntegrityError as exc:
        raise DuplicateRecordError(message) from exc


class SqlAlchemyExperimentRepository:
    """SQLAlchemy adapter that returns only experiment domain objects."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, experiment: Experiment) -> None:
        if self._session.get(ExperimentModel, str(experiment.experiment_id)) is not None:
            raise DuplicateRecordError(
                f"experiment {experiment.experiment_id} already exists"
            )
        self._session.add(_experiment_to_model(experiment))
        _flush_or_duplicate(
            self._session,
            f"experiment {experiment.experiment_id} already exists",
        )
        self._session.add(
            ExperimentEventModel(
                event_id=str(uuid4()),
                experiment_id=str(experiment.experiment_id),
                effective_date=experiment.created_at.date(),
                event_type="created",
                event_metadata_json={
                    "mode": experiment.mode.value,
                    "status": experiment.status.value,
                },
                created_at=experiment.created_at,
            )
        )
        self._session.flush()

    def get(self, experiment_id: UUID) -> Experiment | None:
        model = self._session.get(ExperimentModel, str(experiment_id))
        return _experiment_from_model(model) if model is not None else None

    def list(self, *, include_archived: bool = False) -> list[Experiment]:
        statement = select(ExperimentModel)
        if not include_archived:
            statement = statement.where(
                ExperimentModel.status != ExperimentStatus.ARCHIVED.value
            )
        statement = statement.order_by(
            ExperimentModel.created_at, ExperimentModel.experiment_id
        )
        return [
            _experiment_from_model(model)
            for model in self._session.scalars(statement).all()
        ]

    def transition(
        self,
        experiment_id: UUID,
        target: ExperimentStatus,
        *,
        at: datetime | None = None,
    ) -> Experiment:
        model = self._session.get(ExperimentModel, str(experiment_id))
        if model is None:
            raise RecordNotFoundError(f"experiment {experiment_id} does not exist")
        current = _experiment_from_model(model)
        transitioned = current.transition(target, at=at)
        model.status = transitioned.status.value
        model.updated_at = transitioned.updated_at
        model.archived_at = transitioned.archived_at
        self._session.add(
            ExperimentEventModel(
                event_id=str(uuid4()),
                experiment_id=str(experiment_id),
                effective_date=transitioned.updated_at.date(),
                event_type="status_transition",
                event_metadata_json={
                    "from": current.status.value,
                    "to": target.value,
                },
                created_at=transitioned.updated_at,
            )
        )
        self._session.flush()
        return transitioned

    def archive(
        self, experiment_id: UUID, *, at: datetime | None = None
    ) -> Experiment:
        return self.transition(
            experiment_id, ExperimentStatus.ARCHIVED, at=at
        )


class SqlAlchemySnapshotRepository:
    """SQLAlchemy adapter for one immutable snapshot per experiment."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, snapshot: OptimizationSnapshot) -> None:
        experiment = self._session.get(
            ExperimentModel, str(snapshot.experiment_id)
        )
        if experiment is None:
            raise RecordNotFoundError(
                f"experiment {snapshot.experiment_id} does not exist"
            )
        existing = self._session.scalar(
            select(OptimizationSnapshotModel.snapshot_id).where(
                OptimizationSnapshotModel.experiment_id
                == str(snapshot.experiment_id)
            )
        )
        if existing is not None:
            raise DuplicateRecordError(
                f"experiment {snapshot.experiment_id} already has a snapshot"
            )
        self._session.add(_snapshot_to_model(snapshot))
        _flush_or_duplicate(
            self._session,
            f"experiment {snapshot.experiment_id} already has a snapshot",
        )
        self._session.add(
            ExperimentEventModel(
                event_id=str(uuid4()),
                experiment_id=str(snapshot.experiment_id),
                effective_date=snapshot.created_at.date(),
                event_type="optimization_snapshot_saved",
                event_metadata_json={"snapshot_id": str(snapshot.snapshot_id)},
                created_at=snapshot.created_at,
            )
        )
        self._session.flush()

    def get_for_experiment(
        self, experiment_id: UUID
    ) -> OptimizationSnapshot | None:
        statement = (
            select(OptimizationSnapshotModel)
            .options(selectinload(OptimizationSnapshotModel.allocations))
            .where(
                OptimizationSnapshotModel.experiment_id == str(experiment_id)
            )
        )
        model = self._session.scalar(statement)
        return _snapshot_from_model(model) if model is not None else None

    def activate(
        self, experiment_id: UUID, *, at: datetime | None = None
    ) -> OptimizationSnapshot:
        statement = (
            select(OptimizationSnapshotModel)
            .options(selectinload(OptimizationSnapshotModel.allocations))
            .where(
                OptimizationSnapshotModel.experiment_id == str(experiment_id)
            )
        )
        model = self._session.scalar(statement)
        if model is None:
            raise RecordNotFoundError(
                f"snapshot for experiment {experiment_id} does not exist"
            )
        current = _snapshot_from_model(model)
        activated = current.activate(at=ensure_utc(at or utc_now(), "activated_at"))
        model.activated_at = activated.activated_at
        self._session.add(
            ExperimentEventModel(
                event_id=str(uuid4()),
                experiment_id=str(experiment_id),
                effective_date=activated.activated_at.date(),
                event_type="optimization_snapshot_activated",
                event_metadata_json={"snapshot_id": str(activated.snapshot_id)},
                created_at=activated.activated_at,
            )
        )
        self._session.flush()
        return activated


class SqlAlchemyUnitOfWork:
    """Explicit commit/rollback boundary for SQLAlchemy repositories."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self.session: Session | None = None
        self.experiments: SqlAlchemyExperimentRepository
        self.snapshots: SqlAlchemySnapshotRepository

    def __enter__(self) -> SqlAlchemyUnitOfWork:
        self.session = self._session_factory()
        self.experiments = SqlAlchemyExperimentRepository(self.session)
        self.snapshots = SqlAlchemySnapshotRepository(self.session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.session is None:
            return
        try:
            self.session.rollback()
        finally:
            self.session.close()
            self.session = None

    def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("unit of work is not active")
        self.session.commit()

    def rollback(self) -> None:
        if self.session is None:
            raise RuntimeError("unit of work is not active")
        self.session.rollback()
