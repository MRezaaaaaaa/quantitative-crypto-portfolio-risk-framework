"""Repository protocols and SQLAlchemy adapters for monitoring aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from types import TracebackType
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from .domain import (
    DuplicateRecordError,
    DailyAssetState,
    DailyPortfolioState,
    DataQualityStatus,
    Experiment,
    ExperimentEvent,
    ExperimentMode,
    ExperimentStatus,
    ImmutableRecordError,
    OptimizationSnapshot,
    PriceDataStatus,
    PriceObservation,
    RecordNotFoundError,
    SnapshotAllocation,
    ensure_utc,
    utc_now,
)
from .models import (
    ExperimentEventModel,
    ExperimentModel,
    DailyAssetStateModel,
    DailyPortfolioStateModel,
    OptimizationSnapshotModel,
    PriceObservationModel,
    SnapshotAllocationModel,
)


@dataclass(frozen=True)
class PersistenceCounts:
    """Idempotent persistence outcome for a batch of explicit records."""

    inserted: int = 0
    updated: int = 0
    skipped: int = 0


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


class PriceRepository(Protocol):
    """Persistence contract for explicit immutable price observations."""

    def add_many(
        self, observations: tuple[PriceObservation, ...] | list[PriceObservation]
    ) -> PersistenceCounts: ...

    def list(
        self,
        *,
        symbols: tuple[str, ...] | list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        source: str | None = None,
        quote_currency: str | None = None,
    ) -> list[PriceObservation]: ...


class ValuationRepository(Protocol):
    """Persistence contract for atomic daily portfolio and asset state."""

    def write(self, state: DailyPortfolioState) -> str: ...

    def get(
        self, experiment_id: UUID, state_date: date
    ) -> DailyPortfolioState | None: ...

    def list(self, experiment_id: UUID) -> list[DailyPortfolioState]: ...


class EventRepository(Protocol):
    """Append-only experiment event contract."""

    def add(self, event: ExperimentEvent) -> None: ...

    def list(self, experiment_id: UUID) -> list[ExperimentEvent]: ...


class UnitOfWork(Protocol):
    """Transaction boundary shared by monitoring services."""

    experiments: ExperimentRepository
    snapshots: SnapshotRepository
    prices: PriceRepository
    valuations: ValuationRepository
    events: EventRepository

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


def _price_to_model(observation: PriceObservation) -> PriceObservationModel:
    return PriceObservationModel(
        symbol=observation.symbol,
        observation_date=observation.observation_date,
        price=observation.price,
        quote_currency=observation.quote_currency,
        source=observation.source,
        retrieved_at=observation.retrieved_at,
        data_status=observation.data_status.value,
    )


def _price_from_model(model: PriceObservationModel) -> PriceObservation:
    retrieved_at = _database_timestamp(model.retrieved_at)
    assert retrieved_at is not None
    return PriceObservation(
        symbol=model.symbol,
        observation_date=model.observation_date,
        price=model.price,
        quote_currency=model.quote_currency,
        source=model.source,
        retrieved_at=retrieved_at,
        data_status=PriceDataStatus(model.data_status),
    )


def _asset_state_to_model(state: DailyAssetState) -> DailyAssetStateModel:
    return DailyAssetStateModel(
        experiment_id=str(state.experiment_id),
        state_date=state.state_date,
        asset=state.asset,
        price=state.price,
        quantity=state.quantity,
        market_value=state.market_value,
        target_weight=state.target_weight,
        current_weight=state.current_weight,
        drift_percentage_points=state.drift_percentage_points,
    )


def _asset_state_from_model(
    model: DailyAssetStateModel, *, cash_symbol: str | None = None
) -> DailyAssetState:
    return DailyAssetState(
        experiment_id=UUID(model.experiment_id),
        state_date=model.state_date,
        asset=model.asset,
        price=model.price,
        quantity=model.quantity,
        market_value=model.market_value,
        target_weight=model.target_weight,
        current_weight=model.current_weight,
        drift_percentage_points=model.drift_percentage_points,
        is_cash=cash_symbol is not None and model.asset == cash_symbol,
    )


def _portfolio_state_to_model(state: DailyPortfolioState) -> DailyPortfolioStateModel:
    return DailyPortfolioStateModel(
        experiment_id=str(state.experiment_id),
        state_date=state.state_date,
        nav=state.nav,
        base_100_nav=state.base_100_nav,
        cash_value=state.cash_value,
        daily_return=state.daily_return,
        cumulative_return=state.cumulative_return,
        realized_volatility=state.realized_volatility,
        running_peak=state.running_peak,
        drawdown=state.drawdown,
        maximum_drawdown=state.maximum_drawdown,
        total_drift=state.total_drift,
        return_interval_days=state.return_interval_days,
        benchmark_nav=state.benchmark_nav,
        benchmark_return=state.benchmark_return,
        quality_metadata_json=dict(state.quality_metadata),
        data_quality_status=state.data_quality_status.value,
        calculation_version=state.calculation_version,
        finalized=state.finalized,
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


def _portfolio_state_from_model(
    model: DailyPortfolioStateModel,
    assets: list[DailyAssetStateModel],
    *,
    cash_symbol: str | None = None,
) -> DailyPortfolioState:
    created_at = _database_timestamp(model.created_at)
    updated_at = _database_timestamp(model.updated_at)
    assert created_at is not None and updated_at is not None
    return DailyPortfolioState(
        experiment_id=UUID(model.experiment_id),
        state_date=model.state_date,
        nav=model.nav,
        base_100_nav=model.base_100_nav,
        cash_value=model.cash_value,
        daily_return=model.daily_return,
        cumulative_return=model.cumulative_return,
        realized_volatility=model.realized_volatility,
        running_peak=model.running_peak,
        drawdown=model.drawdown,
        maximum_drawdown=model.maximum_drawdown,
        total_drift=model.total_drift,
        return_interval_days=model.return_interval_days,
        benchmark_nav=model.benchmark_nav,
        benchmark_return=model.benchmark_return,
        quality_metadata=model.quality_metadata_json,
        data_quality_status=DataQualityStatus(model.data_quality_status),
        calculation_version=model.calculation_version,
        finalized=model.finalized,
        asset_states=tuple(
            _asset_state_from_model(item, cash_symbol=cash_symbol)
            for item in sorted(assets, key=lambda item: item.asset)
        ),
        created_at=created_at,
        updated_at=updated_at,
    )


def _event_to_model(event: ExperimentEvent) -> ExperimentEventModel:
    return ExperimentEventModel(
        event_id=str(event.event_id),
        experiment_id=str(event.experiment_id),
        effective_date=event.effective_date,
        event_type=event.event_type,
        event_metadata_json=dict(event.event_metadata),
        created_at=event.created_at,
    )


def _event_from_model(model: ExperimentEventModel) -> ExperimentEvent:
    created_at = _database_timestamp(model.created_at)
    assert created_at is not None
    return ExperimentEvent(
        event_id=UUID(model.event_id),
        experiment_id=UUID(model.experiment_id),
        effective_date=model.effective_date,
        event_type=model.event_type,
        event_metadata=model.event_metadata_json,
        created_at=created_at,
    )


def _state_content(state: DailyPortfolioState) -> tuple:
    """Comparable immutable state content excluding write timestamps."""
    return (
        state.experiment_id,
        state.state_date,
        state.data_quality_status,
        state.calculation_version,
        state.finalized,
        state.nav,
        state.base_100_nav,
        state.cash_value,
        state.daily_return,
        state.cumulative_return,
        state.realized_volatility,
        state.running_peak,
        state.drawdown,
        state.maximum_drawdown,
        state.total_drift,
        state.return_interval_days,
        state.benchmark_nav,
        state.benchmark_return,
        tuple(sorted(state.quality_metadata.items())),
        tuple(
            (
                item.asset,
                item.price,
                item.quantity,
                item.market_value,
                item.target_weight,
                item.current_weight,
                item.drift_percentage_points,
                item.is_cash,
            )
            for item in sorted(state.asset_states, key=lambda item: item.asset)
        ),
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


class SqlAlchemyPriceRepository:
    """SQLAlchemy adapter for explicit, non-forward-filled prices."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_many(
        self, observations: tuple[PriceObservation, ...] | list[PriceObservation]
    ) -> PersistenceCounts:
        inserted = 0
        skipped = 0
        seen: set[tuple[str, date, str, str]] = set()
        for observation in observations:
            natural_key = (
                observation.symbol,
                observation.observation_date,
                observation.quote_currency,
                observation.source,
            )
            if natural_key in seen:
                raise DuplicateRecordError(
                    f"duplicate price observation in input: {natural_key}"
                )
            seen.add(natural_key)
            existing = self._session.scalar(
                select(PriceObservationModel).where(
                    PriceObservationModel.symbol == observation.symbol,
                    PriceObservationModel.observation_date
                    == observation.observation_date,
                    PriceObservationModel.quote_currency
                    == observation.quote_currency,
                    PriceObservationModel.source == observation.source,
                )
            )
            if existing is not None:
                if (
                    math.isclose(
                        existing.price,
                        observation.price,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                    and existing.data_status == observation.data_status.value
                ):
                    skipped += 1
                    continue
                raise ImmutableRecordError(
                    "existing price observation differs; use the future "
                    "audited correction workflow"
                )
            self._session.add(_price_to_model(observation))
            inserted += 1
        _flush_or_duplicate(
            self._session,
            "one or more price observations already exist",
        )
        return PersistenceCounts(inserted=inserted, skipped=skipped)

    def list(
        self,
        *,
        symbols: tuple[str, ...] | list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        source: str | None = None,
        quote_currency: str | None = None,
    ) -> list[PriceObservation]:
        statement = select(PriceObservationModel)
        if symbols:
            normalized = [str(item).strip().upper() for item in symbols]
            statement = statement.where(PriceObservationModel.symbol.in_(normalized))
        if start is not None:
            statement = statement.where(
                PriceObservationModel.observation_date >= start
            )
        if end is not None:
            statement = statement.where(PriceObservationModel.observation_date <= end)
        if source is not None:
            statement = statement.where(PriceObservationModel.source == source)
        if quote_currency is not None:
            statement = statement.where(
                PriceObservationModel.quote_currency == quote_currency.strip().upper()
            )
        statement = statement.order_by(
            PriceObservationModel.observation_date,
            PriceObservationModel.symbol,
            PriceObservationModel.source,
        )
        return [
            _price_from_model(model)
            for model in self._session.scalars(statement).all()
        ]


class SqlAlchemyValuationRepository:
    """SQLAlchemy adapter for atomic, idempotent daily state writes."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _cash_symbol(self, experiment_id: UUID) -> str | None:
        return self._session.scalar(
            select(SnapshotAllocationModel.asset)
            .join(
                OptimizationSnapshotModel,
                OptimizationSnapshotModel.snapshot_id
                == SnapshotAllocationModel.snapshot_id,
            )
            .where(
                OptimizationSnapshotModel.experiment_id == str(experiment_id),
                SnapshotAllocationModel.is_cash.is_(True),
            )
        )

    def _assets(
        self, experiment_id: UUID, state_date: date
    ) -> list[DailyAssetStateModel]:
        statement = select(DailyAssetStateModel).where(
            DailyAssetStateModel.experiment_id == str(experiment_id),
            DailyAssetStateModel.state_date == state_date,
        )
        return list(self._session.scalars(statement).all())

    def write(self, state: DailyPortfolioState) -> str:
        if self._session.get(ExperimentModel, str(state.experiment_id)) is None:
            raise RecordNotFoundError(
                f"experiment {state.experiment_id} does not exist"
            )
        key = (str(state.experiment_id), state.state_date)
        existing = self._session.get(DailyPortfolioStateModel, key)
        if existing is not None:
            restored = _portfolio_state_from_model(
                existing,
                self._assets(state.experiment_id, state.state_date),
                cash_symbol=self._cash_symbol(state.experiment_id),
            )
            if _state_content(restored) == _state_content(state):
                return "skipped"
            if existing.finalized:
                raise ImmutableRecordError(
                    "finalized daily portfolio state cannot be overwritten"
                )
            self._session.execute(
                delete(DailyAssetStateModel).where(
                    DailyAssetStateModel.experiment_id == str(state.experiment_id),
                    DailyAssetStateModel.state_date == state.state_date,
                )
            )
            replacement = _portfolio_state_to_model(state)
            for column in (
                "nav",
                "base_100_nav",
                "cash_value",
                "daily_return",
                "cumulative_return",
                "realized_volatility",
                "running_peak",
                "drawdown",
                "maximum_drawdown",
                "total_drift",
                "return_interval_days",
                "benchmark_nav",
                "benchmark_return",
                "quality_metadata_json",
                "data_quality_status",
                "calculation_version",
                "finalized",
                "updated_at",
            ):
                setattr(existing, column, getattr(replacement, column))
            action = "updated"
        else:
            existing = _portfolio_state_to_model(state)
            self._session.add(existing)
            action = "inserted"
        self._session.add_all(
            [_asset_state_to_model(item) for item in state.asset_states]
        )
        self._session.add(
            ExperimentEventModel(
                event_id=str(uuid4()),
                experiment_id=str(state.experiment_id),
                effective_date=state.state_date,
                event_type="daily_state_" + action,
                event_metadata_json={
                    "state_date": state.state_date.isoformat(),
                    "quality": state.data_quality_status.value,
                    "finalized": state.finalized,
                },
                created_at=state.updated_at,
            )
        )
        self._session.flush()
        return action

    def get(
        self, experiment_id: UUID, state_date: date
    ) -> DailyPortfolioState | None:
        model = self._session.get(
            DailyPortfolioStateModel, (str(experiment_id), state_date)
        )
        if model is None:
            return None
        return _portfolio_state_from_model(
            model,
            self._assets(experiment_id, state_date),
            cash_symbol=self._cash_symbol(experiment_id),
        )

    def list(self, experiment_id: UUID) -> list[DailyPortfolioState]:
        statement = (
            select(DailyPortfolioStateModel)
            .where(DailyPortfolioStateModel.experiment_id == str(experiment_id))
            .order_by(DailyPortfolioStateModel.state_date)
        )
        cash_symbol = self._cash_symbol(experiment_id)
        return [
            _portfolio_state_from_model(
                model,
                self._assets(experiment_id, model.state_date),
                cash_symbol=cash_symbol,
            )
            for model in self._session.scalars(statement).all()
        ]


class SqlAlchemyEventRepository:
    """SQLAlchemy adapter for append-only audit events."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, event: ExperimentEvent) -> None:
        if self._session.get(ExperimentModel, str(event.experiment_id)) is None:
            raise RecordNotFoundError(
                f"experiment {event.experiment_id} does not exist"
            )
        if self._session.get(ExperimentEventModel, str(event.event_id)) is not None:
            raise DuplicateRecordError(f"event {event.event_id} already exists")
        self._session.add(_event_to_model(event))
        _flush_or_duplicate(self._session, f"event {event.event_id} already exists")

    def list(self, experiment_id: UUID) -> list[ExperimentEvent]:
        statement = (
            select(ExperimentEventModel)
            .where(ExperimentEventModel.experiment_id == str(experiment_id))
            .order_by(
                ExperimentEventModel.created_at,
                ExperimentEventModel.event_id,
            )
        )
        return [
            _event_from_model(model)
            for model in self._session.scalars(statement).all()
        ]


class SqlAlchemyUnitOfWork:
    """Explicit commit/rollback boundary for SQLAlchemy repositories."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self.session: Session | None = None
        self.experiments: SqlAlchemyExperimentRepository
        self.snapshots: SqlAlchemySnapshotRepository
        self.prices: SqlAlchemyPriceRepository
        self.valuations: SqlAlchemyValuationRepository
        self.events: SqlAlchemyEventRepository

    def __enter__(self) -> SqlAlchemyUnitOfWork:
        self.session = self._session_factory()
        self.experiments = SqlAlchemyExperimentRepository(self.session)
        self.snapshots = SqlAlchemySnapshotRepository(self.session)
        self.prices = SqlAlchemyPriceRepository(self.session)
        self.valuations = SqlAlchemyValuationRepository(self.session)
        self.events = SqlAlchemyEventRepository(self.session)
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
