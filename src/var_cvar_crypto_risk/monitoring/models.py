"""SQLAlchemy persistence models for Phase 8 portfolio monitoring."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    event,
    inspect,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

from .domain import ImmutableRecordError, utc_now


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base with deterministic constraint naming."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class ExperimentModel(Base):
    """Persistent experiment identity and point-in-time boundaries."""

    __tablename__ = "experiments"
    __table_args__ = (
        CheckConstraint(
            "mode IN ('historical_oos', 'live_forward', 'hybrid')",
            name="mode_values",
        ),
        CheckConstraint(
            "status IN ('draft', 'backfilling', 'active', 'completed', "
            "'failed', 'archived')",
            name="status_values",
        ),
        CheckConstraint("initial_capital > 0", name="positive_initial_capital"),
        Index("ix_experiments_name", "name"),
        Index("ix_experiments_status", "status"),
        Index("ix_experiments_mode", "mode"),
        Index("ix_experiments_launch_date", "launch_date"),
        Index("ix_experiments_updated_at", "updated_at"),
    )

    experiment_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    base_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    initial_capital: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_symbol: Mapped[str | None] = mapped_column(String(64))
    training_start: Mapped[date | None] = mapped_column(Date)
    training_end: Mapped[date | None] = mapped_column(Date)
    optimization_as_of: Mapped[date | None] = mapped_column(Date)
    launch_date: Mapped[date | None] = mapped_column(Date)
    historical_evaluation_end: Mapped[date | None] = mapped_column(Date)
    live_tracking_end: Mapped[date | None] = mapped_column(Date)
    source_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "source_metadata", JSON, nullable=False, default=dict
    )
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    snapshot: Mapped[OptimizationSnapshotModel | None] = relationship(
        back_populates="experiment", uselist=False
    )


class OptimizationSnapshotModel(Base):
    """One immutable optimization recipe and result per experiment."""

    __tablename__ = "optimization_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            name="uq_optimization_snapshots_one_per_experiment",
        ),
        CheckConstraint("length(source_data_hash) = 64", name="source_hash_length"),
        CheckConstraint(
            "length(assumption_recipe_hash) = 64", name="recipe_hash_length"
        ),
    )

    snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("experiments.experiment_id", ondelete="RESTRICT"),
        nullable=False,
    )
    package_version: Mapped[str] = mapped_column(String(64), nullable=False)
    code_version: Mapped[str] = mapped_column(String(128), nullable=False)
    objective: Mapped[str] = mapped_column(String(128), nullable=False)
    solver: Mapped[str] = mapped_column(String(128), nullable=False)
    solver_status: Mapped[str] = mapped_column(String(64), nullable=False)
    assumptions_json: Mapped[dict[str, Any]] = mapped_column(
        "assumptions", JSON, nullable=False
    )
    constraints_json: Mapped[dict[str, Any]] = mapped_column(
        "constraints", JSON, nullable=False
    )
    launch_forecast_json: Mapped[dict[str, Any]] = mapped_column(
        "launch_forecast", JSON, nullable=False
    )
    scenario_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "scenario_metadata", JSON, nullable=False
    )
    return_policy_json: Mapped[dict[str, Any]] = mapped_column(
        "return_policy", JSON, nullable=False
    )
    loss_convention_json: Mapped[dict[str, Any]] = mapped_column(
        "loss_convention", JSON, nullable=False
    )
    residual_validation_json: Mapped[dict[str, Any]] = mapped_column(
        "residual_validation", JSON, nullable=False
    )
    source_data_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    assumption_recipe_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    experiment: Mapped[ExperimentModel] = relationship(back_populates="snapshot")
    allocations: Mapped[list[SnapshotAllocationModel]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan"
    )


class SnapshotAllocationModel(Base):
    """Target allocation frozen with an optimization snapshot."""

    __tablename__ = "snapshot_allocations"

    snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("optimization_snapshots.snapshot_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    asset: Mapped[str] = mapped_column(String(64), primary_key=True)
    asset_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_weight: Mapped[float] = mapped_column(Float, nullable=False)
    launch_price: Mapped[float | None] = mapped_column(Float)
    initial_value: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    is_cash: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    snapshot: Mapped[OptimizationSnapshotModel] = relationship(
        back_populates="allocations"
    )


class PriceObservationModel(Base):
    """One normalized daily close from one declared source."""

    __tablename__ = "price_observations"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "observation_date",
            "quote_currency",
            "source",
            name="uq_price_observations_natural_key",
        ),
        CheckConstraint("price > 0", name="positive_price"),
        CheckConstraint(
            "data_status IN ('complete', 'incomplete', 'corrected', 'rejected')",
            name="data_status_values",
        ),
        Index("ix_price_observations_date", "observation_date"),
        Index("ix_price_observations_symbol_date", "symbol", "observation_date"),
    )

    observation_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    observation_date: Mapped[date] = mapped_column(Date, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    quote_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    data_status: Mapped[str] = mapped_column(String(32), nullable=False)


class DailyPortfolioStateModel(Base):
    """Daily portfolio-level valuation, risk, and data-quality state."""

    __tablename__ = "daily_portfolio_states"
    __table_args__ = (
        CheckConstraint(
            "data_quality_status IN ('complete', 'incomplete', 'missing', "
            "'partial', 'corrected')",
            name="quality_status_values",
        ),
        Index("ix_daily_portfolio_states_experiment_date", "experiment_id", "state_date"),
        Index("ix_daily_portfolio_states_quality", "data_quality_status"),
    )

    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("experiments.experiment_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    state_date: Mapped[date] = mapped_column(Date, primary_key=True)
    nav: Mapped[float | None] = mapped_column(Float)
    base_100_nav: Mapped[float | None] = mapped_column(Float)
    cash_value: Mapped[float | None] = mapped_column(Float)
    daily_return: Mapped[float | None] = mapped_column(Float)
    cumulative_return: Mapped[float | None] = mapped_column(Float)
    realized_volatility: Mapped[float | None] = mapped_column(Float)
    running_peak: Mapped[float | None] = mapped_column(Float)
    drawdown: Mapped[float | None] = mapped_column(Float)
    maximum_drawdown: Mapped[float | None] = mapped_column(Float)
    total_drift: Mapped[float | None] = mapped_column(Float)
    return_interval_days: Mapped[int | None] = mapped_column(Integer)
    benchmark_nav: Mapped[float | None] = mapped_column(Float)
    benchmark_return: Mapped[float | None] = mapped_column(Float)
    quality_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "quality_metadata", JSON, nullable=False, default=dict
    )
    data_quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    calculation_version: Mapped[str] = mapped_column(String(64), nullable=False)
    finalized: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class DailyAssetStateModel(Base):
    """Daily per-asset fixed-quantity valuation and weight drift."""

    __tablename__ = "daily_asset_states"

    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("experiments.experiment_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    state_date: Mapped[date] = mapped_column(Date, primary_key=True)
    asset: Mapped[str] = mapped_column(String(64), primary_key=True)
    price: Mapped[float | None] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    market_value: Mapped[float | None] = mapped_column(Float)
    target_weight: Mapped[float] = mapped_column(Float, nullable=False)
    current_weight: Mapped[float | None] = mapped_column(Float)
    drift_percentage_points: Mapped[float | None] = mapped_column(Float)


class DailyRiskForecastModel(Base):
    """Origin-safe VaR/CVaR forecast and later realized evaluation."""

    __tablename__ = "daily_risk_forecasts"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "origin_date",
            "target_date",
            "horizon_days",
            "evaluation_mode",
            "var_method",
            "cvar_method",
            "confidence_level",
            "model_version",
            name="uq_daily_risk_forecasts_natural_key",
        ),
        CheckConstraint("horizon_days > 0", name="positive_horizon"),
        CheckConstraint(
            "confidence_level > 0 AND confidence_level < 1",
            name="confidence_range",
        ),
        CheckConstraint(
            "evaluation_status IN ('pending', 'evaluated', 'insufficient_window')",
            name="evaluation_status_values",
        ),
        Index("ix_daily_risk_forecasts_experiment_origin", "experiment_id", "origin_date"),
        Index("ix_daily_risk_forecasts_target_status", "target_date", "evaluation_status"),
    )

    forecast_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("experiments.experiment_id", ondelete="RESTRICT"),
        nullable=False,
    )
    origin_date: Mapped[date] = mapped_column(Date, nullable=False)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluation_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    estimation_window: Mapped[int] = mapped_column(Integer, nullable=False)
    var_method: Mapped[str] = mapped_column(String(64), nullable=False)
    cvar_method: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence_level: Mapped[float] = mapped_column(Float, nullable=False)
    horizon_construction: Mapped[str] = mapped_column(String(64), nullable=False)
    convention_version: Mapped[str] = mapped_column(String(64), nullable=False)
    portfolio_definition: Mapped[str | None] = mapped_column(String(64))
    input_max_date: Mapped[date | None] = mapped_column(Date)
    input_data_hash: Mapped[str | None] = mapped_column(String(64))
    forecast_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "forecast_metadata", JSON, nullable=False, default=dict
    )
    forecast_var: Mapped[float | None] = mapped_column(Float)
    forecast_cvar: Mapped[float | None] = mapped_column(Float)
    forecast_volatility: Mapped[float | None] = mapped_column(Float)
    realized_horizon_loss: Mapped[float | None] = mapped_column(Float)
    var_breach: Mapped[bool | None] = mapped_column(Boolean)
    evaluation_status: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MonitoringRunModel(Base):
    """Auditable execution record for backfill and live update attempts."""

    __tablename__ = "monitoring_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed')", name="status_values"
        ),
        Index("ix_monitoring_runs_experiment_started", "experiment_id", "started_at"),
    )

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("experiments.experiment_id", ondelete="RESTRICT"),
        nullable=False,
    )
    run_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_cutoff: Mapped[date | None] = mapped_column(Date)
    actual_cutoff: Mapped[date | None] = mapped_column(Date)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    inserted_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    warning_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_summary: Mapped[str | None] = mapped_column(Text)


class ExperimentEventModel(Base):
    """Append-only lifecycle and correction event."""

    __tablename__ = "experiment_events"
    __table_args__ = (
        Index("ix_experiment_events_experiment_created", "experiment_id", "created_at"),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("experiments.experiment_id", ondelete="RESTRICT"),
        nullable=False,
    )
    effective_date: Mapped[date | None] = mapped_column(Date)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    event_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "event_metadata", JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


def _snapshot_was_activated(snapshot: OptimizationSnapshotModel) -> bool:
    state = inspect(snapshot)
    history = state.attrs.activated_at.history
    if history.deleted:
        return history.deleted[0] is not None
    if not history.has_changes():
        return snapshot.activated_at is not None
    return False


def _allocation_has_activated_parent(
    session: Session, allocation: SnapshotAllocationModel
) -> bool:
    snapshot = allocation.snapshot
    if snapshot is not None:
        if snapshot in session.new:
            return False
        return snapshot.activated_at is not None
    parent = session.get(OptimizationSnapshotModel, allocation.snapshot_id)
    return parent is not None and parent.activated_at is not None


def _portfolio_state_was_finalized(state: DailyPortfolioStateModel) -> bool:
    inspected = inspect(state)
    history = inspected.attrs.finalized.history
    if history.deleted:
        return bool(history.deleted[0])
    if not history.has_changes():
        return bool(state.finalized)
    return False


def _forecast_was_evaluated(forecast: DailyRiskForecastModel) -> bool:
    inspected = inspect(forecast)
    history = inspected.attrs.evaluation_status.history
    if history.deleted:
        return history.deleted[0] == "evaluated"
    if not history.has_changes():
        return forecast.evaluation_status == "evaluated"
    return False


def _asset_has_finalized_parent(
    session: Session, asset_state: DailyAssetStateModel
) -> bool:
    for candidate in session.new:
        if isinstance(candidate, DailyPortfolioStateModel) and (
            candidate.experiment_id == asset_state.experiment_id
            and candidate.state_date == asset_state.state_date
        ):
            return False
    with session.no_autoflush:
        parent = session.get(
            DailyPortfolioStateModel,
            (asset_state.experiment_id, asset_state.state_date),
        )
    return parent is not None and _portfolio_state_was_finalized(parent)


@event.listens_for(Session, "before_flush")
def _protect_activated_snapshots(session: Session, _flush_context, _instances) -> None:
    for snapshot in set(session.dirty).union(session.deleted):
        if isinstance(snapshot, OptimizationSnapshotModel) and _snapshot_was_activated(
            snapshot
        ):
            raise ImmutableRecordError("activated optimization snapshot is immutable")
    allocations = set(session.new).union(session.dirty).union(session.deleted)
    for allocation in allocations:
        if isinstance(allocation, SnapshotAllocationModel) and (
            _allocation_has_activated_parent(session, allocation)
        ):
            raise ImmutableRecordError(
                "allocations of an activated optimization snapshot are immutable"
            )
    for price in set(session.dirty).union(session.deleted):
        if isinstance(price, PriceObservationModel):
            raise ImmutableRecordError(
                "explicit price observations require a future correction workflow"
            )
    for state in set(session.dirty).union(session.deleted):
        if isinstance(state, DailyPortfolioStateModel) and (
            _portfolio_state_was_finalized(state)
        ):
            raise ImmutableRecordError("finalized daily portfolio state is immutable")
    for forecast in set(session.dirty).union(session.deleted):
        if isinstance(forecast, DailyRiskForecastModel) and _forecast_was_evaluated(
            forecast
        ):
            raise ImmutableRecordError("evaluated daily risk forecast is immutable")
    asset_states = set(session.new).union(session.dirty).union(session.deleted)
    for asset_state in asset_states:
        if isinstance(asset_state, DailyAssetStateModel) and (
            _asset_has_finalized_parent(session, asset_state)
        ):
            raise ImmutableRecordError(
                "asset rows of a finalized daily portfolio state are immutable"
            )
