"""Domain values and invariants for portfolio monitoring persistence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from enum import Enum
import math
import re
from typing import Any, Mapping
from uuid import UUID, uuid4


class MonitoringError(Exception):
    """Base class for stable monitoring-domain errors."""


class DomainValidationError(MonitoringError, ValueError):
    """A domain value violates an explicit monitoring contract."""


class InvalidTransitionError(MonitoringError, ValueError):
    """An experiment lifecycle transition is not permitted."""


class DuplicateRecordError(MonitoringError):
    """A persistence uniqueness constraint was violated."""


class RecordNotFoundError(MonitoringError):
    """A requested persistent record does not exist."""


class ImmutableRecordError(MonitoringError):
    """An activated or finalized record cannot be changed."""


class ExperimentMode(str, Enum):
    """Supported point-in-time experiment workflows."""

    HISTORICAL_OOS = "historical_oos"
    LIVE_FORWARD = "live_forward"
    HYBRID = "hybrid"


class ExperimentStatus(str, Enum):
    """Persistent experiment lifecycle states."""

    DRAFT = "draft"
    BACKFILLING = "backfilling"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DataQualityStatus(str, Enum):
    """Portfolio-date quality without inventing missing market observations."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    MISSING = "missing"
    PARTIAL = "partial"
    CORRECTED = "corrected"


class PriceDataStatus(str, Enum):
    """Quality state for one explicit source observation."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    CORRECTED = "corrected"
    REJECTED = "rejected"


ALLOWED_TRANSITIONS: Mapping[ExperimentStatus, frozenset[ExperimentStatus]] = {
    ExperimentStatus.DRAFT: frozenset(
        {
            ExperimentStatus.BACKFILLING,
            ExperimentStatus.ACTIVE,
            ExperimentStatus.ARCHIVED,
        }
    ),
    ExperimentStatus.BACKFILLING: frozenset(
        {
            ExperimentStatus.ACTIVE,
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.ARCHIVED,
        }
    ),
    ExperimentStatus.ACTIVE: frozenset(
        {
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.ARCHIVED,
        }
    ),
    ExperimentStatus.FAILED: frozenset(
        {
            ExperimentStatus.BACKFILLING,
            ExperimentStatus.ACTIVE,
            ExperimentStatus.ARCHIVED,
        }
    ),
    ExperimentStatus.COMPLETED: frozenset({ExperimentStatus.ARCHIVED}),
    ExperimentStatus.ARCHIVED: frozenset(),
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime, field_name: str) -> datetime:
    """Normalize an aware timestamp to UTC and reject ambiguous naive values."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainValidationError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def validate_transition(
    current: ExperimentStatus, target: ExperimentStatus
) -> None:
    """Raise when a lifecycle transition is not in the reviewed state machine."""
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(
            f"transition from {current.value!r} to {target.value!r} is not allowed"
        )


def validate_date_boundaries(
    *,
    mode: ExperimentMode,
    training_start: date | None,
    training_end: date | None,
    optimization_as_of: date | None,
    launch_date: date | None,
    historical_evaluation_end: date | None,
    live_tracking_end: date | None,
    require_complete: bool = False,
) -> None:
    """Validate point-in-time boundaries without silently inventing dates."""
    common = (training_start, training_end, optimization_as_of, launch_date)
    if require_complete and any(item is None for item in common):
        raise DomainValidationError(
            "training, optimization, and launch dates must all be set"
        )
    if (
        training_start is not None
        and training_end is not None
        and training_start > training_end
    ):
        raise DomainValidationError("training_start must not exceed training_end")
    if (
        training_end is not None
        and optimization_as_of is not None
        and training_end > optimization_as_of
    ):
        raise DomainValidationError(
            "training_end must not exceed optimization_as_of"
        )
    if (
        optimization_as_of is not None
        and launch_date is not None
        and optimization_as_of >= launch_date
    ):
        raise DomainValidationError("launch_date must follow optimization_as_of")
    if mode in {ExperimentMode.HISTORICAL_OOS, ExperimentMode.HYBRID}:
        if require_complete and historical_evaluation_end is None:
            raise DomainValidationError(
                "historical_evaluation_end is required for historical modes"
            )
        if (
            launch_date is not None
            and historical_evaluation_end is not None
            and launch_date > historical_evaluation_end
        ):
            raise DomainValidationError(
                "launch_date must not exceed historical_evaluation_end"
            )
    if (
        launch_date is not None
        and live_tracking_end is not None
        and launch_date > live_tracking_end
    ):
        raise DomainValidationError("live_tracking_end must not precede launch_date")
    if (
        mode is ExperimentMode.HYBRID
        and historical_evaluation_end is not None
        and live_tracking_end is not None
        and historical_evaluation_end > live_tracking_end
    ):
        raise DomainValidationError(
            "live_tracking_end must not precede the historical boundary"
        )


@dataclass(frozen=True)
class Experiment:
    """Repository-facing experiment aggregate without ORM coupling."""

    experiment_id: UUID
    name: str
    mode: ExperimentMode
    status: ExperimentStatus
    base_currency: str
    initial_capital: float
    description: str | None = None
    benchmark_symbol: str | None = None
    training_start: date | None = None
    training_end: date | None = None
    optimization_as_of: date | None = None
    launch_date: date | None = None
    historical_evaluation_end: date | None = None
    live_tracking_end: date | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1"
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    archived_at: datetime | None = None

    def __post_init__(self) -> None:
        name = self.name.strip()
        currency = self.base_currency.strip().upper()
        if not name:
            raise DomainValidationError("experiment name is required")
        if not currency:
            raise DomainValidationError("base_currency is required")
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise DomainValidationError("initial_capital must be finite and positive")
        if not self.schema_version.strip():
            raise DomainValidationError("schema_version is required")
        created_at = ensure_utc(self.created_at, "created_at")
        updated_at = ensure_utc(self.updated_at, "updated_at")
        archived_at = (
            ensure_utc(self.archived_at, "archived_at")
            if self.archived_at is not None
            else None
        )
        if self.status is ExperimentStatus.ARCHIVED and archived_at is None:
            raise DomainValidationError("archived status requires archived_at")
        if self.status is not ExperimentStatus.ARCHIVED and archived_at is not None:
            raise DomainValidationError("archived_at is valid only for archived status")
        validate_date_boundaries(
            mode=self.mode,
            training_start=self.training_start,
            training_end=self.training_end,
            optimization_as_of=self.optimization_as_of,
            launch_date=self.launch_date,
            historical_evaluation_end=self.historical_evaluation_end,
            live_tracking_end=self.live_tracking_end,
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "base_currency", currency)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "archived_at", archived_at)
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))

    @classmethod
    def create(
        cls,
        *,
        name: str,
        mode: ExperimentMode,
        base_currency: str,
        initial_capital: float,
        **kwargs: Any,
    ) -> Experiment:
        """Create a draft experiment with a fresh immutable identity."""
        now = utc_now()
        return cls(
            experiment_id=uuid4(),
            name=name,
            mode=mode,
            status=ExperimentStatus.DRAFT,
            base_currency=base_currency,
            initial_capital=initial_capital,
            created_at=now,
            updated_at=now,
            **kwargs,
        )

    def transition(
        self, target: ExperimentStatus, *, at: datetime | None = None
    ) -> Experiment:
        """Return a new aggregate after a valid, timestamped transition."""
        validate_transition(self.status, target)
        timestamp = ensure_utc(at or utc_now(), "transition timestamp")
        return replace(
            self,
            status=target,
            updated_at=timestamp,
            archived_at=timestamp if target is ExperimentStatus.ARCHIVED else None,
        )


@dataclass(frozen=True)
class SnapshotAllocation:
    """One immutable target allocation in an optimization snapshot."""

    asset: str
    asset_type: str
    target_weight: float
    launch_price: float | None
    initial_value: float
    quantity: float
    is_cash: bool = False

    def __post_init__(self) -> None:
        if not self.asset.strip() or not self.asset_type.strip():
            raise DomainValidationError("allocation asset and asset_type are required")
        numeric = {
            "target_weight": self.target_weight,
            "initial_value": self.initial_value,
            "quantity": self.quantity,
        }
        if self.launch_price is not None:
            numeric["launch_price"] = self.launch_price
        for label, value in numeric.items():
            if not math.isfinite(value):
                raise DomainValidationError(f"{label} must be finite")
        if self.launch_price is not None and self.launch_price <= 0:
            raise DomainValidationError("launch_price must be positive when present")
        if self.is_cash and self.launch_price is not None:
            raise DomainValidationError("cash allocation must not have a launch price")


@dataclass(frozen=True)
class OptimizationSnapshot:
    """Immutable repository-facing optimization snapshot."""

    snapshot_id: UUID
    experiment_id: UUID
    package_version: str
    code_version: str
    objective: str
    solver: str
    solver_status: str
    source_data_hash: str
    assumption_recipe_hash: str
    assumptions: Mapping[str, Any]
    constraints: Mapping[str, Any]
    launch_forecast: Mapping[str, Any]
    scenario_metadata: Mapping[str, Any]
    return_policy: Mapping[str, Any]
    loss_convention: Mapping[str, Any]
    residual_validation: Mapping[str, Any]
    allocations: tuple[SnapshotAllocation, ...]
    created_at: datetime = field(default_factory=utc_now)
    activated_at: datetime | None = None

    def __post_init__(self) -> None:
        required = {
            "package_version": self.package_version,
            "code_version": self.code_version,
            "objective": self.objective,
            "solver": self.solver,
            "solver_status": self.solver_status,
        }
        for label, value in required.items():
            if not value.strip():
                raise DomainValidationError(f"{label} is required")
        for label, value in {
            "source_data_hash": self.source_data_hash,
            "assumption_recipe_hash": self.assumption_recipe_hash,
        }.items():
            if not _SHA256_PATTERN.fullmatch(value):
                raise DomainValidationError(f"{label} must be a lowercase SHA-256 hash")
        if not self.allocations:
            raise DomainValidationError("at least one snapshot allocation is required")
        assets = [allocation.asset for allocation in self.allocations]
        if len(assets) != len(set(assets)):
            raise DomainValidationError("snapshot allocation assets must be unique")
        created_at = ensure_utc(self.created_at, "created_at")
        activated_at = (
            ensure_utc(self.activated_at, "activated_at")
            if self.activated_at is not None
            else None
        )
        if activated_at is not None:
            if self.solver_status not in {"optimal", "optimal_inaccurate"}:
                raise DomainValidationError(
                    "activated snapshot requires a reviewed solved status"
                )
            if self.residual_validation.get("passed") is not True:
                raise DomainValidationError(
                    "activated snapshot requires passed residual validation"
                )
            weight_sum = math.fsum(item.target_weight for item in self.allocations)
            if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise DomainValidationError(
                    "activated snapshot target weights must sum to one"
                )
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "activated_at", activated_at)
        for name in (
            "assumptions",
            "constraints",
            "launch_forecast",
            "scenario_metadata",
            "return_policy",
            "loss_convention",
            "residual_validation",
        ):
            object.__setattr__(self, name, dict(getattr(self, name)))

    @classmethod
    def create(cls, *, experiment_id: UUID, **kwargs: Any) -> OptimizationSnapshot:
        """Create a snapshot with a fresh immutable identity."""
        return cls(snapshot_id=uuid4(), experiment_id=experiment_id, **kwargs)

    def activate(self, *, at: datetime | None = None) -> OptimizationSnapshot:
        """Validate and freeze a draft snapshot."""
        if self.activated_at is not None:
            raise ImmutableRecordError("snapshot is already activated")
        return replace(self, activated_at=at or utc_now())


@dataclass(frozen=True)
class PriceObservation:
    """One explicit, normalized price observation from a declared source."""

    symbol: str
    observation_date: date
    price: float
    quote_currency: str
    source: str
    retrieved_at: datetime
    data_status: PriceDataStatus = PriceDataStatus.COMPLETE

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        quote = self.quote_currency.strip().upper()
        source = self.source.strip()
        if not symbol or not quote or not source:
            raise DomainValidationError(
                "price symbol, quote_currency, and source are required"
            )
        if not math.isfinite(self.price) or self.price <= 0.0:
            raise DomainValidationError("price must be finite and positive")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "quote_currency", quote)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self, "retrieved_at", ensure_utc(self.retrieved_at, "retrieved_at")
        )


@dataclass(frozen=True)
class DailyAssetState:
    """One fixed-quantity asset valuation within a monitoring date."""

    experiment_id: UUID
    state_date: date
    asset: str
    quantity: float
    target_weight: float
    price: float | None = None
    market_value: float | None = None
    current_weight: float | None = None
    drift_percentage_points: float | None = None
    is_cash: bool = False

    def __post_init__(self) -> None:
        asset = self.asset.strip().upper()
        if not asset:
            raise DomainValidationError("daily asset name is required")
        for name in ("quantity", "target_weight"):
            if not math.isfinite(float(getattr(self, name))):
                raise DomainValidationError(f"{name} must be finite")
        for name in (
            "price",
            "market_value",
            "current_weight",
            "drift_percentage_points",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise DomainValidationError(f"{name} must be finite when present")
        if self.price is not None and self.price <= 0.0:
            raise DomainValidationError("asset-state price must be positive")
        if self.is_cash and self.price is not None:
            raise DomainValidationError("cash state must not have a market price")
        object.__setattr__(self, "asset", asset)


@dataclass(frozen=True)
class DailyPortfolioState:
    """One atomic portfolio valuation and its per-asset detail rows."""

    experiment_id: UUID
    state_date: date
    data_quality_status: DataQualityStatus
    calculation_version: str
    finalized: bool
    asset_states: tuple[DailyAssetState, ...]
    nav: float | None = None
    base_100_nav: float | None = None
    cash_value: float | None = None
    daily_return: float | None = None
    cumulative_return: float | None = None
    realized_volatility: float | None = None
    running_peak: float | None = None
    drawdown: float | None = None
    maximum_drawdown: float | None = None
    total_drift: float | None = None
    return_interval_days: int | None = None
    benchmark_nav: float | None = None
    benchmark_return: float | None = None
    quality_metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.calculation_version.strip():
            raise DomainValidationError("calculation_version is required")
        if not self.asset_states:
            raise DomainValidationError("at least one daily asset state is required")
        keys = [(item.asset, item.state_date) for item in self.asset_states]
        if len(keys) != len(set(keys)):
            raise DomainValidationError("daily asset states must be unique")
        if any(
            item.experiment_id != self.experiment_id
            or item.state_date != self.state_date
            for item in self.asset_states
        ):
            raise DomainValidationError(
                "daily asset states must share the portfolio experiment and date"
            )
        numeric_names = (
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
            "benchmark_nav",
            "benchmark_return",
        )
        for name in numeric_names:
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise DomainValidationError(f"{name} must be finite when present")
        if self.return_interval_days is not None and self.return_interval_days < 0:
            raise DomainValidationError("return_interval_days must be non-negative")
        if self.data_quality_status is DataQualityStatus.COMPLETE:
            required = (
                self.nav,
                self.base_100_nav,
                self.cash_value,
                self.daily_return,
                self.cumulative_return,
                self.running_peak,
                self.drawdown,
                self.maximum_drawdown,
                self.total_drift,
                self.return_interval_days,
            )
            if any(value is None for value in required):
                raise DomainValidationError(
                    "complete portfolio state is missing valuation fields"
                )
            if not self.finalized:
                raise DomainValidationError("complete portfolio state must be finalized")
            if self.nav is not None and self.nav <= 0.0:
                raise DomainValidationError("complete portfolio NAV must be positive")
            if self.drawdown is not None and self.drawdown > 1e-12:
                raise DomainValidationError("drawdown must not be positive")
            if self.maximum_drawdown is not None and self.maximum_drawdown > 1e-12:
                raise DomainValidationError("maximum_drawdown must not be positive")
            if self.total_drift is not None and self.total_drift < -1e-12:
                raise DomainValidationError("total_drift must not be negative")
            current_weights = [
                item.current_weight
                for item in self.asset_states
                if item.current_weight is not None
            ]
            if len(current_weights) != len(self.asset_states) or not math.isclose(
                math.fsum(current_weights), 1.0, rel_tol=0.0, abs_tol=1e-8
            ):
                raise DomainValidationError(
                    "complete daily asset current weights must sum to one"
                )
        elif self.finalized:
            raise DomainValidationError(
                "incomplete portfolio state must remain non-finalized"
            )
        object.__setattr__(self, "quality_metadata", dict(self.quality_metadata))
        object.__setattr__(self, "created_at", ensure_utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", ensure_utc(self.updated_at, "updated_at"))


@dataclass(frozen=True)
class ExperimentEvent:
    """Append-only audit event returned by repository queries."""

    event_id: UUID
    experiment_id: UUID
    event_type: str
    event_metadata: Mapping[str, Any] = field(default_factory=dict)
    effective_date: date | None = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        event_type = self.event_type.strip()
        if not event_type:
            raise DomainValidationError("event_type is required")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "event_metadata", dict(self.event_metadata))
        object.__setattr__(self, "created_at", ensure_utc(self.created_at, "created_at"))
