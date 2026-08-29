"""Application services for the experiment registry and atomic persistence."""

from __future__ import annotations

from datetime import date, datetime
from typing import Callable
from uuid import UUID

from .domain import (
    DailyPortfolioState,
    DomainValidationError,
    Experiment,
    ExperimentMode,
    ExperimentStatus,
    OptimizationSnapshot,
    PriceObservation,
    RecordNotFoundError,
    validate_date_boundaries,
)
from .recipes import OptimizationRecipe
from .repository import PersistenceCounts, UnitOfWork
from .hashing import sha256_fingerprint


UnitOfWorkFactory = Callable[[], UnitOfWork]


def _snapshot_signature(snapshot: OptimizationSnapshot) -> str:
    return sha256_fingerprint(
        {
            "snapshot_id": snapshot.snapshot_id,
            "experiment_id": snapshot.experiment_id,
            "package_version": snapshot.package_version,
            "code_version": snapshot.code_version,
            "objective": snapshot.objective,
            "solver": snapshot.solver,
            "solver_status": snapshot.solver_status,
            "source_data_hash": snapshot.source_data_hash,
            "assumption_recipe_hash": snapshot.assumption_recipe_hash,
            "assumptions": snapshot.assumptions,
            "constraints": snapshot.constraints,
            "launch_forecast": snapshot.launch_forecast,
            "scenario_metadata": snapshot.scenario_metadata,
            "return_policy": snapshot.return_policy,
            "loss_convention": snapshot.loss_convention,
            "residual_validation": snapshot.residual_validation,
            "allocations": [
                {
                    "asset": item.asset,
                    "asset_type": item.asset_type,
                    "target_weight": item.target_weight,
                    "launch_price": item.launch_price,
                    "initial_value": item.initial_value,
                    "quantity": item.quantity,
                    "is_cash": item.is_cash,
                }
                for item in sorted(snapshot.allocations, key=lambda item: item.asset)
            ],
            "created_at": snapshot.created_at,
            "activated_at": snapshot.activated_at,
        }
    )


class ExperimentRegistry:
    """Create, query, transition, archive, and snapshot model experiments."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    def create(
        self,
        *,
        name: str,
        mode: ExperimentMode,
        base_currency: str,
        initial_capital: float,
        recipe: OptimizationRecipe,
        training_start: date,
        training_end: date,
        optimization_as_of: date,
        launch_date: date,
        historical_evaluation_end: date | None = None,
        live_tracking_end: date | None = None,
        description: str | None = None,
        benchmark_symbol: str | None = None,
    ) -> Experiment:
        validate_date_boundaries(
            mode=mode,
            training_start=training_start,
            training_end=training_end,
            optimization_as_of=optimization_as_of,
            launch_date=launch_date,
            historical_evaluation_end=historical_evaluation_end,
            live_tracking_end=live_tracking_end,
            require_complete=True,
        )
        if mode in {ExperimentMode.LIVE_FORWARD, ExperimentMode.HYBRID} and not (
            recipe.source.refreshable
        ):
            raise DomainValidationError(
                "Live Forward and Hybrid experiments require a refreshable source"
            )
        experiment = Experiment.create(
            name=name,
            mode=mode,
            base_currency=base_currency,
            initial_capital=initial_capital,
            description=description,
            benchmark_symbol=(
                benchmark_symbol.strip().upper() if benchmark_symbol else None
            ),
            training_start=training_start,
            training_end=training_end,
            optimization_as_of=optimization_as_of,
            launch_date=launch_date,
            historical_evaluation_end=historical_evaluation_end,
            live_tracking_end=live_tracking_end,
            source_metadata={
                "source": recipe.source.to_dict(),
                "recipe_fingerprint": recipe.fingerprint,
                "optimization_recipe": recipe.to_dict(),
            },
        )
        with self._uow_factory() as uow:
            uow.experiments.add(experiment)
            uow.commit()
        return experiment

    def get(self, experiment_id: UUID) -> Experiment:
        with self._uow_factory() as uow:
            experiment = uow.experiments.get(experiment_id)
        if experiment is None:
            raise RecordNotFoundError(f"experiment {experiment_id} does not exist")
        return experiment

    def list(self, *, include_archived: bool = False) -> list[Experiment]:
        with self._uow_factory() as uow:
            return uow.experiments.list(include_archived=include_archived)

    def transition(
        self,
        experiment_id: UUID,
        target: ExperimentStatus,
        *,
        at: datetime | None = None,
    ) -> Experiment:
        with self._uow_factory() as uow:
            experiment = uow.experiments.transition(experiment_id, target, at=at)
            uow.commit()
            return experiment

    def archive(
        self, experiment_id: UUID, *, at: datetime | None = None
    ) -> Experiment:
        with self._uow_factory() as uow:
            experiment = uow.experiments.archive(experiment_id, at=at)
            uow.commit()
            return experiment

    def save_snapshot(self, snapshot: OptimizationSnapshot) -> str:
        """Persist one already-validated immutable snapshot idempotently."""
        if snapshot.activated_at is None:
            raise DomainValidationError(
                "registry accepts only validated activated optimization snapshots"
            )
        with self._uow_factory() as uow:
            existing = uow.snapshots.get_for_experiment(snapshot.experiment_id)
            if existing is not None:
                if _snapshot_signature(existing) == _snapshot_signature(snapshot):
                    return "skipped"
                raise DomainValidationError(
                    "experiment already has a different optimization snapshot"
                )
            uow.snapshots.add(snapshot)
            uow.commit()
            return "inserted"


class MonitoringPersistenceService:
    """Persist explicit prices and daily valuations in one transaction."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    def persist(
        self,
        *,
        observations: tuple[PriceObservation, ...],
        states: tuple[DailyPortfolioState, ...],
    ) -> dict[str, PersistenceCounts]:
        with self._uow_factory() as uow:
            price_counts = uow.prices.add_many(observations)
            state_inserted = 0
            state_updated = 0
            state_skipped = 0
            for state in states:
                outcome = uow.valuations.write(state)
                if outcome == "inserted":
                    state_inserted += 1
                elif outcome == "updated":
                    state_updated += 1
                elif outcome == "skipped":
                    state_skipped += 1
                else:  # pragma: no cover - adapter contract guard
                    raise RuntimeError(f"unknown valuation persistence outcome {outcome}")
            uow.commit()
        return {
            "prices": price_counts,
            "states": PersistenceCounts(
                inserted=state_inserted,
                updated=state_updated,
                skipped=state_skipped,
            ),
        }


__all__ = [
    "ExperimentRegistry",
    "MonitoringPersistenceService",
    "UnitOfWorkFactory",
]
