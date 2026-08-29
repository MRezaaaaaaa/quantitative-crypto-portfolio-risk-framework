"""Sequential Historical Out-of-Sample replay orchestration.

Historical replay is retrospective evidence, not a live forward test.  The
service deliberately reveals post-cutoff prices one date at a time and calls
the same pure valuation and risk functions intended for later live updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Mapping
from uuid import UUID, uuid4

import pandas as pd

from .domain import (
    DailyPortfolioState,
    DomainValidationError,
    Experiment,
    ExperimentEvent,
    ExperimentMode,
    ExperimentStatus,
    ForecastEvaluationStatus,
    ImmutableRecordError,
    OptimizationSnapshot,
    RecordNotFoundError,
    validate_date_boundaries,
)
from .optimization_adapter import build_point_in_time_snapshot
from .prices import (
    NormalizedPriceData,
    fingerprint_price_slice,
    missing_symbols_on_date,
    normalize_monitoring_prices,
)
from .recipes import OptimizationRecipe
from .repository import PersistenceCounts
from .risk_forecasts import (
    build_origin_safe_forecast,
    evaluate_matured_forecast,
)
from .services import UnitOfWorkFactory
from .valuation import value_fixed_holdings


@dataclass(frozen=True)
class HistoricalReplayResult:
    """Deterministic persistence summary for one replay invocation."""

    experiment_id: UUID
    final_status: ExperimentStatus
    snapshot_id: UUID
    processed_dates: int
    price_counts: PersistenceCounts
    state_counts: PersistenceCounts
    forecast_counts: PersistenceCounts
    evaluated_forecasts: int
    incomplete_dates: int
    historical_boundary: date


def _normalized_slice(
    normalized: NormalizedPriceData,
    *,
    start: date | None = None,
    end: date,
) -> NormalizedPriceData:
    frame = normalized.prices.loc[: pd.Timestamp(end)]
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if frame.empty:
        raise DomainValidationError("requested replay price slice is empty")
    return normalize_monitoring_prices(
        frame,
        source=normalized.source,
        quote_currency=normalized.quote_currency,
        retrieved_at=normalized.retrieved_at,
    )


def _add_counts(left: PersistenceCounts, right: PersistenceCounts) -> PersistenceCounts:
    return PersistenceCounts(
        inserted=left.inserted + right.inserted,
        updated=left.updated + right.updated,
        skipped=left.skipped + right.skipped,
    )


def _outcome_count(outcome: str) -> PersistenceCounts:
    if outcome == "inserted":
        return PersistenceCounts(inserted=1)
    if outcome in {"updated", "evaluated"}:
        return PersistenceCounts(updated=1)
    if outcome == "skipped":
        return PersistenceCounts(skipped=1)
    raise RuntimeError(f"unsupported persistence outcome {outcome!r}")


def _verify_existing_snapshot(
    *,
    snapshot: OptimizationSnapshot,
    experiment: Experiment,
    normalized: NormalizedPriceData,
    universe: tuple[str, ...],
    recipe: OptimizationRecipe,
) -> None:
    if snapshot.activated_at is None:
        raise ImmutableRecordError("historical replay snapshot is not activated")
    if snapshot.assumption_recipe_hash != recipe.fingerprint:
        raise ImmutableRecordError("persisted snapshot uses a different recipe")
    assert experiment.training_start is not None
    assert experiment.training_end is not None
    training = normalized.prices.loc[
        (normalized.prices.index >= pd.Timestamp(experiment.training_start))
        & (normalized.prices.index <= pd.Timestamp(experiment.training_end)),
        list(universe),
    ]
    source_hash = fingerprint_price_slice(
        training,
        source=normalized.source,
        quote_currency=normalized.quote_currency,
    )
    if source_hash != snapshot.source_data_hash:
        raise ImmutableRecordError(
            "persisted snapshot training data differs from replay input"
        )
    snapshot_assets = tuple(
        sorted(item.asset for item in snapshot.allocations if not item.is_cash)
    )
    if snapshot_assets != tuple(sorted(universe)):
        raise ImmutableRecordError("persisted snapshot universe differs from replay")


def _snapshot_replay_content(snapshot: OptimizationSnapshot) -> tuple:
    """Compare cutoff-rebuilt content without random identity or run timestamps."""
    return (
        snapshot.package_version,
        snapshot.code_version,
        snapshot.objective,
        snapshot.solver,
        snapshot.solver_status,
        snapshot.source_data_hash,
        snapshot.assumption_recipe_hash,
        snapshot.assumptions,
        snapshot.constraints,
        snapshot.launch_forecast,
        snapshot.scenario_metadata,
        snapshot.return_policy,
        snapshot.loss_convention,
        snapshot.residual_validation,
        snapshot.allocations,
    )


class HistoricalReplayService:
    """Build a cutoff-safe snapshot and replay a finite historical OOS interval."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    def _load_experiment(self, experiment_id: UUID) -> Experiment:
        with self._uow_factory() as uow:
            experiment = uow.experiments.get(experiment_id)
        if experiment is None:
            raise RecordNotFoundError(f"experiment {experiment_id} does not exist")
        return experiment

    @staticmethod
    def _validate_inputs(
        *,
        experiment: Experiment,
        normalized: NormalizedPriceData,
        universe: tuple[str, ...],
        recipe: OptimizationRecipe,
    ) -> date:
        if experiment.mode not in {
            ExperimentMode.HISTORICAL_OOS,
            ExperimentMode.HYBRID,
        }:
            raise DomainValidationError(
                "historical replay supports only Historical OOS and Hybrid modes"
            )
        validate_date_boundaries(
            mode=experiment.mode,
            training_start=experiment.training_start,
            training_end=experiment.training_end,
            optimization_as_of=experiment.optimization_as_of,
            launch_date=experiment.launch_date,
            historical_evaluation_end=experiment.historical_evaluation_end,
            live_tracking_end=experiment.live_tracking_end,
            require_complete=True,
        )
        assert experiment.historical_evaluation_end is not None
        assert experiment.launch_date is not None
        if normalized.source != recipe.source.provider:
            raise DomainValidationError("replay actual source differs from frozen recipe")
        if normalized.quote_currency != recipe.source.quote_currency:
            raise DomainValidationError(
                "replay quote currency differs from frozen recipe"
            )
        if recipe.scenario.source != "historical" and recipe.scenario.random_seed is None:
            raise DomainValidationError(
                "historical replay requires a deterministic Monte Carlo random_seed"
            )
        recorded_fingerprint = experiment.source_metadata.get("recipe_fingerprint")
        if recorded_fingerprint != recipe.fingerprint:
            raise DomainValidationError(
                "replay recipe differs from the experiment creation recipe"
            )
        required = (*universe,)
        if experiment.benchmark_symbol:
            required = (*required, experiment.benchmark_symbol)
        absent = [symbol for symbol in required if symbol not in normalized.prices]
        if absent:
            raise DomainValidationError(
                "replay source is missing required symbols: " + ", ".join(absent)
            )
        boundary = experiment.historical_evaluation_end
        if pd.Timestamp(boundary) not in normalized.prices.index:
            raise DomainValidationError(
                "historical evaluation end must be an explicit source observation"
            )
        missing_end = missing_symbols_on_date(normalized, boundary, universe)
        if missing_end:
            raise DomainValidationError(
                "historical evaluation end is incomplete; missing: "
                + ", ".join(missing_end)
            )
        if normalized.prices.index.min().date() > experiment.training_start:
            raise DomainValidationError("replay source begins after training_start")
        return boundary

    def run(
        self,
        *,
        experiment_id: UUID,
        normalized: NormalizedPriceData,
        universe: tuple[str, ...] | list[str],
        recipe: OptimizationRecipe,
        package_version: str,
        code_version: str,
        calculation_version: str,
        asset_types: Mapping[str, str] | None = None,
    ) -> HistoricalReplayResult:
        """Run or idempotently replay one finite historical interval."""
        experiment = self._load_experiment(experiment_id)
        assets = tuple(dict.fromkeys(str(item).strip().upper() for item in universe))
        if not assets or any(not item for item in assets):
            raise DomainValidationError("historical replay requires a frozen universe")
        boundary = self._validate_inputs(
            experiment=experiment,
            normalized=normalized,
            universe=assets,
            recipe=recipe,
        )
        assert experiment.optimization_as_of is not None
        assert experiment.launch_date is not None
        assert experiment.training_start is not None

        bounded = _normalized_slice(
            normalized, start=experiment.training_start, end=boundary
        )
        launch_visible = _normalized_slice(
            bounded,
            end=experiment.launch_date,
        )
        with self._uow_factory() as uow:
            snapshot = uow.snapshots.get_for_experiment(experiment_id)
        rebuilt_snapshot = build_point_in_time_snapshot(
            experiment=experiment,
            normalized=launch_visible,
            universe=assets,
            recipe=recipe,
            package_version=package_version,
            code_version=code_version,
            asset_types=asset_types,
        )
        if snapshot is not None:
            _verify_existing_snapshot(
                snapshot=snapshot,
                experiment=experiment,
                normalized=launch_visible,
                universe=assets,
                recipe=recipe,
            )
            if _snapshot_replay_content(snapshot) != _snapshot_replay_content(
                rebuilt_snapshot
            ):
                raise ImmutableRecordError(
                    "persisted snapshot does not match cutoff-rebuilt optimization"
                )
        else:
            snapshot = rebuilt_snapshot

        started_backfill = experiment.status in {
            ExperimentStatus.DRAFT,
            ExperimentStatus.FAILED,
        }
        if experiment.status not in {
            ExperimentStatus.DRAFT,
            ExperimentStatus.FAILED,
            ExperimentStatus.BACKFILLING,
            ExperimentStatus.COMPLETED,
            ExperimentStatus.ACTIVE,
        }:
            raise DomainValidationError(
                f"experiment status {experiment.status.value!r} cannot replay"
            )

        price_counts = PersistenceCounts()
        state_counts = PersistenceCounts()
        forecast_counts = PersistenceCounts()
        evaluated_count = 0
        incomplete_count = 0
        processed_dates = 0

        try:
            with self._uow_factory() as uow:
                if started_backfill:
                    experiment = uow.experiments.transition(
                        experiment_id, ExperimentStatus.BACKFILLING
                    )
                if uow.snapshots.get_for_experiment(experiment_id) is None:
                    uow.snapshots.add(snapshot)
                training_visible = _normalized_slice(
                    bounded, start=experiment.training_start, end=experiment.optimization_as_of
                )
                price_counts = _add_counts(
                    price_counts, uow.prices.add_many(training_visible.observations())
                )
                uow.commit()

            replay_rows = bounded.prices.loc[
                (bounded.prices.index > pd.Timestamp(experiment.optimization_as_of))
                & (bounded.prices.index <= pd.Timestamp(boundary))
            ]
            if replay_rows.empty:
                raise DomainValidationError("historical replay interval contains no rows")

            for timestamp in replay_rows.index:
                current_date = timestamp.date()
                revealed = _normalized_slice(bounded, end=current_date)
                one_date = _normalized_slice(
                    bounded, start=current_date, end=current_date
                )
                state: DailyPortfolioState | None = None
                new_forecast = None
                if current_date >= experiment.launch_date:
                    states = value_fixed_holdings(
                        experiment=experiment,
                        snapshot=snapshot,
                        normalized=revealed,
                        cash_policy=recipe.cash,
                        calculation_version=calculation_version,
                    )
                    state = states[-1]
                    if state.state_date != current_date:
                        raise DomainValidationError(
                            "sequential valuation did not end at the revealed date"
                        )
                    if not state.finalized:
                        incomplete_count += 1
                    target_date = current_date + timedelta(
                        days=recipe.risk.horizon_days
                    )
                    scheduled = recipe.risk.evaluation_mode == "overlapping" or (
                        (current_date - experiment.launch_date).days
                        % recipe.risk.horizon_days
                        == 0
                    )
                    if (
                        state.finalized
                        and scheduled
                        and target_date <= boundary
                    ):
                        new_forecast = build_origin_safe_forecast(
                            normalized=revealed,
                            state=state,
                            recipe=recipe.risk,
                            cash_policy=recipe.cash,
                            model_version=code_version,
                        )

                with self._uow_factory() as uow:
                    price_counts = _add_counts(
                        price_counts, uow.prices.add_many(one_date.observations())
                    )
                    if state is not None:
                        outcome = uow.valuations.write(state)
                        state_counts = _add_counts(state_counts, _outcome_count(outcome))
                    pending = uow.forecasts.list(
                        experiment_id,
                        evaluation_status=ForecastEvaluationStatus.PENDING,
                    )
                    for forecast in pending:
                        if forecast.target_date != current_date or state is None:
                            continue
                        if not state.finalized:
                            continue
                        origin_state = uow.valuations.get(
                            experiment_id, forecast.origin_date
                        )
                        if origin_state is None:
                            raise DomainValidationError(
                                "persisted forecast origin state is missing"
                            )
                        evaluated = evaluate_matured_forecast(
                            forecast,
                            origin_state=origin_state,
                            target_state=state,
                        )
                        outcome = uow.forecasts.write(evaluated)
                        forecast_counts = _add_counts(
                            forecast_counts, _outcome_count(outcome)
                        )
                        if outcome == "evaluated":
                            evaluated_count += 1
                    if new_forecast is not None:
                        outcome = uow.forecasts.write(new_forecast)
                        forecast_counts = _add_counts(
                            forecast_counts, _outcome_count(outcome)
                        )
                    uow.commit()
                processed_dates += 1

            with self._uow_factory() as uow:
                current = uow.experiments.get(experiment_id)
                assert current is not None
                target_status = (
                    ExperimentStatus.COMPLETED
                    if current.mode is ExperimentMode.HISTORICAL_OOS
                    else ExperimentStatus.ACTIVE
                )
                if current.status is ExperimentStatus.BACKFILLING:
                    current = uow.experiments.transition(experiment_id, target_status)
                    uow.events.add(
                        ExperimentEvent(
                            event_id=uuid4(),
                            experiment_id=experiment_id,
                            effective_date=boundary,
                            event_type="historical_replay_boundary_reached",
                            event_metadata={
                                "historical_boundary": boundary.isoformat(),
                                "next_status": target_status.value,
                                "historical_oos_not_live_forward": True,
                            },
                        )
                    )
                elif current.status is not target_status:
                    raise DomainValidationError(
                        "replay finished in an incompatible experiment status"
                    )
                uow.commit()
                final_status = current.status
        except Exception:
            with self._uow_factory() as uow:
                current = uow.experiments.get(experiment_id)
                if current is not None and current.status is ExperimentStatus.BACKFILLING:
                    uow.experiments.transition(experiment_id, ExperimentStatus.FAILED)
                    uow.events.add(
                        ExperimentEvent(
                            event_id=uuid4(),
                            experiment_id=experiment_id,
                            effective_date=None,
                            event_type="historical_replay_failed",
                            event_metadata={
                                "error_code": "historical_replay_failed",
                                "error_details_sanitized": True,
                            },
                        )
                    )
                    uow.commit()
            raise

        return HistoricalReplayResult(
            experiment_id=experiment_id,
            final_status=final_status,
            snapshot_id=snapshot.snapshot_id,
            processed_dates=processed_dates,
            price_counts=price_counts,
            state_counts=state_counts,
            forecast_counts=forecast_counts,
            evaluated_forecasts=evaluated_count,
            incomplete_dates=incomplete_count,
            historical_boundary=boundary,
        )


__all__ = ["HistoricalReplayResult", "HistoricalReplayService"]
