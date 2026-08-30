"""Atomic one-shot Live Forward and Hybrid monitoring updates.

The service is intentionally scheduler-agnostic.  It excludes the partial
current UTC day, appends explicit observations, values fixed quantities, and
evaluates matured VaR forecasts exactly once.  It never optimizes again and
never changes a finalized portfolio state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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
    MonitoringRun,
    MonitoringRunStatus,
    OptimizationSnapshot,
    RecordNotFoundError,
)
from .prices import NormalizedPriceData, normalize_monitoring_prices
from .providers import PriceProviderRegistry
from .recipes import OptimizationRecipe, optimization_recipe_from_dict
from .repository import PersistenceCounts
from .risk_forecasts import build_origin_safe_forecast, evaluate_matured_forecast
from .services import UnitOfWorkFactory
from .valuation import value_fixed_holdings


@dataclass(frozen=True)
class LiveUpdateResult:
    """One successful, committed update summary."""

    experiment_id: UUID
    run_id: UUID
    final_status: ExperimentStatus
    requested_cutoff: date
    actual_cutoff: date | None
    actual_source: str | None
    processed_dates: int
    price_counts: PersistenceCounts
    state_counts: PersistenceCounts
    forecast_counts: PersistenceCounts
    evaluated_forecasts: int
    incomplete_dates: int
    warning_count: int


@dataclass(frozen=True)
class AllActiveUpdateResult:
    """Independent results for a one-shot all-active invocation."""

    completed: tuple[LiveUpdateResult, ...]
    failed_experiment_ids: tuple[UUID, ...]


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


def _load_recipe(experiment: Experiment) -> OptimizationRecipe:
    raw = experiment.source_metadata.get("optimization_recipe")
    if not isinstance(raw, Mapping):
        raise DomainValidationError(
            "experiment lacks the persisted Batch 5 optimization recipe"
        )
    recipe = optimization_recipe_from_dict(raw)
    recorded = experiment.source_metadata.get("recipe_fingerprint")
    if recipe.fingerprint != recorded:
        raise DomainValidationError("persisted optimization recipe hash mismatch")
    if not recipe.source.refreshable:
        raise DomainValidationError("Live Forward and Hybrid require a refreshable source")
    return recipe


def _market_assets(snapshot: OptimizationSnapshot) -> tuple[str, ...]:
    assets = tuple(item.asset for item in snapshot.allocations if not item.is_cash)
    if not assets:
        raise DomainValidationError("monitoring snapshot has no market assets")
    return assets


def _state_price_frame(
    *,
    states: list[DailyPortfolioState],
    market_assets: tuple[str, ...],
    benchmark_symbol: str | None,
    initial_capital: float,
) -> pd.DataFrame:
    """Restore experiment-specific historical prices from immutable states."""
    columns = list(market_assets)
    if benchmark_symbol and benchmark_symbol not in columns:
        columns.append(benchmark_symbol)
    rows: dict[pd.Timestamp, dict[str, float | None]] = {}
    for state in states:
        values: dict[str, float | None] = {symbol: None for symbol in columns}
        for asset_state in state.asset_states:
            if not asset_state.is_cash and asset_state.asset in market_assets:
                values[asset_state.asset] = asset_state.price
        if (
            benchmark_symbol is not None
            and benchmark_symbol not in market_assets
            and state.benchmark_nav is not None
        ):
            values[benchmark_symbol] = state.benchmark_nav / initial_capital
        rows[pd.Timestamp(state.state_date)] = values
    if not rows:
        return pd.DataFrame(columns=columns, dtype=float)
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns).sort_index()


def _scale_provider_benchmark(
    provider_frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    benchmark_symbol: str | None,
    market_assets: tuple[str, ...],
) -> pd.DataFrame:
    """Put new benchmark prices on the persisted Base-1 benchmark scale."""
    frame = provider_frame.copy(deep=True)
    if (
        benchmark_symbol is None
        or benchmark_symbol in market_assets
        or benchmark_symbol not in frame
    ):
        return frame
    if state_frame.empty or benchmark_symbol not in state_frame:
        return frame
    candidates = state_frame[benchmark_symbol].dropna().index.intersection(frame.index)
    candidates = candidates[frame.loc[candidates, benchmark_symbol].notna()]
    if len(candidates) == 0:
        frame[benchmark_symbol] = float("nan")
        return frame
    anchor = candidates.max()
    persisted_ratio = float(state_frame.at[anchor, benchmark_symbol])
    provider_price = float(frame.at[anchor, benchmark_symbol])
    frame[benchmark_symbol] = frame[benchmark_symbol] * (
        persisted_ratio / provider_price
    )
    return frame


def _working_prices(
    *,
    provider: NormalizedPriceData,
    states: list[DailyPortfolioState],
    market_assets: tuple[str, ...],
    experiment: Experiment,
    cutoff: date,
) -> NormalizedPriceData:
    state_frame = _state_price_frame(
        states=states,
        market_assets=market_assets,
        benchmark_symbol=experiment.benchmark_symbol,
        initial_capital=experiment.initial_capital,
    )
    provider_frame = _scale_provider_benchmark(
        provider.prices.loc[: pd.Timestamp(cutoff)],
        state_frame,
        experiment.benchmark_symbol,
        market_assets,
    )
    combined = provider_frame.copy(deep=True)
    all_columns = list(market_assets)
    if experiment.benchmark_symbol and experiment.benchmark_symbol not in all_columns:
        all_columns.append(experiment.benchmark_symbol)
    combined = combined.reindex(columns=all_columns)
    if not state_frame.empty:
        combined = combined.reindex(combined.index.union(state_frame.index))
        for column in state_frame.columns:
            for timestamp, value in state_frame[column].items():
                combined.at[timestamp, column] = value
    start = combined.index.min()
    full_index = pd.date_range(start, pd.Timestamp(cutoff), freq="D")
    combined = combined.reindex(full_index)
    return normalize_monitoring_prices(
        combined,
        source=provider.source,
        quote_currency=provider.quote_currency,
        retrieved_at=provider.retrieved_at,
    )


def _slice_normalized(
    normalized: NormalizedPriceData, *, end: date
) -> NormalizedPriceData:
    frame = normalized.prices.loc[: pd.Timestamp(end)]
    return normalize_monitoring_prices(
        frame,
        source=normalized.source,
        quote_currency=normalized.quote_currency,
        retrieved_at=normalized.retrieved_at,
    )


def _candidate_start(
    experiment: Experiment, states: list[DailyPortfolioState]
) -> date:
    assert experiment.launch_date is not None
    if not states:
        return experiment.launch_date
    latest = states[-1]
    return latest.state_date if not latest.finalized else latest.state_date + timedelta(days=1)


def _complete_provider_cutoff(
    normalized: NormalizedPriceData,
    *,
    assets: tuple[str, ...],
    policy_cutoff: date,
    provider_complete_through: date,
) -> date | None:
    bounded = min(policy_cutoff, provider_complete_through)
    frame = normalized.prices.loc[: pd.Timestamp(bounded), list(assets)]
    if frame.empty:
        return None
    complete = frame.notna().all(axis=1)
    if not complete.any():
        return None
    return frame.index[complete].max().date()


def _scheduled_origin(
    *, experiment: Experiment, origin: date, recipe: OptimizationRecipe
) -> bool:
    assert experiment.launch_date is not None
    return recipe.risk.evaluation_mode == "overlapping" or (
        (origin - experiment.launch_date).days % recipe.risk.horizon_days == 0
    )


def _sanitized_provider_metadata(metadata: Mapping) -> dict:
    """Persist only reviewed non-secret provider capability flags."""
    return {
        "research_grade_vendor": bool(metadata.get("research_grade_vendor", False))
    }


class LiveMonitoringService:
    """Update one or all active live-capable experiments exactly once."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        provider_registry: PriceProviderRegistry,
    ) -> None:
        self._uow_factory = uow_factory
        self._providers = provider_registry

    def _load(
        self, experiment_id: UUID
    ) -> tuple[Experiment, OptimizationSnapshot, list[DailyPortfolioState]]:
        with self._uow_factory() as uow:
            experiment = uow.experiments.get(experiment_id)
            snapshot = uow.snapshots.get_for_experiment(experiment_id)
            states = uow.valuations.list(experiment_id)
        if experiment is None:
            raise RecordNotFoundError(f"experiment {experiment_id} does not exist")
        if snapshot is None or snapshot.activated_at is None:
            raise DomainValidationError("active monitoring requires an activated snapshot")
        return experiment, snapshot, states

    def _start_run(
        self,
        *,
        experiment: Experiment,
        requested_cutoff: date,
        started_at: datetime,
        effective_as_of: datetime,
    ) -> MonitoringRun:
        with self._uow_factory() as uow:
            prior_failures = [
                item
                for item in uow.runs.list(experiment.experiment_id)
                if item.status is MonitoringRunStatus.FAILED
            ]
            retry_of = str(prior_failures[-1].run_id) if prior_failures else None
            run = MonitoringRun.start(
                experiment_id=experiment.experiment_id,
                run_type="live_update",
                requested_cutoff=requested_cutoff,
                started_at=started_at,
                run_metadata={
                    "requested_provider": experiment.source_metadata.get("source", {}).get(
                        "provider"
                    ),
                    "retry_of": retry_of,
                    "partial_current_utc_day_excluded": True,
                    "effective_as_of_utc": effective_as_of.isoformat(),
                },
            )
            uow.runs.add(run)
            uow.events.add(
                ExperimentEvent(
                    event_id=uuid4(),
                    experiment_id=experiment.experiment_id,
                    event_type=(
                        "monitoring_retry_started"
                        if retry_of is not None
                        else "monitoring_run_started"
                    ),
                    effective_date=requested_cutoff,
                    event_metadata={"run_id": str(run.run_id), "retry_of": retry_of},
                    created_at=started_at,
                )
            )
            uow.commit()
        return run

    def _record_failure(self, run: MonitoringRun) -> None:
        failed = run.fail(
            error_code="live_update_failed",
            error_summary="monitoring update failed; no financial writes were committed",
            run_metadata={
                **run.run_metadata,
                "error_details_sanitized": True,
                "financial_transaction_rolled_back": True,
            },
        )
        with self._uow_factory() as uow:
            uow.runs.finish(failed)
            uow.events.add(
                ExperimentEvent(
                    event_id=uuid4(),
                    experiment_id=run.experiment_id,
                    event_type="monitoring_run_failed",
                    effective_date=run.requested_cutoff,
                    event_metadata={
                        "run_id": str(run.run_id),
                        "error_code": failed.error_code,
                        "error_details_sanitized": True,
                    },
                )
            )
            uow.commit()

    def update_experiment(
        self,
        experiment_id: UUID,
        *,
        requested_cutoff: date | None = None,
        as_of: datetime | None = None,
        code_version: str,
        calculation_version: str,
    ) -> LiveUpdateResult:
        """Fetch and atomically append all eligible complete observations."""
        invoked_at = datetime.now(timezone.utc)
        timestamp = as_of or invoked_at
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise DomainValidationError("live update as_of must be timezone-aware")
        timestamp = timestamp.astimezone(timezone.utc)
        if timestamp > invoked_at + timedelta(minutes=5):
            raise DomainValidationError("live update as_of cannot be in the future")
        last_complete_utc_day = timestamp.date() - timedelta(days=1)
        requested = requested_cutoff or last_complete_utc_day

        experiment, snapshot, existing_states = self._load(experiment_id)
        if experiment.mode not in {ExperimentMode.LIVE_FORWARD, ExperimentMode.HYBRID}:
            raise DomainValidationError("live update requires Live Forward or Hybrid mode")
        if experiment.status is not ExperimentStatus.ACTIVE:
            raise DomainValidationError("live update requires an active experiment")
        assert experiment.launch_date is not None
        recipe = _load_recipe(experiment)
        if snapshot.assumption_recipe_hash != recipe.fingerprint:
            raise DomainValidationError("snapshot and persisted monitoring recipe differ")
        policy_cutoff = min(requested, last_complete_utc_day)
        if experiment.live_tracking_end is not None:
            policy_cutoff = min(policy_cutoff, experiment.live_tracking_end)
        run = self._start_run(
            experiment=experiment,
            requested_cutoff=requested,
            started_at=invoked_at,
            effective_as_of=timestamp,
        )

        try:
            market_assets = _market_assets(snapshot)
            symbols = market_assets
            if experiment.benchmark_symbol and experiment.benchmark_symbol not in symbols:
                symbols = (*symbols, experiment.benchmark_symbol)
            start_date = _candidate_start(experiment, existing_states)
            if policy_cutoff < start_date:
                return self._complete_noop(
                    run=run,
                    experiment=experiment,
                    requested=requested,
                    actual_cutoff=(existing_states[-1].state_date if existing_states else None),
                    reason="requested cutoff has no new complete UTC dates",
                )
            lookback_start = start_date - timedelta(days=recipe.risk.estimation_window)
            batch, fallback_used = self._providers.fetch(
                source=recipe.source,
                symbols=symbols,
                start_date=lookback_start,
                end_date=policy_cutoff,
                requested_at=invoked_at,
            )
            if batch.quote_currency != recipe.source.quote_currency:
                raise DomainValidationError("provider quote currency differs from recipe")
            normalized = normalize_monitoring_prices(
                batch.prices,
                source=batch.actual_source,
                quote_currency=batch.quote_currency,
                retrieved_at=batch.retrieved_at,
            )
            actual_cutoff = _complete_provider_cutoff(
                normalized,
                assets=market_assets,
                policy_cutoff=policy_cutoff,
                provider_complete_through=batch.complete_through,
            )
            if actual_cutoff is None or actual_cutoff < start_date:
                return self._complete_noop(
                    run=run,
                    experiment=experiment,
                    requested=requested,
                    actual_cutoff=(existing_states[-1].state_date if existing_states else None),
                    reason="provider returned no new complete frozen-universe date",
                    actual_source=batch.actual_source,
                    fallback_used=fallback_used,
                )

            working = _working_prices(
                provider=normalized,
                states=existing_states,
                market_assets=market_assets,
                experiment=experiment,
                cutoff=actual_cutoff,
            )
            candidate_dates = list(pd.date_range(start_date, actual_cutoff, freq="D").date)
            states_to_write: list[DailyPortfolioState] = []
            forecasts_to_write = []
            for current_date in candidate_dates:
                revealed = _slice_normalized(working, end=current_date)
                state = value_fixed_holdings(
                    experiment=experiment,
                    snapshot=snapshot,
                    normalized=revealed,
                    cash_policy=recipe.cash,
                    calculation_version=calculation_version,
                )[-1]
                if state.state_date != current_date:
                    raise DomainValidationError("live valuation did not end at update date")
                states_to_write.append(state)
                if state.finalized and _scheduled_origin(
                    experiment=experiment, origin=current_date, recipe=recipe
                ):
                    forecasts_to_write.append(
                        build_origin_safe_forecast(
                            normalized=revealed,
                            state=state,
                            recipe=recipe.risk,
                            cash_policy=recipe.cash,
                            model_version=code_version,
                            created_at=invoked_at,
                        )
                    )

            provider_new = normalized.prices.loc[
                (normalized.prices.index >= pd.Timestamp(start_date))
                & (normalized.prices.index <= pd.Timestamp(actual_cutoff))
            ]
            new_observations = normalize_monitoring_prices(
                provider_new,
                source=normalized.source,
                quote_currency=normalized.quote_currency,
                retrieved_at=normalized.retrieved_at,
            ).observations()
            return self._commit_update(
                run=run,
                experiment=experiment,
                recipe=recipe,
                requested=requested,
                actual_cutoff=actual_cutoff,
                actual_source=batch.actual_source,
                fallback_used=fallback_used,
                provider_metadata=batch.metadata,
                observations=new_observations,
                states=states_to_write,
                forecasts=forecasts_to_write,
                processed_dates=len(candidate_dates),
            )
        except Exception:
            self._record_failure(run)
            raise

    def _complete_noop(
        self,
        *,
        run: MonitoringRun,
        experiment: Experiment,
        requested: date,
        actual_cutoff: date | None,
        reason: str,
        actual_source: str | None = None,
        fallback_used: bool = False,
    ) -> LiveUpdateResult:
        metadata = {
            **run.run_metadata,
            "actual_source": actual_source,
            "fallback_used": fallback_used,
            "no_op": True,
            "reason": reason,
        }
        completed = run.complete(
            actual_cutoff=actual_cutoff,
            inserted_count=0,
            updated_count=0,
            skipped_count=0,
            warning_count=1,
            run_metadata=metadata,
        )
        with self._uow_factory() as uow:
            uow.runs.finish(completed)
            uow.events.add(
                ExperimentEvent(
                    event_id=uuid4(),
                    experiment_id=experiment.experiment_id,
                    event_type="monitoring_run_completed",
                    effective_date=actual_cutoff,
                    event_metadata={
                        "run_id": str(run.run_id),
                        "no_op": True,
                        "reason": reason,
                    },
                )
            )
            uow.commit()
        return LiveUpdateResult(
            experiment_id=experiment.experiment_id,
            run_id=run.run_id,
            final_status=experiment.status,
            requested_cutoff=requested,
            actual_cutoff=actual_cutoff,
            actual_source=actual_source,
            processed_dates=0,
            price_counts=PersistenceCounts(),
            state_counts=PersistenceCounts(),
            forecast_counts=PersistenceCounts(),
            evaluated_forecasts=0,
            incomplete_dates=0,
            warning_count=1,
        )

    def _commit_update(
        self,
        *,
        run: MonitoringRun,
        experiment: Experiment,
        recipe: OptimizationRecipe,
        requested: date,
        actual_cutoff: date,
        actual_source: str,
        fallback_used: bool,
        provider_metadata: Mapping,
        observations,
        states: list[DailyPortfolioState],
        forecasts: list,
        processed_dates: int,
    ) -> LiveUpdateResult:
        state_counts = PersistenceCounts()
        forecast_counts = PersistenceCounts()
        evaluated_count = 0
        incomplete_count = sum(not state.finalized for state in states)
        warning_count = int(fallback_used) + incomplete_count
        with self._uow_factory() as uow:
            price_counts = uow.prices.add_many(observations)
            for state in states:
                state_counts = _add_counts(
                    state_counts, _outcome_count(uow.valuations.write(state))
                )
            for forecast in forecasts:
                forecast_counts = _add_counts(
                    forecast_counts, _outcome_count(uow.forecasts.write(forecast))
                )
            pending = uow.forecasts.list(
                experiment.experiment_id,
                evaluation_status=ForecastEvaluationStatus.PENDING,
            )
            for forecast in pending:
                if forecast.target_date > actual_cutoff:
                    continue
                origin_state = uow.valuations.get(
                    experiment.experiment_id, forecast.origin_date
                )
                target_state = uow.valuations.get(
                    experiment.experiment_id, forecast.target_date
                )
                if (
                    origin_state is None
                    or target_state is None
                    or not origin_state.finalized
                    or not target_state.finalized
                ):
                    continue
                evaluated = evaluate_matured_forecast(
                    forecast,
                    origin_state=origin_state,
                    target_state=target_state,
                )
                outcome = uow.forecasts.write(evaluated)
                forecast_counts = _add_counts(
                    forecast_counts, _outcome_count(outcome)
                )
                if outcome == "evaluated":
                    evaluated_count += 1

            final_status = experiment.status
            if (
                experiment.live_tracking_end is not None
                and actual_cutoff >= experiment.live_tracking_end
            ):
                end_state = uow.valuations.get(
                    experiment.experiment_id, experiment.live_tracking_end
                )
                if end_state is not None and end_state.finalized:
                    final_status = uow.experiments.transition(
                        experiment.experiment_id, ExperimentStatus.COMPLETED
                    ).status
            inserted = (
                price_counts.inserted
                + state_counts.inserted
                + forecast_counts.inserted
            )
            updated = state_counts.updated + forecast_counts.updated
            skipped = price_counts.skipped + state_counts.skipped + forecast_counts.skipped
            metadata = {
                **run.run_metadata,
                "actual_source": actual_source,
                "fallback_used": fallback_used,
                "provider_metadata": _sanitized_provider_metadata(provider_metadata),
                "financial_transaction_rolled_back": False,
                "counts": {
                    "prices": price_counts.__dict__,
                    "states": state_counts.__dict__,
                    "forecasts": forecast_counts.__dict__,
                    "evaluated_forecasts": evaluated_count,
                },
            }
            completed = run.complete(
                actual_cutoff=actual_cutoff,
                inserted_count=inserted,
                updated_count=updated,
                skipped_count=skipped,
                warning_count=warning_count,
                run_metadata=metadata,
            )
            uow.runs.finish(completed)
            uow.events.add(
                ExperimentEvent(
                    event_id=uuid4(),
                    experiment_id=experiment.experiment_id,
                    event_type="monitoring_run_completed",
                    effective_date=actual_cutoff,
                    event_metadata={
                        "run_id": str(run.run_id),
                        "actual_source": actual_source,
                        "actual_cutoff": actual_cutoff.isoformat(),
                        "fallback_used": fallback_used,
                        "inserted_count": inserted,
                        "updated_count": updated,
                        "skipped_count": skipped,
                    },
                )
            )
            uow.commit()

        return LiveUpdateResult(
            experiment_id=experiment.experiment_id,
            run_id=run.run_id,
            final_status=final_status,
            requested_cutoff=requested,
            actual_cutoff=actual_cutoff,
            actual_source=actual_source,
            processed_dates=processed_dates,
            price_counts=price_counts,
            state_counts=state_counts,
            forecast_counts=forecast_counts,
            evaluated_forecasts=evaluated_count,
            incomplete_dates=incomplete_count,
            warning_count=warning_count,
        )

    def update_all_active(
        self,
        *,
        requested_cutoff: date | None = None,
        as_of: datetime | None = None,
        code_version: str,
        calculation_version: str,
    ) -> AllActiveUpdateResult:
        """Attempt each active live-capable experiment independently."""
        with self._uow_factory() as uow:
            experiments = [
                item
                for item in uow.experiments.list()
                if item.status is ExperimentStatus.ACTIVE
                and item.mode in {ExperimentMode.LIVE_FORWARD, ExperimentMode.HYBRID}
            ]
        completed: list[LiveUpdateResult] = []
        failed: list[UUID] = []
        for experiment in experiments:
            try:
                completed.append(
                    self.update_experiment(
                        experiment.experiment_id,
                        requested_cutoff=requested_cutoff,
                        as_of=as_of,
                        code_version=code_version,
                        calculation_version=calculation_version,
                    )
                )
            except Exception:
                failed.append(experiment.experiment_id)
        return AllActiveUpdateResult(
            completed=tuple(completed), failed_experiment_ids=tuple(failed)
        )


__all__ = ["AllActiveUpdateResult", "LiveMonitoringService", "LiveUpdateResult"]
