"""Explicit experiment-creation workflows used by the monitoring UI.

The workflow composes reviewed domain services.  Historical and Hybrid modes
always rebuild through the Historical OOS service.  Live Forward freezes one
point-in-time snapshot and launch state; it does not reuse Streamlit session
optimizer output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping
from uuid import uuid4

import pandas as pd

from .domain import (
    Experiment,
    ExperimentEvent,
    ExperimentMode,
    ExperimentStatus,
    validate_date_boundaries,
)
from .historical_replay import HistoricalReplayResult, HistoricalReplayService
from .optimization_adapter import build_point_in_time_snapshot
from .prices import NormalizedPriceData, normalize_monitoring_prices
from .recipes import OptimizationRecipe
from .risk_forecasts import build_origin_safe_forecast
from .services import ExperimentRegistry, UnitOfWorkFactory
from .valuation import value_fixed_holdings


@dataclass(frozen=True)
class ExperimentInitializationResult:
    """Persisted identity and initialization outcome."""

    experiment: Experiment
    historical_replay: HistoricalReplayResult | None


def _bounded_prices(
    normalized: NormalizedPriceData, *, start: date, end: date
) -> NormalizedPriceData:
    frame = normalized.prices.loc[
        (normalized.prices.index >= pd.Timestamp(start))
        & (normalized.prices.index <= pd.Timestamp(end))
    ]
    return normalize_monitoring_prices(
        frame,
        source=normalized.source,
        quote_currency=normalized.quote_currency,
        retrieved_at=normalized.retrieved_at,
    )


class ExperimentCreationWorkflow:
    """Create and initialize one point-in-time experiment from explicit prices."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory
        self._registry = ExperimentRegistry(uow_factory)

    def create(
        self,
        *,
        name: str,
        mode: ExperimentMode,
        base_currency: str,
        initial_capital: float,
        recipe: OptimizationRecipe,
        normalized: NormalizedPriceData,
        universe: list[str] | tuple[str, ...],
        training_start: date,
        training_end: date,
        optimization_as_of: date,
        launch_date: date,
        historical_evaluation_end: date | None,
        live_tracking_end: date | None,
        package_version: str,
        code_version: str,
        calculation_version: str,
        description: str | None = None,
        benchmark_symbol: str | None = None,
        asset_types: Mapping[str, str] | None = None,
    ) -> ExperimentInitializationResult:
        """Create from a frozen recipe without any session-state result reuse."""
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
        assets = tuple(dict.fromkeys(str(item).strip().upper() for item in universe))
        experiment = self._registry.create(
            name=name,
            mode=mode,
            base_currency=base_currency,
            initial_capital=initial_capital,
            recipe=recipe,
            training_start=training_start,
            training_end=training_end,
            optimization_as_of=optimization_as_of,
            launch_date=launch_date,
            historical_evaluation_end=historical_evaluation_end,
            live_tracking_end=live_tracking_end,
            description=description,
            benchmark_symbol=benchmark_symbol,
        )
        if mode in {ExperimentMode.HISTORICAL_OOS, ExperimentMode.HYBRID}:
            replay = HistoricalReplayService(self._uow_factory).run(
                experiment_id=experiment.experiment_id,
                normalized=normalized,
                universe=assets,
                recipe=recipe,
                package_version=package_version,
                code_version=code_version,
                calculation_version=calculation_version,
                asset_types=asset_types,
            )
            return ExperimentInitializationResult(
                experiment=self._registry.get(experiment.experiment_id),
                historical_replay=replay,
            )

        bounded = _bounded_prices(
            normalized, start=training_start, end=launch_date
        )
        snapshot = build_point_in_time_snapshot(
            experiment=experiment,
            normalized=bounded,
            universe=assets,
            recipe=recipe,
            package_version=package_version,
            code_version=code_version,
            asset_types=asset_types,
        )
        states = value_fixed_holdings(
            experiment=experiment,
            snapshot=snapshot,
            normalized=bounded,
            cash_policy=recipe.cash,
            calculation_version=calculation_version,
        )
        launch_states = tuple(item for item in states if item.state_date == launch_date)
        if len(launch_states) != 1 or not launch_states[0].finalized:
            raise ValueError("Live Forward launch did not produce one finalized state")
        launch_forecast = build_origin_safe_forecast(
            normalized=bounded,
            state=launch_states[0],
            recipe=recipe.risk,
            cash_policy=recipe.cash,
            model_version=code_version,
        )
        with self._uow_factory() as uow:
            uow.snapshots.add(snapshot)
            uow.prices.add_many(bounded.observations())
            uow.valuations.write(launch_states[0])
            uow.forecasts.write(launch_forecast)
            activated = uow.experiments.transition(
                experiment.experiment_id, ExperimentStatus.ACTIVE
            )
            uow.events.add(
                ExperimentEvent(
                    experiment_id=experiment.experiment_id,
                    event_id=uuid4(),
                    event_type="live_forward_initialized",
                    effective_date=launch_date,
                    event_metadata={
                        "snapshot_id": str(snapshot.snapshot_id),
                        "launch_date": launch_date.isoformat(),
                        "session_optimizer_reused": False,
                    },
                )
            )
            uow.commit()
        return ExperimentInitializationResult(
            experiment=activated,
            historical_replay=None,
        )


__all__ = ["ExperimentCreationWorkflow", "ExperimentInitializationResult"]
