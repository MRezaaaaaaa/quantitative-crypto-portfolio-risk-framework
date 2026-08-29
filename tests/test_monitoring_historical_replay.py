"""Sequential Historical OOS replay and no-look-ahead tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from var_cvar_crypto_risk.monitoring.database import (
    create_monitoring_engine,
    create_session_factory,
)
from var_cvar_crypto_risk.monitoring.domain import (
    ExperimentMode,
    ExperimentStatus,
    ForecastEvaluationStatus,
    ImmutableRecordError,
)
from var_cvar_crypto_risk.monitoring.historical_replay import (
    HistoricalReplayService,
)
from var_cvar_crypto_risk.monitoring.models import Base
from var_cvar_crypto_risk.monitoring.optimization_adapter import (
    build_point_in_time_snapshot,
)
from var_cvar_crypto_risk.monitoring.prices import normalize_monitoring_prices
from var_cvar_crypto_risk.monitoring.recipes import (
    OptimizationRecipe,
    RiskMonitoringRecipe,
    ScenarioRecipe,
    SourceRecipe,
)
from var_cvar_crypto_risk.monitoring.repository import SqlAlchemyUnitOfWork
from var_cvar_crypto_risk.monitoring.services import ExperimentRegistry


RETRIEVED_AT = datetime(2026, 1, 20, 12, tzinfo=timezone.utc)


def _recipe(
    *, horizon_days: int = 1, evaluation_mode: str = "overlapping"
) -> OptimizationRecipe:
    return OptimizationRecipe(
        scenario=ScenarioRecipe(
            source="historical",
            horizon_days=horizon_days,
            random_seed=17,
        ),
        risk=RiskMonitoringRecipe(
            horizon_days=horizon_days,
            estimation_window=4,
            evaluation_mode=evaluation_mode,
        ),
        source=SourceRecipe(
            provider="fixture",
            symbol_mapping={"BTC": "bitcoin", "ETH": "ethereum"},
            refreshable=True,
        ),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BTC": [
                100,
                101,
                99,
                102,
                101,
                104,
                103,
                105,
                104,
                106,
                107,
                105,
                108,
                109,
                106,
                110,
            ],
            "ETH": [
                50,
                51,
                50,
                52,
                51,
                53,
                52,
                54,
                53,
                55,
                56,
                54,
                57,
                58,
                55,
                59,
            ],
        },
        index=pd.date_range("2026-01-01", periods=16, freq="D"),
        dtype=float,
    )


def _setup(root: Path, *, mode: ExperimentMode, recipe: OptimizationRecipe):
    engine = create_monitoring_engine(
        f"sqlite+pysqlite:///{(root / 'monitoring.db').as_posix()}"
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)

    def uow_factory():
        return SqlAlchemyUnitOfWork(session_factory)

    registry = ExperimentRegistry(uow_factory)
    experiment = registry.create(
        name="historical replay",
        mode=mode,
        base_currency="USD",
        initial_capital=100_000.0,
        benchmark_symbol="BTC",
        recipe=recipe,
        training_start=date(2026, 1, 1),
        training_end=date(2026, 1, 10),
        optimization_as_of=date(2026, 1, 10),
        launch_date=date(2026, 1, 11),
        historical_evaluation_end=date(2026, 1, 16),
        live_tracking_end=(
            date(2026, 2, 1) if mode is ExperimentMode.HYBRID else None
        ),
    )
    return engine, uow_factory, experiment


def _run(
    root: Path,
    frame: pd.DataFrame,
    *,
    mode=ExperimentMode.HISTORICAL_OOS,
    recipe: OptimizationRecipe | None = None,
):
    recipe = recipe or _recipe()
    engine, uow_factory, experiment = _setup(root, mode=mode, recipe=recipe)
    normalized = normalize_monitoring_prices(
        frame, source="fixture", retrieved_at=RETRIEVED_AT
    )
    result = HistoricalReplayService(uow_factory).run(
        experiment_id=experiment.experiment_id,
        normalized=normalized,
        universe=["BTC", "ETH"],
        recipe=recipe,
        package_version="1.0.0",
        code_version="batch4-test",
        calculation_version="valuation-v1",
    )
    return engine, uow_factory, experiment, recipe, normalized, result


def test_historical_replay_builds_snapshot_states_and_matured_forecasts(
    tmp_path: Path,
) -> None:
    engine, uow_factory, experiment, _recipe_value, _normalized, result = _run(
        tmp_path, _frame()
    )
    try:
        assert result.final_status is ExperimentStatus.COMPLETED
        assert result.processed_dates == 6
        assert result.incomplete_dates == 0
        assert result.evaluated_forecasts == 5
        with uow_factory() as uow:
            persisted = uow.experiments.get(experiment.experiment_id)
            snapshot = uow.snapshots.get_for_experiment(experiment.experiment_id)
            states = uow.valuations.list(experiment.experiment_id)
            forecasts = uow.forecasts.list(experiment.experiment_id)
        assert persisted is not None and persisted.status is ExperimentStatus.COMPLETED
        assert snapshot is not None and snapshot.activated_at is not None
        assert [state.state_date for state in states] == list(
            pd.date_range("2026-01-11", "2026-01-16", freq="D").date
        )
        assert states[0].daily_return == 0.0
        assert states[0].nav == pytest.approx(experiment.initial_capital)
        assert states[0].benchmark_nav == pytest.approx(experiment.initial_capital)
        assert states[-1].benchmark_nav is not None
        assert len(forecasts) == 5
        assert all(
            forecast.evaluation_status is ForecastEvaluationStatus.EVALUATED
            for forecast in forecasts
        )
        assert all(
            forecast.input_max_date <= forecast.origin_date
            and forecast.forecast_metadata["future_frame_used"] is False
            for forecast in forecasts
        )
    finally:
        engine.dispose()


def test_replay_is_idempotent_and_does_not_duplicate_finalized_rows(
    tmp_path: Path,
) -> None:
    engine, uow_factory, experiment, recipe, normalized, first = _run(
        tmp_path, _frame()
    )
    try:
        second = HistoricalReplayService(uow_factory).run(
            experiment_id=experiment.experiment_id,
            normalized=normalized,
            universe=["BTC", "ETH"],
            recipe=recipe,
            package_version="1.0.0",
            code_version="batch4-test",
            calculation_version="valuation-v1",
        )
        assert second.snapshot_id == first.snapshot_id
        assert second.state_counts.inserted == 0
        assert second.state_counts.skipped == 6
        with uow_factory() as uow:
            assert len(uow.valuations.list(experiment.experiment_id)) == 6
            assert len(uow.forecasts.list(experiment.experiment_id)) == 5
    finally:
        engine.dispose()


def test_replay_rejects_a_preexisting_snapshot_not_equal_to_cutoff_rebuild(
    tmp_path: Path,
) -> None:
    recipe = _recipe()
    engine, uow_factory, experiment = _setup(
        tmp_path, mode=ExperimentMode.HISTORICAL_OOS, recipe=recipe
    )
    normalized = normalize_monitoring_prices(
        _frame(), source="fixture", retrieved_at=RETRIEVED_AT
    )
    legitimate = build_point_in_time_snapshot(
        experiment=experiment,
        normalized=normalize_monitoring_prices(
            normalized.prices.loc[:"2026-01-11"],
            source="fixture",
            retrieved_at=RETRIEVED_AT,
        ),
        universe=["BTC", "ETH"],
        recipe=recipe,
        package_version="1.0.0",
        code_version="batch4-test",
    )
    substituted = replace(
        legitimate,
        launch_forecast={**legitimate.launch_forecast, "cvar": 999.0},
    )
    try:
        with uow_factory() as uow:
            uow.snapshots.add(substituted)
            uow.commit()
        with pytest.raises(ImmutableRecordError, match="cutoff-rebuilt"):
            HistoricalReplayService(uow_factory).run(
                experiment_id=experiment.experiment_id,
                normalized=normalized,
                universe=["BTC", "ETH"],
                recipe=recipe,
                package_version="1.0.0",
                code_version="batch4-test",
                calculation_version="valuation-v1",
            )
    finally:
        engine.dispose()


def test_post_origin_price_changes_cannot_change_snapshot_or_earlier_forecasts(
    tmp_path: Path,
) -> None:
    first_engine, first_uow, first_experiment, *_ = _run(
        tmp_path / "first", _frame()
    )
    changed = _frame()
    changed.loc["2026-01-15":, "BTC"] = [1_000.0, 2_000.0]
    changed.loc["2026-01-15":, "ETH"] = [10.0, 20.0]
    second_engine, second_uow, second_experiment, *_ = _run(
        tmp_path / "second", changed
    )
    try:
        with first_uow() as uow:
            first_snapshot = uow.snapshots.get_for_experiment(
                first_experiment.experiment_id
            )
            first_forecasts = uow.forecasts.list(first_experiment.experiment_id)
        with second_uow() as uow:
            second_snapshot = uow.snapshots.get_for_experiment(
                second_experiment.experiment_id
            )
            second_forecasts = uow.forecasts.list(second_experiment.experiment_id)
        assert first_snapshot is not None and second_snapshot is not None
        assert first_snapshot.source_data_hash == second_snapshot.source_data_hash
        assert first_snapshot.assumptions == second_snapshot.assumptions
        assert first_snapshot.allocations == second_snapshot.allocations
        first_early = {
            item.origin_date: (
                item.forecast_var,
                item.forecast_cvar,
                item.forecast_volatility,
                item.input_data_hash,
            )
            for item in first_forecasts
            if item.origin_date <= date(2026, 1, 14)
        }
        second_early = {
            item.origin_date: (
                item.forecast_var,
                item.forecast_cvar,
                item.forecast_volatility,
                item.input_data_hash,
            )
            for item in second_forecasts
            if item.origin_date <= date(2026, 1, 14)
        }
        assert first_early == second_early
    finally:
        first_engine.dispose()
        second_engine.dispose()


def test_hybrid_replay_transitions_to_active_at_historical_boundary(
    tmp_path: Path,
) -> None:
    engine, uow_factory, experiment, *_rest, result = _run(
        tmp_path, _frame(), mode=ExperimentMode.HYBRID
    )
    try:
        assert result.final_status is ExperimentStatus.ACTIVE
        with uow_factory() as uow:
            events = uow.events.list(experiment.experiment_id)
        boundary_events = [
            item for item in events if item.event_type == "historical_replay_boundary_reached"
        ]
        assert boundary_events[0].effective_date == date(2026, 1, 16)
        assert boundary_events[0].event_metadata[
            "historical_oos_not_live_forward"
        ] is True
    finally:
        engine.dispose()


def test_non_overlapping_replay_schedules_disjoint_calendar_horizons(
    tmp_path: Path,
) -> None:
    recipe = _recipe(horizon_days=2, evaluation_mode="non_overlapping")
    engine, uow_factory, experiment, *_ = _run(
        tmp_path, _frame(), recipe=recipe
    )
    try:
        with uow_factory() as uow:
            forecasts = uow.forecasts.list(experiment.experiment_id)
        assert [(item.origin_date, item.target_date) for item in forecasts] == [
            (date(2026, 1, 11), date(2026, 1, 13)),
            (date(2026, 1, 13), date(2026, 1, 15)),
        ]
        assert all(
            item.evaluation_status is ForecastEvaluationStatus.EVALUATED
            for item in forecasts
        )
    finally:
        engine.dispose()


def test_missing_price_is_not_forward_filled_and_target_forecast_stays_pending(
    tmp_path: Path,
) -> None:
    frame = _frame()
    frame.loc["2026-01-13", "ETH"] = float("nan")
    engine, uow_factory, experiment, *_ = _run(tmp_path, frame)
    try:
        with uow_factory() as uow:
            states = uow.valuations.list(experiment.experiment_id)
            forecasts = uow.forecasts.list(experiment.experiment_id)
        missing_state = next(
            item for item in states if item.state_date == date(2026, 1, 13)
        )
        assert missing_state.finalized is False
        assert missing_state.nav is None
        target_forecast = next(
            item for item in forecasts if item.target_date == date(2026, 1, 13)
        )
        assert target_forecast.evaluation_status is ForecastEvaluationStatus.PENDING
    finally:
        engine.dispose()
