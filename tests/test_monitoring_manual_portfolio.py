"""Manual-only Phase 8 creation, replay, UI and live-update acceptance gates."""

from dataclasses import replace
from datetime import date, datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from var_cvar_crypto_risk.monitoring.database import (
    create_monitoring_engine,
    create_session_factory,
)
from var_cvar_crypto_risk.monitoring.domain import (
    DomainValidationError,
    ExperimentMode,
    ForecastEvaluationStatus,
    ImmutableRecordError,
)
from var_cvar_crypto_risk.monitoring.historical_replay import HistoricalReplayService
from var_cvar_crypto_risk.monitoring.live_update import LiveMonitoringService
from var_cvar_crypto_risk.monitoring.manual_portfolio import build_manual_snapshot
from var_cvar_crypto_risk.monitoring.models import Base
from var_cvar_crypto_risk.monitoring.prices import normalize_monitoring_prices
from var_cvar_crypto_risk.monitoring.providers import (
    PriceProviderRegistry,
    ProviderPriceBatch,
)
from var_cvar_crypto_risk.monitoring.recipes import (
    CashPolicy,
    ManualMonitoringRecipe,
    OptimizationRecipe,
    RiskMonitoringRecipe,
    SourceRecipe,
    monitoring_recipe_from_dict,
)
from var_cvar_crypto_risk.monitoring.repository import SqlAlchemyUnitOfWork
from var_cvar_crypto_risk.monitoring.services import ExperimentRegistry
from var_cvar_crypto_risk.monitoring.workflows import ExperimentCreationWorkflow
from var_cvar_crypto_risk.monitoring.dashboard import MonitoringReadService
from var_cvar_crypto_risk.monitoring.exports import export_experiment_bundle


def _frame():
    return pd.DataFrame(
        {
            "BTC": [100.0 + i + i % 3 for i in range(24)],
            "ETH": [50.0 + i * 0.7 + i % 4 for i in range(24)],
        },
        index=pd.date_range("2026-01-01", periods=24, freq="D"),
    )


def _recipe(weights=None, *, cash=False, source="fixture"):
    return ManualMonitoringRecipe(
        weights=weights if weights is not None else {"BTC": 0.60, "ETH": 0.40},
        risk=RiskMonitoringRecipe(estimation_window=4),
        cash=CashPolicy(enabled=cash),
        source=SourceRecipe(
            provider=source,
            refreshable=True,
            symbol_mapping={"BTC": "BTC-USD", "ETH": "ETH-USD"},
        ),
    )


@pytest.fixture
def store(tmp_path):
    engine = create_monitoring_engine(f"sqlite+pysqlite:///{tmp_path / 'manual.db'}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    yield engine, lambda: SqlAlchemyUnitOfWork(factory)
    engine.dispose()


@pytest.fixture
def monitoring_ui():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from var_cvar_crypto_risk.streamlit_ui import monitoring

    return monitoring


@pytest.fixture
def app_test(monitoring_ui):
    from streamlit.testing.v1 import AppTest

    return AppTest


@pytest.fixture
def no_optimizer(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Monitoring must not optimize manual weights")

    monkeypatch.setattr(
        "var_cvar_crypto_risk.monitoring.optimization_adapter._run_optimizer", forbidden
    )
    monkeypatch.setattr(
        "var_cvar_crypto_risk.monitoring.historical_replay.build_point_in_time_snapshot",
        forbidden,
    )


def _create(uow_factory, mode, *, recipe=None, frame=None):
    selected = recipe or _recipe()
    return ExperimentCreationWorkflow(uow_factory).create(
        name="manual portfolio",
        mode=mode,
        base_currency="USD",
        initial_capital=100_000.0,
        recipe=selected,
        normalized=normalize_monitoring_prices(
            _frame() if frame is None else frame, source=selected.source.provider
        ),
        universe=getattr(selected, "market_assets", ("BTC", "ETH")),
        training_start=date(2026, 1, 1),
        training_end=date(2026, 1, 10),
        optimization_as_of=date(2026, 1, 10),
        launch_date=date(2026, 1, 11),
        historical_evaluation_end=None
        if mode is ExperimentMode.LIVE_FORWARD
        else date(2026, 1, 16),
        live_tracking_end=None
        if mode is ExperimentMode.HISTORICAL_OOS
        else date(2026, 2, 1),
        package_version="1.0.0",
        code_version="manual-test",
        calculation_version="valuation-v1",
        benchmark_symbol="BTC",
    )


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"BTC": 0.70, "ETH": 0.40},
        {"BTC": 0.0, "ETH": 1.0},
        {"BTC": -0.10, "ETH": 1.10},
        {"BTC": float("nan"), "ETH": 0.40},
        {"BTC": float("inf")},
        {"BTC": 0.50, " btc ": 0.50},
        {"SOL": 1.0},
    ],
)
def test_invalid_manual_weights_are_not_normalized(weights):
    with pytest.raises(DomainValidationError):
        _recipe(weights)


def test_manual_recipe_roundtrip_and_legacy_compatibility():
    recipe = _recipe()
    restored = monitoring_recipe_from_dict(recipe.to_dict())
    assert isinstance(restored, ManualMonitoringRecipe)
    assert restored.to_dict() == recipe.to_dict()
    assert restored.fingerprint == recipe.fingerprint
    assert not {"assumptions", "scenario", "optimizer"}.intersection(recipe.to_dict())
    legacy = OptimizationRecipe()
    assert (
        monitoring_recipe_from_dict(legacy.to_dict()).fingerprint == legacy.fingerprint
    )
    with pytest.raises(DomainValidationError):
        monitoring_recipe_from_dict({"construction_method": "manual"})


@pytest.mark.parametrize("field", ["optimizer", "assumptions", "scenario"])
def test_manual_recipe_cannot_include_hidden_optimization_settings(field):
    payload = _recipe().to_dict()
    payload[field] = {}
    with pytest.raises(DomainValidationError, match="optimization settings"):
        monitoring_recipe_from_dict(payload)


@pytest.mark.parametrize("mode", list(ExperimentMode))
def test_manual_creation_uses_exact_weights_and_fixed_launch_quantities(
    store, no_optimizer, mode
):
    _, factory = store
    result = _create(factory, mode)
    with factory() as uow:
        snapshot = uow.snapshots.get_for_experiment(result.experiment.experiment_id)
        states = uow.valuations.list(result.experiment.experiment_id)
        forecasts = uow.forecasts.list(result.experiment.experiment_id)
    assert snapshot.objective == "manual"
    assert snapshot.solver == "none"
    assert snapshot.solver_status == "manual_validated"
    assert snapshot.launch_forecast["expected_return"] is None
    assert {a.asset: a.target_weight for a in snapshot.allocations} == {
        "BTC": 0.60,
        "ETH": 0.40,
    }
    launch_prices = _frame().loc["2026-01-11"]
    for allocation in snapshot.allocations:
        assert allocation.quantity == pytest.approx(
            100_000 * allocation.target_weight / launch_prices[allocation.asset]
        )
    assert states[0].nav == pytest.approx(100_000.0)
    assert states[0].daily_return == 0.0
    assert len(states) == (1 if mode is ExperimentMode.LIVE_FORWARD else 6)
    assert "monitoring_recipe" in result.experiment.source_metadata
    assert "optimization_recipe" not in result.experiment.source_metadata
    assert all(f.input_max_date <= f.origin_date < f.target_date for f in forecasts)
    assert all(
        a.quantity
        == pytest.approx(
            next(i.quantity for i in snapshot.allocations if i.asset == a.asset)
        )
        for state in states
        for a in state.asset_states
    )


def test_manual_cash_is_an_explicit_weight(store, no_optimizer):
    _, factory = store
    recipe = _recipe({"BTC": 0.50, "ETH": 0.30, "CASH": 0.20}, cash=True)
    result = _create(factory, ExperimentMode.HISTORICAL_OOS, recipe=recipe)
    with factory() as uow:
        snapshot = uow.snapshots.get_for_experiment(result.experiment.experiment_id)
        states = uow.valuations.list(result.experiment.experiment_id)
    cash = next(a for a in snapshot.allocations if a.is_cash)
    assert cash.quantity == cash.initial_value == 20_000.0
    assert cash.launch_price is None
    assert all(s.cash_value == 20_000.0 for s in states)
    with pytest.raises(DomainValidationError, match="cash"):
        _recipe({"BTC": 0.80, "CASH": 0.20})


def test_documented_offline_manual_example_is_executable(store, no_optimizer):
    _, factory = store
    prices = pd.read_csv(
        Path(__file__).parent / "fixtures" / "synthetic_daily_prices.csv",
        index_col="Date",
    )
    recipe = ManualMonitoringRecipe(
        weights={"BTC": 0.50, "ETH": 0.30, "SOL": 0.20},
        risk=RiskMonitoringRecipe(estimation_window=30),
        source=SourceRecipe(
            provider="uploaded_csv",
            symbol_mapping={"BTC": "BTC", "ETH": "ETH", "SOL": "SOL"},
        ),
    )
    result = ExperimentCreationWorkflow(factory).create(
        name="documented manual CSV example",
        mode=ExperimentMode.HISTORICAL_OOS,
        base_currency="USD",
        initial_capital=100_000.0,
        recipe=recipe,
        normalized=normalize_monitoring_prices(prices, source="uploaded_csv"),
        universe=recipe.market_assets,
        training_start=date(2024, 1, 1),
        training_end=date(2024, 3, 1),
        optimization_as_of=date(2024, 3, 1),
        launch_date=date(2024, 3, 2),
        historical_evaluation_end=date(2024, 4, 5),
        live_tracking_end=None,
        benchmark_symbol="BTC",
        package_version="1.0.0",
        code_version="offline-example",
        calculation_version="valuation-v1",
    )
    dashboard = MonitoringReadService(factory).load(result.experiment.experiment_id)
    assert dashboard.portfolio.iloc[0]["nav"] == pytest.approx(100_000.0)
    assert dashboard.portfolio.iloc[-1]["date"].date() == date(2024, 4, 5)
    assert dashboard.snapshot.assumptions["weights"] == dict(recipe.weights)
    assert dashboard.snapshot.objective == "manual"
    assert not dashboard.risk.empty
    assert dashboard.allocation.groupby("asset")["quantity"].nunique().eq(1).all()


def test_future_prices_cannot_change_manual_snapshot_and_replay_is_idempotent(
    store, no_optimizer
):
    _, factory = store
    recipe = _recipe()
    result = _create(factory, ExperimentMode.HISTORICAL_OOS, recipe=recipe)
    frame = _frame()
    changed = frame.copy()
    changed.loc["2026-01-12":] *= 4.0
    args = dict(
        experiment=result.experiment,
        universe=recipe.market_assets,
        recipe=recipe,
        package_version="1.0.0",
        code_version="manual-test",
    )
    first = build_manual_snapshot(
        normalized=normalize_monitoring_prices(frame, source="fixture"), **args
    )
    future = build_manual_snapshot(
        normalized=normalize_monitoring_prices(changed, source="fixture"), **args
    )
    assert first.allocations == future.allocations
    assert first.source_data_hash == future.source_data_hash
    replay = HistoricalReplayService(factory).run(
        experiment_id=result.experiment.experiment_id,
        normalized=normalize_monitoring_prices(frame, source="fixture"),
        universe=recipe.market_assets,
        recipe=recipe,
        package_version="1.0.0",
        code_version="manual-test",
        calculation_version="valuation-v1",
    )
    assert replay.state_counts.inserted == 0
    assert replay.price_counts.inserted == 0
    assert replay.forecast_counts.inserted == 0
    with pytest.raises(DomainValidationError, match="provenance"):
        replace(first, solver_status="optimal")
    bad = replace(first.allocations[0], quantity=first.allocations[0].quantity * 2)
    with pytest.raises(DomainValidationError, match="quantities"):
        replace(first, allocations=(bad, first.allocations[1]))


class FakeProvider:
    provider_name = "fixture"

    def fetch(self, request):
        return ProviderPriceBatch(
            prices=_frame(),
            actual_source=self.provider_name,
            quote_currency="USD",
            retrieved_at=request.requested_at,
            complete_through=date(2026, 1, 24),
        )


@pytest.mark.parametrize("mode", [ExperimentMode.LIVE_FORWARD, ExperimentMode.HYBRID])
def test_manual_live_updates_preserve_weights_snapshot_and_are_idempotent(
    store, no_optimizer, mode
):
    _, factory = store
    result = _create(factory, mode)
    service = LiveMonitoringService(factory, PriceProviderRegistry([FakeProvider()]))
    args = dict(
        requested_cutoff=date(2026, 1, 24),
        as_of=datetime(2026, 1, 21, 12, tzinfo=timezone.utc),
        code_version="manual-test",
        calculation_version="valuation-v1",
    )
    first = service.update_experiment(result.experiment.experiment_id, **args)
    assert first.actual_cutoff == date(2026, 1, 20)
    with factory() as uow:
        snapshot = uow.snapshots.get_for_experiment(result.experiment.experiment_id)
        states = uow.valuations.list(result.experiment.experiment_id)
        forecasts = uow.forecasts.list(result.experiment.experiment_id)
    assert states[-1].state_date == date(2026, 1, 20)
    assert snapshot.assumptions["weights"] == {"BTC": 0.60, "ETH": 0.40}
    assert any(
        f.evaluation_status is ForecastEvaluationStatus.EVALUATED for f in forecasts
    )
    second = service.update_experiment(result.experiment.experiment_id, **args)
    assert second.processed_dates == 0
    with factory() as uow:
        assert (
            uow.snapshots.get_for_experiment(result.experiment.experiment_id)
            == snapshot
        )
        assert len(uow.valuations.list(result.experiment.experiment_id)) == len(states)


def test_manual_ui_weight_parser_rejects_duplicates_and_keeps_percentages(
    monitoring_ui,
):
    table = pd.DataFrame({"Symbol": [" btc ", "ETH"], "Weight (%)": [60.0, 40.0]})
    assert monitoring_ui._parse_manual_weights(table) == {"BTC": 0.60, "ETH": 0.40}
    table.loc[1, "Symbol"] = "BTC"
    with pytest.raises(DomainValidationError, match="Duplicate"):
        monitoring_ui._parse_manual_weights(table)
    table.loc[1, "Symbol"] = None
    with pytest.raises(DomainValidationError, match="Every"):
        monitoring_ui._parse_manual_weights(table)


def test_new_creation_rejects_optimizer_recipe_before_persistence(store):
    _, factory = store
    with pytest.raises(DomainValidationError, match="manual"):
        _create(factory, ExperimentMode.LIVE_FORWARD, recipe=OptimizationRecipe())
    assert ExperimentRegistry(factory).list() == []


def test_manual_creation_ui_has_no_optimizer_controls(
    store, monkeypatch, no_optimizer, app_test
):
    engine, _ = store
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        conn.execute(
            text("INSERT INTO alembic_version VALUES ('0004_batch5_run_metadata')")
        )
    monkeypatch.setenv("QCPRF_MONITORING_DATABASE_URL", str(engine.url))
    at = app_test.from_file("app.py", default_timeout=60).run()
    next(w for w in at.radio if w.label == "Workspace").set_value(
        "Portfolio Monitor"
    ).run()
    next(w for w in at.radio if w.label == "Monitoring view").set_value(
        "Create Forward Test"
    ).run()
    assert not at.exception
    assert not {
        "Optimizer objective",
        "Scenario source",
        "Expected-return estimator",
        "Covariance estimator",
    }.intersection(w.label for w in at.selectbox)
    assert not any(w.label == "Maximum weight per asset" for w in at.slider)
    assert any(w.label == "Validate and create manual portfolio" for w in at.button)
    assert any("Manual portfolio" in w.value for w in at.markdown)


def test_manual_manifest_identifies_construction_and_weights(
    store, no_optimizer, monitoring_ui
):
    _, factory = store
    result = _create(factory, ExperimentMode.HISTORICAL_OOS)
    dashboard = MonitoringReadService(factory).load(result.experiment.experiment_id)
    payload = json.loads(monitoring_ui._experiment_manifest(dashboard))
    assert payload["methodology"]["construction_method"] == "manual"
    assert payload["methodology"]["weights"] == {"BTC": 0.60, "ETH": 0.40}
    assert "optimizer" not in payload["methodology"]


def test_manual_snapshot_accepts_explicit_later_launch_and_rejects_partial_prices(
    store, no_optimizer
):
    _, factory = store
    result = _create(factory, ExperimentMode.HISTORICAL_OOS)
    experiment = replace(result.experiment, launch_date=date(2026, 1, 13))
    args = dict(
        experiment=experiment,
        universe=("BTC", "ETH"),
        recipe=_recipe(),
        package_version="1.0.0",
        code_version="manual-test",
    )
    snapshot = build_manual_snapshot(
        normalized=normalize_monitoring_prices(_frame(), source="fixture"), **args
    )
    assert snapshot.allocations[0].launch_price == _frame().loc["2026-01-13", "BTC"]
    frame = _frame()
    frame.loc["2026-01-13", "ETH"] = float("nan")
    with pytest.raises(DomainValidationError, match="incomplete"):
        build_manual_snapshot(
            normalized=normalize_monitoring_prices(frame, source="fixture"), **args
        )
    with pytest.raises(DomainValidationError, match="completed UTC"):
        build_manual_snapshot(
            normalized=normalize_monitoring_prices(
                _frame(),
                source="fixture",
                retrieved_at=datetime(2026, 1, 13, 12, tzinfo=timezone.utc),
            ),
            **args,
        )
    with pytest.raises(DomainValidationError, match="base currency"):
        build_manual_snapshot(
            normalized=normalize_monitoring_prices(
                _frame(), source="fixture", quote_currency="EUR"
            ),
            **args,
        )


def test_manual_replay_rejects_changed_weights_and_exports_honest_snapshot(
    store, no_optimizer, tmp_path
):
    _, factory = store
    result = _create(factory, ExperimentMode.HISTORICAL_OOS)
    with factory() as uow:
        snapshot = uow.snapshots.get_for_experiment(result.experiment.experiment_id)
    forged = {**snapshot.assumptions, "weights": {"BTC": 0.50, "ETH": 0.50}}
    with pytest.raises(DomainValidationError, match="frozen recipe"):
        replace(snapshot, assumptions=forged)
    with pytest.raises(ImmutableRecordError):
        snapshot.activate()
    bundle = export_experiment_bundle(
        uow_factory=factory,
        experiment_id=result.experiment.experiment_id,
        output_directory=tmp_path / "export",
    )
    assert (bundle.parent / "portfolio_snapshot.json").exists()
    assert not (bundle.parent / "optimization_snapshot.json").exists()


def test_manual_ui_can_create_an_experiment_without_live_network_or_optimizer(
    store, monkeypatch, no_optimizer, app_test
):
    engine, factory = store
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        conn.execute(
            text("INSERT INTO alembic_version VALUES ('0004_batch5_run_metadata')")
        )
    monkeypatch.setenv("QCPRF_MONITORING_DATABASE_URL", str(engine.url))
    provider = FakeProvider()
    provider.provider_name = "coingecko"
    monkeypatch.setattr(
        "var_cvar_crypto_risk.streamlit_ui.monitoring.default_provider_registry",
        lambda: PriceProviderRegistry([provider]),
    )
    at = app_test.from_file("app.py", default_timeout=60).run()
    next(w for w in at.radio if w.label == "Workspace").set_value(
        "Portfolio Monitor"
    ).run()
    next(w for w in at.radio if w.label == "Monitoring view").set_value(
        "Create Forward Test"
    ).run()
    values = {
        "Risk history start": date(2026, 1, 1),
        "Allocation decision / risk-history cutoff": date(2026, 1, 10),
        "Launch date": date(2026, 1, 11),
        "Historical OOS evaluation end": date(2026, 1, 16),
    }
    for widget in at.date_input:
        if widget.label in values:
            widget.set_value(values[widget.label])
    at.run()
    at.session_state["monitor_manual_portfolio"] = {
        "edited_rows": {0: {"Weight (%)": 60.0}, 1: {"Weight (%)": 60.0}},
        "added_rows": [],
        "deleted_rows": [],
    }
    next(
        w for w in at.button if w.label == "Validate and create manual portfolio"
    ).click().run()
    assert not at.exception
    assert any("sum to 100%" in item.value for item in at.error)
    assert ExperimentRegistry(factory).list() == []
    at.session_state["monitor_manual_portfolio"] = {
        "edited_rows": {0: {"Weight (%)": 60.0}, 1: {"Weight (%)": 40.0}},
        "added_rows": [],
        "deleted_rows": [],
    }
    next(
        w for w in at.button if w.label == "Validate and create manual portfolio"
    ).click().run()
    assert not at.exception
    assert not at.error
    assert any("No optimization was performed" in item.value for item in at.success)
    experiments = ExperimentRegistry(factory).list()
    assert len(experiments) == 1
    with factory() as uow:
        snapshot = uow.snapshots.get_for_experiment(experiments[0].experiment_id)
    assert snapshot.assumptions["weights"] == {"BTC": 0.60, "ETH": 0.40}
