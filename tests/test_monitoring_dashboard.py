"""Read-model, Plotly, and UI-boundary tests for monitoring Batch 6."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("plotly")

from var_cvar_crypto_risk.monitoring.charts import (  # noqa: E402
    allocation_chart,
    breach_timeline_chart,
    comparison_nav_chart,
    comparison_scatter_chart,
    drawdown_chart,
    drift_chart,
    forecast_realized_chart,
    nav_chart,
    risk_history_chart,
    stable_asset_color,
    target_current_chart,
)
from var_cvar_crypto_risk.monitoring.dashboard import (  # noqa: E402
    MonitoringReadService,
)
from var_cvar_crypto_risk.monitoring.database import (  # noqa: E402
    create_monitoring_engine,
    create_session_factory,
)
from var_cvar_crypto_risk.monitoring.domain import (  # noqa: E402
    ExperimentMode,
    ExperimentStatus,
)
from var_cvar_crypto_risk.monitoring.models import Base  # noqa: E402
from var_cvar_crypto_risk.monitoring.prices import (  # noqa: E402
    normalize_monitoring_prices,
)
from var_cvar_crypto_risk.monitoring.recipes import (  # noqa: E402
    OptimizationRecipe,
    RiskMonitoringRecipe,
    ScenarioRecipe,
    SourceRecipe,
)
from var_cvar_crypto_risk.monitoring.repository import (  # noqa: E402
    SqlAlchemyUnitOfWork,
)
from var_cvar_crypto_risk.monitoring.workflows import (  # noqa: E402
    ExperimentCreationWorkflow,
)
from var_cvar_crypto_risk.streamlit_ui.monitoring import (  # noqa: E402
    _experiment_manifest,
)


RETRIEVED_AT = datetime(2026, 1, 20, 12, tzinfo=timezone.utc)


def _frame(scale: float = 1.0) -> pd.DataFrame:
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
    ) * scale


def _recipe() -> OptimizationRecipe:
    return OptimizationRecipe(
        scenario=ScenarioRecipe(
            source="historical",
            horizon_days=1,
            random_seed=19,
        ),
        risk=RiskMonitoringRecipe(
            horizon_days=1,
            estimation_window=4,
            evaluation_mode="overlapping",
        ),
        source=SourceRecipe(
            provider="fixture",
            symbol_mapping={"BTC": "bitcoin", "ETH": "ethereum"},
            refreshable=True,
        ),
    )


@pytest.fixture
def dashboard_store(tmp_path: Path):
    engine = create_monitoring_engine(
        f"sqlite+pysqlite:///{(tmp_path / 'monitoring.db').as_posix()}"
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)

    def uow_factory():
        return SqlAlchemyUnitOfWork(session_factory)

    workflow = ExperimentCreationWorkflow(uow_factory)

    def create(
        name: str,
        *,
        scale: float = 1.0,
        optimization_as_of: date = date(2026, 1, 10),
        launch_date: date = date(2026, 1, 11),
        missing_date: date | None = None,
    ):
        frame = _frame(scale)
        if missing_date is not None:
            frame.loc[pd.Timestamp(missing_date), "ETH"] = float("nan")
        normalized = normalize_monitoring_prices(
            frame, source="fixture", retrieved_at=RETRIEVED_AT
        )
        return workflow.create(
            name=name,
            description="offline Batch 6 chart fixture",
            mode=ExperimentMode.HISTORICAL_OOS,
            base_currency="USD",
            initial_capital=100_000.0,
            recipe=_recipe(),
            normalized=normalized,
            universe=("BTC", "ETH"),
            training_start=date(2026, 1, 1),
            training_end=optimization_as_of,
            optimization_as_of=optimization_as_of,
            launch_date=launch_date,
            historical_evaluation_end=date(2026, 1, 16),
            live_tracking_end=None,
            benchmark_symbol="BTC",
            package_version="1.0.0",
            code_version="batch6-test",
            calculation_version="valuation-v1",
        )

    try:
        yield uow_factory, create
    finally:
        engine.dispose()


def test_dashboard_reads_persisted_monitoring_records_without_revaluation(
    dashboard_store,
) -> None:
    uow_factory, create = dashboard_store
    result = create("dashboard experiment")
    dashboard = MonitoringReadService(uow_factory).load(
        result.experiment.experiment_id
    )

    assert dashboard.experiment.status is ExperimentStatus.COMPLETED
    assert len(dashboard.portfolio) == 6
    assert dashboard.portfolio["phase"].eq("historical_oos").all()
    assert dashboard.kpis["latest_date"] == date(2026, 1, 16)
    assert dashboard.kpis["realized_volatility"] is not None
    complete = dashboard.allocation[dashboard.allocation["finalized"]]
    assert complete.groupby("date")["current_weight"].sum().to_numpy() == pytest.approx(
        1.0
    )
    assert dashboard.risk["input_max_date"].le(
        dashboard.risk["origin_date"]
    ).all()

    manifest_bytes = _experiment_manifest(dashboard)
    manifest = json.loads(manifest_bytes)
    assert manifest["methodology"]["risk"]["horizon_days"] == 1
    assert manifest["methodology"]["risk"]["confidence_level"] == 0.95
    assert manifest["snapshot"]["code_version"] == "batch6-test"
    assert manifest["tables"]["risk"].endswith("_risk.csv")
    lowered = manifest_bytes.lower()
    assert b"/users/" not in lowered
    assert b"password" not in lowered
    assert b"api_key" not in lowered


def test_monitoring_charts_keep_financial_semantics_and_stable_colors(
    dashboard_store,
) -> None:
    uow_factory, create = dashboard_store
    result = create("chart experiment")
    dashboard = MonitoringReadService(uow_factory).load(
        result.experiment.experiment_id
    )

    nav = nav_chart(
        dashboard.portfolio,
        unit="base_100",
        historical_boundary=date(2026, 1, 14),
    )
    assert {trace.name for trace in nav.data} == {"Portfolio", "Benchmark"}
    assert nav.layout.shapes and nav.layout.annotations

    allocation = allocation_chart(
        dashboard.allocation,
        historical_boundary=date(2026, 1, 14),
    )
    assert all(trace.stackgroup == "one" for trace in allocation.data)
    assert all(trace.groupnorm == "percent" for trace in allocation.data)
    for trace in allocation.data:
        assert trace.line.color == stable_asset_color(trace.name)
        assert "current_weight" not in trace.hovertemplate

    breach = breach_timeline_chart(dashboard.risk)
    assert all("CVaR exception" not in str(trace.name) for trace in breach.data)
    assert "CVaR is not an exception threshold" in breach.layout.title.text

    forecast = forecast_realized_chart(dashboard.risk)
    assert all(trace.type != "scatter" or trace.fill is None for trace in forecast.data)
    assert any(
        "point forecasts only" in annotation.text
        for annotation in forecast.layout.annotations
    )

    target_current = target_current_chart(dashboard.allocation)
    assert {trace.name for trace in target_current.data} == {"Target", "Current"}
    assert target_current.layout.barmode == "group"

    drift = drift_chart(dashboard.allocation, dashboard.portfolio)
    assert [trace.type for trace in drift.data] == ["heatmap", "scatter"]

    drawdown = drawdown_chart(dashboard.portfolio)
    assert drawdown.data[0].fill == "tozeroy"

    risk_history = risk_history_chart(dashboard.risk)
    assert {trace.name for trace in risk_history.data} == {
        "Forecast VaR",
        "Forecast CVaR / ES",
        "Realized horizon loss",
    }


def test_comparison_requires_an_explicit_alignment_policy(dashboard_store) -> None:
    uow_factory, create = dashboard_store
    first = create("first experiment")
    second = create(
        "second experiment",
        optimization_as_of=date(2026, 1, 11),
        launch_date=date(2026, 1, 12),
        missing_date=date(2026, 1, 14),
    )
    service = MonitoringReadService(uow_factory)
    ids = (first.experiment.experiment_id, second.experiment.experiment_id)

    calendar = service.compare(ids, alignment="common_calendar")
    launch_age = service.compare(ids, alignment="launch_age")

    assert calendar.nav.index.name == "date"
    assert launch_age.nav.index.name == "days_since_launch"
    assert list(launch_age.nav.index) == [0, 1, 3, 4]
    assert calendar.nav.index.min() == pd.Timestamp("2026-01-12")
    assert calendar.nav.iloc[0].to_numpy() == pytest.approx(100.0)
    assert launch_age.nav.iloc[0].to_numpy() == pytest.approx(100.0)
    assert len(calendar.summary) == 2
    assert calendar.summary["observations"].eq(4).all()
    assert len(comparison_nav_chart(calendar.nav, alignment=calendar.alignment).data) == 2
    comparison_scatter = comparison_scatter_chart(calendar.summary)
    assert comparison_scatter.data[0].mode == "markers+text"
    with pytest.raises(ValueError, match="explicit"):
        service.compare(ids, alignment="implicit")  # type: ignore[arg-type]


def test_empty_chart_states_do_not_invent_observations() -> None:
    empty_portfolio = pd.DataFrame(columns=["finalized"])
    empty_allocation = pd.DataFrame(
        columns=["finalized", "current_weight", "drift_percentage_points"]
    )
    empty_risk = pd.DataFrame(
        columns=["realized_loss", "forecast_var", "var_breach"]
    )

    assert "No finalized NAV" in nav_chart(
        empty_portfolio, unit="currency"
    ).layout.annotations[0].text
    assert "No finalized allocation" in allocation_chart(
        empty_allocation
    ).layout.annotations[0].text
    assert "No evaluated VaR" in breach_timeline_chart(
        empty_risk
    ).layout.annotations[0].text
    assert "Forecast path unavailable" in forecast_realized_chart(
        empty_risk
    ).layout.annotations[0].text
