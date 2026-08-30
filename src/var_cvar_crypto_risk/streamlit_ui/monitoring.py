"""Persistent experiment-monitoring workspace for Streamlit.

The UI calls monitoring services and consumes chart-ready read models.  It does
not run a scheduler, perform hidden rebalancing, or recalculate financial
metrics inside presentation code.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
from uuid import UUID

import pandas as pd
from sqlalchemy import inspect, text
import streamlit as st

from var_cvar_crypto_risk import __version__
from var_cvar_crypto_risk.monitoring.charts import (
    allocation_chart,
    breach_timeline_chart,
    comparison_nav_chart,
    comparison_scatter_chart,
    drawdown_chart,
    drift_chart,
    forecast_realized_chart,
    nav_chart,
    risk_history_chart,
    target_current_chart,
)
from var_cvar_crypto_risk.monitoring.dashboard import (
    MODE_LABELS,
    MonitoringReadService,
)
from var_cvar_crypto_risk.monitoring.database import (
    create_monitoring_engine,
    create_session_factory,
    resolve_database_url,
    sanitized_database_label,
)
from var_cvar_crypto_risk.monitoring.domain import (
    DomainValidationError,
    ExperimentMode,
    ExperimentStatus,
)
from var_cvar_crypto_risk.monitoring.live_update import LiveMonitoringService
from var_cvar_crypto_risk.monitoring.prices import normalize_monitoring_prices
from var_cvar_crypto_risk.monitoring.providers import default_provider_registry
from var_cvar_crypto_risk.monitoring.recipes import (
    AssumptionRecipe,
    CashPolicy,
    OptimizationRecipe,
    OptimizerRecipe,
    RiskMonitoringRecipe,
    ScenarioRecipe,
    SourceRecipe,
)
from var_cvar_crypto_risk.monitoring.repository import SqlAlchemyUnitOfWork
from var_cvar_crypto_risk.monitoring.services import ExperimentRegistry
from var_cvar_crypto_risk.monitoring.workflows import ExperimentCreationWorkflow


MODE_BY_LABEL = {label: mode for mode, label in MODE_LABELS.items()}
CALCULATION_VERSION = "valuation-v1"
MONITORING_SCHEMA_REVISION = "0004_batch5_run_metadata"


@st.cache_resource(show_spinner=False)
def _database_resources(database_url: str):
    engine = create_monitoring_engine(database_url)
    session_factory = create_session_factory(engine)

    def uow_factory():
        return SqlAlchemyUnitOfWork(session_factory)

    return engine, uow_factory


def _schema_ready(engine) -> bool:
    inspector = inspect(engine)
    if not (
        inspector.has_table("experiments")
        and inspector.has_table("alembic_version")
    ):
        return False
    with engine.connect() as connection:
        revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
    return revision == MONITORING_SCHEMA_REVISION


def _experiment_options(frame: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    values = frame["experiment_id"].tolist() if not frame.empty else []
    labels = {
        row["experiment_id"]: (
            f"{row['name']} · {row['experiment_id'][:8]} · "
            f"{row['mode_label']} · {row['status']}"
        )
        for _, row in frame.iterrows()
    }
    return values, labels


def _select_experiment(read_service: MonitoringReadService, *, key: str):
    frame = read_service.list_experiments(include_archived=True)
    values, labels = _experiment_options(frame)
    if not values:
        st.info("No portfolio experiments exist yet.")
        return None
    selected = st.selectbox(
        "Experiment (name + authoritative ID)",
        values,
        format_func=lambda item: labels[item],
        key=key,
    )
    return UUID(selected)


def _format_percentage(value) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value) * 100:.2f}%"


def _format_money(value, currency: str) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{currency} {float(value):,.2f}"


def _safe_error_message(exc: Exception) -> str:
    """Expose validated user errors, but never provider/database internals."""
    if isinstance(exc, DomainValidationError):
        return str(exc)
    return f"Unexpected {type(exc).__name__}; inspect private application logs."


def _experiment_manifest(dashboard) -> bytes:
    experiment = dashboard.experiment
    snapshot = dashboard.snapshot
    recipe = dict(experiment.source_metadata.get("optimization_recipe", {}))
    experiment_id = str(experiment.experiment_id)
    payload = {
        "experiment_id": experiment_id,
        "name": experiment.name,
        "mode": experiment.mode.value,
        "mode_label": MODE_LABELS[experiment.mode],
        "status": experiment.status.value,
        "boundaries": {
            "training_start": experiment.training_start,
            "training_end": experiment.training_end,
            "optimization_as_of": experiment.optimization_as_of,
            "launch_date": experiment.launch_date,
            "historical_evaluation_end": experiment.historical_evaluation_end,
            "live_tracking_end": experiment.live_tracking_end,
        },
        "snapshot": (
            {
                "snapshot_id": str(snapshot.snapshot_id),
                "objective": snapshot.objective,
                "solver": snapshot.solver,
                "solver_status": snapshot.solver_status,
                "package_version": snapshot.package_version,
                "code_version": snapshot.code_version,
                "recipe_hash": snapshot.assumption_recipe_hash,
                "source_hash": snapshot.source_data_hash,
            }
            if snapshot is not None
            else None
        ),
        "methodology": {
            "assumptions": recipe.get("assumptions"),
            "scenario": recipe.get("scenario"),
            "optimizer": recipe.get("optimizer"),
            "risk": recipe.get("risk"),
            "cash": recipe.get("cash"),
            "source": recipe.get("source"),
        },
        "tables": {
            "portfolio": f"{experiment_id}_portfolio.csv",
            "allocation": f"{experiment_id}_allocation.csv",
            "risk": f"{experiment_id}_risk.csv",
        },
        "privacy": "private_by_default",
        "limitations": [
            "Fixed holdings; no rebalancing",
            "No fees, slippage, liquidity, taxes, or custody costs",
            "Research output; no performance guarantee or suitability claim",
        ],
    }
    return (
        json.dumps(payload, default=str, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _render_experiments(read_service, uow_factory) -> None:
    st.subheader("Experiments")
    include_archived = st.checkbox("Include archived experiments", value=False)
    frame = read_service.list_experiments(include_archived=include_archived)
    if frame.empty:
        st.info("No experiments found. Use Create Forward Test to initialize one.")
        return
    display = frame.copy()
    display["latest_nav"] = display["latest_nav"].map(
        lambda value: None if pd.isna(value) else round(float(value), 2)
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    values, labels = _experiment_options(frame)
    selected = st.selectbox(
        "Archive an experiment",
        values,
        format_func=lambda item: labels[item],
        key="monitor_archive_id",
    )
    row = frame[frame["experiment_id"] == selected].iloc[0]
    if row["status"] == ExperimentStatus.ARCHIVED.value:
        st.caption("This experiment is already archived. History remains queryable.")
        return
    confirm = st.checkbox(
        "I understand archive is irreversible in the current UI and does not delete history",
        key="monitor_archive_confirm",
    )
    if st.button("Archive experiment", disabled=not confirm, type="secondary"):
        ExperimentRegistry(uow_factory).archive(UUID(selected))
        st.success("Experiment archived; snapshot and history were retained.")
        st.rerun()


def _parse_symbol_mapping(raw: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DomainValidationError("Symbol mapping must be valid JSON") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise DomainValidationError("Symbol mapping must be a non-empty JSON object")
    return {str(key).strip().upper(): str(value).strip() for key, value in parsed.items()}


def _uploaded_prices(uploaded, *, source: str):
    if uploaded is None:
        raise DomainValidationError("Upload a wide daily price CSV")
    frame = pd.read_csv(uploaded)
    date_columns = [column for column in frame.columns if str(column).strip().lower() == "date"]
    if len(date_columns) != 1:
        raise DomainValidationError("CSV requires exactly one Date column")
    frame = frame.set_index(date_columns[0])
    return normalize_monitoring_prices(frame, source=source)


def _render_methodology_preview(
    *,
    mode: ExperimentMode,
    source: str,
    training_start: date,
    optimization_as_of: date,
    launch_date: date,
    horizon: int,
    confidence: float,
    covariance_method: str,
    expected_return_method: str,
) -> None:
    with st.expander("Methodology preview and point-in-time gate", expanded=True):
        st.markdown(
            f"""
- **Label:** {MODE_LABELS[mode]}
- **Information set:** training prices from `{training_start}` through `{optimization_as_of}` only
- **Launch:** first explicitly requested complete close on `{launch_date}`; launch return is zero
- **Source:** `{source}` with frozen symbol mapping
- **Expected return / covariance:** `{expected_return_method}` / `{covariance_method}`
- **Risk horizon / confidence:** `{horizon}` calendar day(s) / `{confidence:.1%}`
- **Post-launch policy:** fixed quantities, Simple-return wealth arithmetic, no re-optimization or rebalancing
"""
        )
        st.warning(
            "Replay and forward monitoring omit fees, slippage, liquidity, taxes, "
            "custody, and rebalancing. Results are conditional research evidence, "
            "not a performance guarantee or complete model validation."
        )


def _render_create(uow_factory) -> None:
    st.subheader("Create Forward Test")
    st.caption(
        "Historical modes rebuild from the frozen cutoff. Current Risk Lab optimizer "
        "session results are never reused."
    )
    today = date.today()
    name = st.text_input("Experiment name", value="BTC-ETH fixed-holdings experiment")
    description = st.text_area("Description", value="Research-only portfolio monitoring experiment.")
    mode_label = st.selectbox("Experiment mode", list(MODE_BY_LABEL))
    mode = MODE_BY_LABEL[mode_label]
    source_label = st.selectbox(
        "Price source",
        ["CoinGecko", "yfinance", "Uploaded daily price CSV"],
    )
    source = {
        "CoinGecko": "coingecko",
        "yfinance": "yfinance",
        "Uploaded daily price CSV": "uploaded_csv",
    }[source_label]
    if source == "uploaded_csv" and mode is not ExperimentMode.HISTORICAL_OOS:
        st.error("Uploaded files are not refreshable and cannot create Live Forward or Hybrid experiments.")
    uploaded = (
        st.file_uploader("Wide CSV: Date,BTC,ETH,...", type=["csv"])
        if source == "uploaded_csv"
        else None
    )
    default_mapping = (
        '{"BTC":"BTC-USD","ETH":"ETH-USD"}'
        if source == "yfinance"
        else '{"BTC":"bitcoin","ETH":"ethereum"}'
    )
    mapping_raw = st.text_area(
        "Frozen symbol mapping (JSON)",
        value=default_mapping,
        help="Keys are project symbols; values are provider coin IDs or tickers.",
    )
    benchmark = st.text_input("Benchmark symbol (must be present in the data)", value="BTC").strip().upper()

    date_cols = st.columns(4)
    training_start = date_cols[0].date_input("Training start", value=today - timedelta(days=410))
    training_end = date_cols[1].date_input("Training end", value=today - timedelta(days=10))
    optimization_as_of = date_cols[2].date_input("Optimization as-of", value=today - timedelta(days=10))
    launch_date = date_cols[3].date_input("Launch date", value=today - timedelta(days=9))
    historical_end = None
    if mode in {ExperimentMode.HISTORICAL_OOS, ExperimentMode.HYBRID}:
        historical_end = st.date_input("Historical OOS evaluation end", value=today - timedelta(days=1))
    track_to_date = st.checkbox("Set a finite live tracking end", value=False)
    live_end = (
        st.date_input("Live tracking end", value=today + timedelta(days=60))
        if track_to_date and mode in {ExperimentMode.LIVE_FORWARD, ExperimentMode.HYBRID}
        else None
    )

    settings = st.columns(4)
    initial_capital = settings[0].number_input("Initial capital", min_value=1.0, value=100_000.0)
    horizon = settings[1].number_input("Risk horizon (calendar days)", min_value=1, max_value=30, value=1)
    confidence = settings[2].selectbox("Confidence", [0.90, 0.95, 0.975, 0.99], index=1)
    estimation_window = settings[3].number_input("Risk estimation window", min_value=30, max_value=2000, value=252)
    assumption_cols = st.columns(4)
    expected_method = assumption_cols[0].selectbox(
        "Expected-return estimator", ["mean", "median", "trimmed", "winsorized", "shrinkage", "zero"]
    )
    covariance_method = assumption_cols[1].selectbox("Covariance estimator", ["sample", "shrinkage", "ewma"])
    scenario_source = assumption_cols[2].selectbox(
        "Scenario source", ["historical", "normal_mc", "student_t_mc"]
    )
    objective = assumption_cols[3].selectbox("Optimizer objective", ["min_cvar", "max_sharpe"])
    cash_enabled = st.checkbox("Include explicit cash asset", value=False)
    cash_rate = (
        st.number_input("Cash annual rate", min_value=-0.99, max_value=1.0, value=0.0, step=0.005)
        if cash_enabled
        else 0.0
    )
    max_weight = st.slider("Maximum weight per asset", min_value=0.05, max_value=1.0, value=1.0, step=0.05)

    _render_methodology_preview(
        mode=mode,
        source=source,
        training_start=training_start,
        optimization_as_of=optimization_as_of,
        launch_date=launch_date,
        horizon=int(horizon),
        confidence=float(confidence),
        covariance_method=covariance_method,
        expected_return_method=expected_method,
    )
    invalid_source = source == "uploaded_csv" and mode is not ExperimentMode.HISTORICAL_OOS
    if not st.button("Validate, rebuild, and create experiment", type="primary", disabled=invalid_source):
        return

    try:
        mapping = _parse_symbol_mapping(mapping_raw)
        source_recipe = SourceRecipe(
            provider=source,
            symbol_mapping=mapping,
            refreshable=source != "uploaded_csv",
        )
        recipe = OptimizationRecipe(
            assumptions=AssumptionRecipe(
                expected_return_method=expected_method,
                covariance_method=covariance_method,
            ),
            scenario=ScenarioRecipe(
                source=scenario_source,
                horizon_days=int(horizon),
                n_scenarios=5_000,
                random_seed=42,
            ),
            optimizer=OptimizerRecipe(
                objective=objective,
                confidence_level=float(confidence),
                long_only=True,
                max_weight=float(max_weight),
                risk_free_rate=float(cash_rate),
            ),
            risk=RiskMonitoringRecipe(
                var_method="historical",
                cvar_method="historical",
                confidence_level=float(confidence),
                horizon_days=int(horizon),
                estimation_window=int(estimation_window),
                evaluation_mode="overlapping",
            ),
            cash=CashPolicy(
                enabled=cash_enabled,
                mode=(
                    "annual_rate"
                    if cash_enabled and cash_rate != 0.0
                    else "zero"
                ),
                annual_rate=float(cash_rate),
            ),
            source=source_recipe,
        )
        universe = tuple(mapping)
        if source == "uploaded_csv":
            normalized = _uploaded_prices(uploaded, source=source)
        else:
            data_end = historical_end or launch_date
            symbols = universe
            if benchmark and benchmark not in symbols:
                symbols = (*symbols, benchmark)
            batch, fallback_used = default_provider_registry().fetch(
                source=source_recipe,
                symbols=symbols,
                start_date=training_start,
                end_date=data_end,
                requested_at=datetime.now(timezone.utc),
            )
            if fallback_used or batch.actual_source != source:
                raise DomainValidationError(
                    "Creation requires the frozen primary source; fallback data cannot silently define the snapshot"
                )
            normalized = normalize_monitoring_prices(
                batch.prices,
                source=batch.actual_source,
                quote_currency=batch.quote_currency,
                retrieved_at=batch.retrieved_at,
            )
        result = ExperimentCreationWorkflow(uow_factory).create(
            name=name,
            description=description,
            mode=mode,
            base_currency="USD",
            initial_capital=float(initial_capital),
            recipe=recipe,
            normalized=normalized,
            universe=universe,
            training_start=training_start,
            training_end=training_end,
            optimization_as_of=optimization_as_of,
            launch_date=launch_date,
            historical_evaluation_end=historical_end,
            live_tracking_end=live_end,
            benchmark_symbol=benchmark or None,
            package_version=__version__,
            code_version=os.getenv("QCPRF_CODE_VERSION", "working-tree"),
            calculation_version=CALCULATION_VERSION,
        )
    except Exception as exc:
        st.error(f"Experiment was not initialized: {_safe_error_message(exc)}")
        return
    st.session_state["monitor_selected_experiment"] = str(result.experiment.experiment_id)
    st.success(
        f"Created {result.experiment.name} with ID {result.experiment.experiment_id}. "
        "The optimization snapshot and methodology are frozen."
    )


def _render_snapshot(dashboard) -> None:
    snapshot = dashboard.snapshot
    if snapshot is None:
        st.warning("This draft has no activated optimization snapshot.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Asset": item.asset,
                    "Type": item.asset_type,
                    "Target weight": item.target_weight,
                    "Launch price": item.launch_price,
                    "Initial value": item.initial_value,
                    "Fixed quantity": item.quantity,
                    "Cash": item.is_cash,
                }
                for item in snapshot.allocations
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Snapshot `{snapshot.snapshot_id}` · recipe `{snapshot.assumption_recipe_hash}` · "
        f"source `{snapshot.source_data_hash}`"
    )


def _render_monitor(read_service) -> None:
    st.subheader("Portfolio Monitor")
    experiment_id = _select_experiment(read_service, key="monitor_experiment_id")
    if experiment_id is None:
        return
    dashboard = read_service.load(experiment_id)
    experiment = dashboard.experiment
    st.markdown(f"### {experiment.name}")
    st.caption(
        f"ID `{experiment.experiment_id}` · {dashboard.kpis['mode_label']} · "
        f"status `{experiment.status.value}`"
    )
    cards = st.columns(6)
    cards[0].metric("Latest NAV", _format_money(dashboard.kpis["nav"], experiment.base_currency))
    cards[1].metric("Cumulative return", _format_percentage(dashboard.kpis["cumulative_return"]))
    cards[2].metric("Realized volatility", _format_percentage(dashboard.kpis["realized_volatility"]))
    cards[3].metric("Maximum drawdown", _format_percentage(dashboard.kpis["maximum_drawdown"]))
    cards[4].metric("Total drift", _format_percentage(dashboard.kpis["total_drift"]))
    cards[5].metric("VaR exceptions", str(dashboard.kpis["var_breaches"]))

    overview, allocation_tab, risk_tab, forecast_tab, provenance = st.tabs(
        ["Overview", "Allocation & Drift", "Risk & Breaches", "Forecast vs Realized", "Snapshot & Provenance"]
    )
    with overview:
        unit = st.radio("NAV display", ["currency", "base_100"], horizontal=True)
        st.plotly_chart(
            nav_chart(
                dashboard.portfolio,
                unit=unit,
                historical_boundary=(
                    experiment.historical_evaluation_end
                    if experiment.mode is ExperimentMode.HYBRID
                    else None
                ),
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            drawdown_chart(
                dashboard.portfolio,
                historical_boundary=(
                    experiment.historical_evaluation_end
                    if experiment.mode is ExperimentMode.HYBRID
                    else None
                ),
            ),
            use_container_width=True,
        )
    with allocation_tab:
        st.info("Weights drift because quantities are fixed. This view does not recommend rebalancing.")
        st.plotly_chart(
            allocation_chart(
                dashboard.allocation,
                historical_boundary=(
                    experiment.historical_evaluation_end
                    if experiment.mode is ExperimentMode.HYBRID
                    else None
                ),
            ),
            use_container_width=True,
        )
        st.plotly_chart(target_current_chart(dashboard.allocation), use_container_width=True)
        st.plotly_chart(drift_chart(dashboard.allocation, dashboard.portfolio), use_container_width=True)
    with risk_tab:
        st.warning("A VaR exception is realized loss > VaR. CVaR/Expected Shortfall is not an exception threshold.")
        st.plotly_chart(risk_history_chart(dashboard.risk), use_container_width=True)
        st.plotly_chart(breach_timeline_chart(dashboard.risk), use_container_width=True)
        st.dataframe(dashboard.risk, use_container_width=True, hide_index=True)
    with forecast_tab:
        st.plotly_chart(forecast_realized_chart(dashboard.risk), use_container_width=True)
        st.caption("Forecast path unavailable unless genuine frozen path percentiles were persisted; no fan is synthesized.")
    with provenance:
        _render_snapshot(dashboard)
        st.json(
            {
                "training_start": experiment.training_start,
                "training_end": experiment.training_end,
                "optimization_as_of": experiment.optimization_as_of,
                "launch_date": experiment.launch_date,
                "historical_evaluation_end": experiment.historical_evaluation_end,
                "live_tracking_end": experiment.live_tracking_end,
            }
        )
        st.dataframe(dashboard.events, use_container_width=True, hide_index=True)

    st.markdown("### Downloads")
    downloads = st.columns(4)
    downloads[0].download_button(
        "Portfolio CSV",
        dashboard.portfolio.to_csv(index=False).encode("utf-8"),
        file_name=f"{experiment.experiment_id}_portfolio.csv",
        mime="text/csv",
    )
    downloads[1].download_button(
        "Allocation CSV",
        dashboard.allocation.to_csv(index=False).encode("utf-8"),
        file_name=f"{experiment.experiment_id}_allocation.csv",
        mime="text/csv",
    )
    downloads[2].download_button(
        "Risk CSV",
        dashboard.risk.to_csv(index=False).encode("utf-8"),
        file_name=f"{experiment.experiment_id}_risk.csv",
        mime="text/csv",
    )
    downloads[3].download_button(
        "Manifest JSON",
        _experiment_manifest(dashboard),
        file_name=f"{experiment.experiment_id}_manifest.json",
        mime="application/json",
    )
    st.warning(
        "Downloads are private by default and may contain holdings, quantities, prices, and realized performance. Review before publication."
    )


def _render_comparison(read_service) -> None:
    st.subheader("Experiment Comparison")
    frame = read_service.list_experiments(include_archived=True)
    values, labels = _experiment_options(frame)
    if len(values) < 2:
        st.info("At least two experiments are required for comparison.")
        return
    alignment_label = st.radio(
        "Alignment policy",
        ["Common calendar intersection", "Days since launch"],
        horizontal=True,
    )
    alignment = (
        "common_calendar"
        if alignment_label == "Common calendar intersection"
        else "launch_age"
    )
    selected = st.multiselect(
        "Experiments",
        values,
        default=values[:2],
        format_func=lambda item: labels[item],
    )
    if len(selected) < 2:
        st.info("Select at least two experiments after choosing the alignment policy.")
        return
    comparison = read_service.compare(
        [UUID(item) for item in selected], alignment=alignment
    )
    st.caption(
        "All paths are rebased to 100 at the shared comparison start, and the "
        "table/scatter use the same intersection. Common calendar uses shared "
        "dates and only consecutive dates for annualized realized volatility. "
        "Days since launch compares shared observation age, not the same market dates."
    )
    st.plotly_chart(
        comparison_nav_chart(comparison.nav, alignment=alignment),
        use_container_width=True,
    )
    st.plotly_chart(comparison_scatter_chart(comparison.summary), use_container_width=True)
    st.dataframe(comparison.summary, use_container_width=True, hide_index=True)


def _render_quality(read_service, uow_factory) -> None:
    st.subheader("Data Quality & Update Now")
    experiment_id = _select_experiment(read_service, key="quality_experiment_id")
    if experiment_id is None:
        return
    dashboard = read_service.load(experiment_id)
    experiment = dashboard.experiment
    complete = int((dashboard.quality["status"] == "complete").sum()) if not dashboard.quality.empty else 0
    incomplete = len(dashboard.quality) - complete
    cards = st.columns(4)
    cards[0].metric("Complete dates", complete)
    cards[1].metric("Incomplete dates", incomplete)
    cards[2].metric("Failed runs", dashboard.kpis["failed_runs"])
    latest_source = (
        dashboard.runs["actual_source"].dropna().iloc[-1]
        if not dashboard.runs.empty and dashboard.runs["actual_source"].notna().any()
        else "N/A"
    )
    cards[3].metric("Latest actual source", latest_source)
    st.dataframe(dashboard.quality, use_container_width=True, hide_index=True)
    st.dataframe(dashboard.runs, use_container_width=True, hide_index=True)

    can_update = (
        experiment.status is ExperimentStatus.ACTIVE
        and experiment.mode in {ExperimentMode.LIVE_FORWARD, ExperimentMode.HYBRID}
    )
    if not can_update:
        st.caption("Update Now is available only for active Live Forward or Hybrid experiments.")
        return
    st.warning("Update Now performs one bounded provider refresh. Streamlit does not run a scheduler or background loop.")
    if st.button("Update Now", type="primary"):
        try:
            result = LiveMonitoringService(
                uow_factory, default_provider_registry()
            ).update_experiment(
                experiment_id,
                code_version=os.getenv("QCPRF_CODE_VERSION", "working-tree"),
                calculation_version=CALCULATION_VERSION,
            )
        except Exception as exc:
            st.error(
                "Update failed; financial writes were rolled back: "
                f"{_safe_error_message(exc)}"
            )
            return
        st.success(
            f"Update complete through {result.actual_cutoff}; "
            f"processed {result.processed_dates} date(s) from {result.actual_source or 'no new source'}."
        )
        st.rerun()


def render_monitoring_workspace(project_root: str | Path | None = None) -> None:
    """Render persistent experiment monitoring without modifying the Risk Lab."""
    st.title("Portfolio Experiment Monitor")
    st.caption(
        "Persistent Historical OOS, Live Forward, and Hybrid fixed-holdings experiments"
    )
    database_url = resolve_database_url(
        project_root=Path(project_root) if project_root is not None else None
    )
    engine, uow_factory = _database_resources(database_url)
    if not _schema_ready(engine):
        st.info("The monitoring database has not been initialized.")
        st.code("uv run --locked --no-sync alembic upgrade head", language="bash")
        st.caption(
            f"Configured storage: {sanitized_database_label(engine.url)}. "
            "No credential or absolute database path is displayed."
        )
        return
    read_service = MonitoringReadService(uow_factory)
    page = st.radio(
        "Monitoring view",
        ["Experiments", "Create Forward Test", "Portfolio Monitor", "Comparison", "Data Quality"],
        horizontal=True,
        key="monitoring_view",
    )
    if page == "Experiments":
        _render_experiments(read_service, uow_factory)
    elif page == "Create Forward Test":
        _render_create(uow_factory)
    elif page == "Portfolio Monitor":
        _render_monitor(read_service)
    elif page == "Comparison":
        _render_comparison(read_service)
    else:
        _render_quality(read_service, uow_factory)

    st.divider()
    st.caption(
        "Research software only. No fees, slippage, liquidity, taxes, custody, "
        "rebalancing, suitability assessment, or performance guarantee."
    )
