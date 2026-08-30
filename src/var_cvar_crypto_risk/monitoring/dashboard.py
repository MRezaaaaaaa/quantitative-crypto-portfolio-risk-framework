"""Chart-ready monitoring read models built only from persisted records.

This module is deliberately read-only.  It reshapes persisted portfolio,
allocation, forecast, quality, and run records for the Streamlit/Plotly layer;
it does not fetch prices, value holdings, recompute risk, or optimize.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Literal
from uuid import UUID

import pandas as pd

from .domain import (
    DailyPortfolioState,
    Experiment,
    ExperimentMode,
    MonitoringRunStatus,
    OptimizationSnapshot,
    RecordNotFoundError,
)
from .services import UnitOfWorkFactory


AlignmentPolicy = Literal["common_calendar", "launch_age"]

MODE_LABELS = {
    ExperimentMode.HISTORICAL_OOS: "Historical Out-of-Sample Replay",
    ExperimentMode.LIVE_FORWARD: "Live Forward Test",
    ExperimentMode.HYBRID: "Hybrid Historical OOS + Live Forward",
}


@dataclass(frozen=True)
class ExperimentDashboard:
    """One immutable read snapshot for the monitoring interface."""

    experiment: Experiment
    snapshot: OptimizationSnapshot | None
    portfolio: pd.DataFrame
    allocation: pd.DataFrame
    risk: pd.DataFrame
    quality: pd.DataFrame
    runs: pd.DataFrame
    events: pd.DataFrame
    kpis: dict[str, object]


@dataclass(frozen=True)
class ExperimentComparison:
    """Explicitly aligned comparison data for multiple experiments."""

    alignment: AlignmentPolicy
    nav: pd.DataFrame
    summary: pd.DataFrame


def _phase_label(experiment: Experiment, state_date) -> str:
    if experiment.mode is ExperimentMode.LIVE_FORWARD:
        return "live"
    if experiment.mode is ExperimentMode.HISTORICAL_OOS:
        return "historical_oos"
    boundary = experiment.historical_evaluation_end
    return "live" if boundary is not None and state_date > boundary else "historical_oos"


def _portfolio_frame(
    experiment: Experiment, states: list[DailyPortfolioState]
) -> pd.DataFrame:
    columns = [
        "date",
        "nav",
        "base_100_nav",
        "benchmark_nav",
        "benchmark_base_100",
        "daily_return",
        "cumulative_return",
        "realized_volatility",
        "drawdown",
        "maximum_drawdown",
        "total_drift",
        "quality_status",
        "finalized",
        "phase",
    ]
    rows = []
    for state in states:
        benchmark_base = (
            state.benchmark_nav / experiment.initial_capital * 100.0
            if state.benchmark_nav is not None
            else None
        )
        rows.append(
            {
                "date": pd.Timestamp(state.state_date),
                "nav": state.nav,
                "base_100_nav": state.base_100_nav,
                "benchmark_nav": state.benchmark_nav,
                "benchmark_base_100": benchmark_base,
                "daily_return": state.daily_return,
                "cumulative_return": state.cumulative_return,
                "realized_volatility": state.realized_volatility,
                "drawdown": state.drawdown,
                "maximum_drawdown": state.maximum_drawdown,
                "total_drift": state.total_drift,
                "quality_status": state.data_quality_status.value,
                "finalized": state.finalized,
                "phase": _phase_label(experiment, state.state_date),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("date").reset_index(drop=True)


def _allocation_frame(states: list[DailyPortfolioState]) -> pd.DataFrame:
    columns = [
        "date",
        "asset",
        "current_weight",
        "target_weight",
        "drift_percentage_points",
        "market_value",
        "quantity",
        "price",
        "is_cash",
        "finalized",
    ]
    rows = []
    for state in states:
        for item in state.asset_states:
            rows.append(
                {
                    "date": pd.Timestamp(state.state_date),
                    "asset": item.asset,
                    "current_weight": item.current_weight,
                    "target_weight": item.target_weight,
                    "drift_percentage_points": item.drift_percentage_points,
                    "market_value": item.market_value,
                    "quantity": item.quantity,
                    "price": item.price,
                    "is_cash": item.is_cash,
                    "finalized": state.finalized,
                }
            )
    frame = pd.DataFrame(rows, columns=columns)
    complete = frame[frame["finalized"] & frame["current_weight"].notna()]
    if not complete.empty:
        totals = complete.groupby("date")["current_weight"].sum()
        invalid = totals[~totals.map(lambda value: math.isclose(value, 1.0, abs_tol=1e-8))]
        if not invalid.empty:
            raise ValueError("persisted complete allocation weights do not sum to one")
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


def _risk_frame(forecasts) -> pd.DataFrame:
    columns = [
        "forecast_id",
        "origin_date",
        "target_date",
        "horizon_days",
        "confidence_level",
        "var_method",
        "cvar_method",
        "forecast_var",
        "forecast_cvar",
        "forecast_volatility",
        "realized_loss",
        "var_breach",
        "evaluation_status",
        "portfolio_definition",
        "input_max_date",
        "model_version",
    ]
    rows = [
        {
            "forecast_id": str(item.forecast_id),
            "origin_date": pd.Timestamp(item.origin_date),
            "target_date": pd.Timestamp(item.target_date),
            "horizon_days": item.horizon_days,
            "confidence_level": item.confidence_level,
            "var_method": item.var_method,
            "cvar_method": item.cvar_method,
            "forecast_var": item.forecast_var,
            "forecast_cvar": item.forecast_cvar,
            "forecast_volatility": item.forecast_volatility,
            "realized_loss": item.realized_horizon_loss,
            "var_breach": item.var_breach,
            "evaluation_status": item.evaluation_status.value,
            "portfolio_definition": item.portfolio_definition,
            "input_max_date": pd.Timestamp(item.input_max_date),
            "model_version": item.model_version,
        }
        for item in forecasts
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("origin_date").reset_index(drop=True)


def _quality_frame(states: list[DailyPortfolioState]) -> pd.DataFrame:
    columns = ["date", "status", "finalized", "missing_assets", "metadata"]
    rows = []
    for state in states:
        metadata = dict(state.quality_metadata)
        rows.append(
            {
                "date": pd.Timestamp(state.state_date),
                "status": state.data_quality_status.value,
                "finalized": state.finalized,
                "missing_assets": ", ".join(metadata.get("missing_assets", [])),
                "metadata": json.dumps(metadata, sort_keys=True, allow_nan=False),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("date").reset_index(drop=True)


def _runs_frame(runs) -> pd.DataFrame:
    columns = [
        "run_id",
        "status",
        "requested_cutoff",
        "actual_cutoff",
        "actual_source",
        "inserted_count",
        "updated_count",
        "skipped_count",
        "warning_count",
        "error_code",
        "error_summary",
        "started_at",
        "ended_at",
    ]
    rows = [
        {
            "run_id": str(item.run_id),
            "status": item.status.value,
            "requested_cutoff": item.requested_cutoff,
            "actual_cutoff": item.actual_cutoff,
            "actual_source": item.run_metadata.get("actual_source"),
            "inserted_count": item.inserted_count,
            "updated_count": item.updated_count,
            "skipped_count": item.skipped_count,
            "warning_count": item.warning_count,
            "error_code": item.error_code,
            "error_summary": item.error_summary,
            "started_at": item.started_at,
            "ended_at": item.ended_at,
        }
        for item in runs
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("started_at").reset_index(drop=True)


def _events_frame(events) -> pd.DataFrame:
    columns = ["event_id", "event_type", "effective_date", "created_at", "metadata"]
    rows = [
        {
            "event_id": str(item.event_id),
            "event_type": item.event_type,
            "effective_date": item.effective_date,
            "created_at": item.created_at,
            "metadata": json.dumps(item.event_metadata, sort_keys=True, allow_nan=False),
        }
        for item in events
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("created_at").reset_index(drop=True)


def _kpis(
    experiment: Experiment,
    portfolio: pd.DataFrame,
    risk: pd.DataFrame,
    runs: pd.DataFrame,
) -> dict[str, object]:
    finalized = portfolio[portfolio["finalized"]]
    latest = finalized.iloc[-1] if not finalized.empty else None
    evaluated = risk[risk["evaluation_status"] == "evaluated"]
    breaches = int(evaluated["var_breach"].eq(True).sum()) if not evaluated.empty else 0
    failed_runs = (
        int((runs["status"] == MonitoringRunStatus.FAILED.value).sum())
        if not runs.empty
        else 0
    )
    return {
        "mode_label": MODE_LABELS[experiment.mode],
        "latest_date": latest["date"].date() if latest is not None else None,
        "nav": latest["nav"] if latest is not None else None,
        "cumulative_return": latest["cumulative_return"] if latest is not None else None,
        "realized_volatility": latest["realized_volatility"] if latest is not None else None,
        "maximum_drawdown": latest["maximum_drawdown"] if latest is not None else None,
        "total_drift": latest["total_drift"] if latest is not None else None,
        "evaluated_forecasts": len(evaluated),
        "var_breaches": breaches,
        "failed_runs": failed_runs,
    }


class MonitoringReadService:
    """Read persisted experiment records into chart-ready data frames."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    def list_experiments(self, *, include_archived: bool = False) -> pd.DataFrame:
        columns = [
            "experiment_id",
            "name",
            "mode",
            "mode_label",
            "status",
            "launch_date",
            "historical_boundary",
            "live_end",
            "latest_complete_date",
            "latest_nav",
            "latest_update",
            "quality_status",
        ]
        with self._uow_factory() as uow:
            experiments = uow.experiments.list(include_archived=include_archived)
            rows = []
            for experiment in experiments:
                states = uow.valuations.list(experiment.experiment_id)
                runs = uow.runs.list(experiment.experiment_id)
                complete = [item for item in states if item.finalized]
                latest = complete[-1] if complete else None
                latest_run = runs[-1] if runs else None
                rows.append(
                    {
                        "experiment_id": str(experiment.experiment_id),
                        "name": experiment.name,
                        "mode": experiment.mode.value,
                        "mode_label": MODE_LABELS[experiment.mode],
                        "status": experiment.status.value,
                        "launch_date": experiment.launch_date,
                        "historical_boundary": experiment.historical_evaluation_end,
                        "live_end": experiment.live_tracking_end,
                        "latest_complete_date": latest.state_date if latest else None,
                        "latest_nav": latest.nav if latest else None,
                        "latest_update": latest_run.ended_at if latest_run else None,
                        "quality_status": (
                            latest.data_quality_status.value if latest else "no_data"
                        ),
                    }
                )
        return pd.DataFrame(rows, columns=columns)

    def load(self, experiment_id: UUID) -> ExperimentDashboard:
        with self._uow_factory() as uow:
            experiment = uow.experiments.get(experiment_id)
            if experiment is None:
                raise RecordNotFoundError(f"experiment {experiment_id} does not exist")
            snapshot = uow.snapshots.get_for_experiment(experiment_id)
            states = uow.valuations.list(experiment_id)
            forecasts = uow.forecasts.list(experiment_id)
            runs = uow.runs.list(experiment_id)
            events = uow.events.list(experiment_id)
        portfolio = _portfolio_frame(experiment, states)
        allocation = _allocation_frame(states)
        risk = _risk_frame(forecasts)
        run_frame = _runs_frame(runs)
        return ExperimentDashboard(
            experiment=experiment,
            snapshot=snapshot,
            portfolio=portfolio,
            allocation=allocation,
            risk=risk,
            quality=_quality_frame(states),
            runs=run_frame,
            events=_events_frame(events),
            kpis=_kpis(experiment, portfolio, risk, run_frame),
        )

    def compare(
        self,
        experiment_ids: list[UUID] | tuple[UUID, ...],
        *,
        alignment: AlignmentPolicy,
    ) -> ExperimentComparison:
        if alignment not in {"common_calendar", "launch_age"}:
            raise ValueError("comparison alignment must be explicit")
        if len(experiment_ids) < 2:
            raise ValueError("comparison requires at least two experiments")
        dashboards = [self.load(item) for item in experiment_ids]
        series: list[pd.Series] = []
        labels: list[str] = []
        for dashboard in dashboards:
            complete = dashboard.portfolio[dashboard.portfolio["finalized"]].copy()
            label = f"{dashboard.experiment.name} · {str(dashboard.experiment.experiment_id)[:8]}"
            if alignment == "common_calendar":
                values = complete.set_index("date")["base_100_nav"].rename(label)
            else:
                launch_age = (
                    complete["date"]
                    - pd.Timestamp(dashboard.experiment.launch_date)
                ).dt.days
                values = pd.Series(
                    complete["base_100_nav"].to_numpy(),
                    index=pd.Index(launch_age, name="days_since_launch"),
                    name=label,
                )
            series.append(values)
            labels.append(label)
        nav = pd.concat(series, axis=1, join="inner").sort_index().dropna()
        nav.index.name = "date" if alignment == "common_calendar" else "days_since_launch"
        if not nav.empty:
            nav = nav.divide(nav.iloc[0]).multiply(100.0)

        summary_rows = []
        for dashboard, label in zip(dashboards, labels, strict=True):
            values = nav[label] if label in nav else pd.Series(dtype=float)
            if values.empty:
                cumulative_return = None
                realized_volatility = None
                maximum_drawdown = None
                var_breaches = 0
            else:
                cumulative_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
                returns = values.pct_change().dropna()
                if alignment == "common_calendar":
                    intervals = values.index.to_series().diff().dt.days
                    returns = returns[intervals.loc[returns.index] == 1]
                realized_volatility = (
                    float(returns.std(ddof=1) * math.sqrt(365.0))
                    if len(returns) >= 2
                    else None
                )
                maximum_drawdown = float((values / values.cummax() - 1.0).min())
                evaluated = dashboard.risk[
                    dashboard.risk["evaluation_status"].eq("evaluated")
                ].copy()
                if alignment == "common_calendar":
                    comparable_dates = set(pd.DatetimeIndex(nav.index).normalize())
                    eligible = evaluated["target_date"].isin(comparable_dates)
                else:
                    ages = (
                        evaluated["target_date"]
                        - pd.Timestamp(dashboard.experiment.launch_date)
                    ).dt.days
                    eligible = ages.isin(nav.index)
                var_breaches = int(
                    evaluated.loc[eligible, "var_breach"].eq(True).sum()
                )
            summary_rows.append(
                {
                    "experiment_id": str(dashboard.experiment.experiment_id),
                    "name": dashboard.experiment.name,
                    "mode": MODE_LABELS[dashboard.experiment.mode],
                    "launch_date": dashboard.experiment.launch_date,
                    "historical_boundary": dashboard.experiment.historical_evaluation_end,
                    "observations": len(values),
                    "cumulative_return": cumulative_return,
                    "realized_volatility": realized_volatility,
                    "maximum_drawdown": maximum_drawdown,
                    "var_breaches": var_breaches,
                }
            )
        return ExperimentComparison(
            alignment=alignment,
            nav=nav,
            summary=pd.DataFrame(summary_rows),
        )


__all__ = [
    "AlignmentPolicy",
    "ExperimentComparison",
    "ExperimentDashboard",
    "MODE_LABELS",
    "MonitoringReadService",
]
