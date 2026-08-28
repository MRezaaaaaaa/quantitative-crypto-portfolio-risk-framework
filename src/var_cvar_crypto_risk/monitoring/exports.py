"""Explicit private-by-default CSV/JSON experiment export bundles."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd

from .domain import RecordNotFoundError
from .services import UnitOfWorkFactory


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_experiment_bundle(
    *,
    uow_factory: UnitOfWorkFactory,
    experiment_id: UUID,
    output_directory: str | Path,
    generated_at: datetime | None = None,
) -> Path:
    """Export one experiment without URLs, credentials, or local DB paths.

    Holdings and realized performance can still be sensitive, so the manifest
    labels the bundle private-by-default. Publication is a separate deliberate
    user action.
    """
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    with uow_factory() as uow:
        experiment = uow.experiments.get(experiment_id)
        if experiment is None:
            raise RecordNotFoundError(f"experiment {experiment_id} does not exist")
        snapshot = uow.snapshots.get_for_experiment(experiment_id)
        states = uow.valuations.list(experiment_id)
        events = uow.events.list(experiment_id)
        source_info = experiment.source_metadata.get("source", {})
        provider = source_info.get("provider") if isinstance(source_info, dict) else None
        quote = (
            source_info.get("quote_currency")
            if isinstance(source_info, dict)
            else experiment.base_currency
        )
        assets = (
            [item.asset for item in snapshot.allocations if not item.is_cash]
            if snapshot is not None
            else []
        )
        if experiment.benchmark_symbol:
            assets.append(experiment.benchmark_symbol)
        observations = uow.prices.list(
            symbols=assets or None,
            start=experiment.training_start,
            end=experiment.live_tracking_end or experiment.historical_evaluation_end,
            source=provider,
            quote_currency=quote,
        )

    safe_source = {}
    if isinstance(source_info, dict):
        safe_source = {
            key: source_info.get(key)
            for key in (
                "provider",
                "quote_currency",
                "symbol_mapping",
                "refreshable",
            )
            if key in source_info
        }
    experiment_payload = {
        "experiment_id": str(experiment.experiment_id),
        "name": experiment.name,
        "description": experiment.description,
        "mode": experiment.mode.value,
        "status": experiment.status.value,
        "base_currency": experiment.base_currency,
        "initial_capital": experiment.initial_capital,
        "benchmark_symbol": experiment.benchmark_symbol,
        "training_start": experiment.training_start.isoformat()
        if experiment.training_start
        else None,
        "training_end": experiment.training_end.isoformat()
        if experiment.training_end
        else None,
        "optimization_as_of": experiment.optimization_as_of.isoformat()
        if experiment.optimization_as_of
        else None,
        "launch_date": experiment.launch_date.isoformat()
        if experiment.launch_date
        else None,
        "historical_evaluation_end": experiment.historical_evaluation_end.isoformat()
        if experiment.historical_evaluation_end
        else None,
        "live_tracking_end": experiment.live_tracking_end.isoformat()
        if experiment.live_tracking_end
        else None,
        "source": safe_source,
        "recipe_fingerprint": experiment.source_metadata.get("recipe_fingerprint"),
        "schema_version": experiment.schema_version,
    }
    experiment_path = destination / "experiment.json"
    _write_json(experiment_path, experiment_payload)

    allocation_rows: list[dict[str, Any]] = []
    snapshot_payload: dict[str, Any] | None = None
    if snapshot is not None:
        snapshot_payload = {
            "snapshot_id": str(snapshot.snapshot_id),
            "experiment_id": str(snapshot.experiment_id),
            "package_version": snapshot.package_version,
            "code_version": snapshot.code_version,
            "objective": snapshot.objective,
            "solver": snapshot.solver,
            "solver_status": snapshot.solver_status,
            "source_data_hash": snapshot.source_data_hash,
            "assumption_recipe_hash": snapshot.assumption_recipe_hash,
            "assumptions": dict(snapshot.assumptions),
            "constraints": dict(snapshot.constraints),
            "launch_forecast": dict(snapshot.launch_forecast),
            "scenario_metadata": dict(snapshot.scenario_metadata),
            "return_policy": dict(snapshot.return_policy),
            "loss_convention": dict(snapshot.loss_convention),
            "residual_validation": dict(snapshot.residual_validation),
            "created_at": snapshot.created_at.isoformat(),
            "activated_at": snapshot.activated_at.isoformat()
            if snapshot.activated_at
            else None,
        }
        allocation_rows = [
            {
                "snapshot_id": str(snapshot.snapshot_id),
                "asset": item.asset,
                "asset_type": item.asset_type,
                "target_weight": item.target_weight,
                "launch_price": item.launch_price,
                "initial_value": item.initial_value,
                "quantity": item.quantity,
                "is_cash": item.is_cash,
            }
            for item in snapshot.allocations
        ]
    snapshot_path = destination / "optimization_snapshot.json"
    _write_json(snapshot_path, snapshot_payload)
    allocations_path = destination / "snapshot_allocations.csv"
    pd.DataFrame(allocation_rows).to_csv(allocations_path, index=False)

    portfolio_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    for state in states:
        portfolio_rows.append(
            {
                "experiment_id": str(state.experiment_id),
                "state_date": state.state_date.isoformat(),
                "nav": state.nav,
                "base_100_nav": state.base_100_nav,
                "cash_value": state.cash_value,
                "daily_return": state.daily_return,
                "cumulative_return": state.cumulative_return,
                "realized_volatility": state.realized_volatility,
                "running_peak": state.running_peak,
                "drawdown": state.drawdown,
                "maximum_drawdown": state.maximum_drawdown,
                "total_drift": state.total_drift,
                "return_interval_days": state.return_interval_days,
                "benchmark_nav": state.benchmark_nav,
                "benchmark_return": state.benchmark_return,
                "data_quality_status": state.data_quality_status.value,
                "quality_metadata": json.dumps(
                    state.quality_metadata, sort_keys=True, allow_nan=False
                ),
                "calculation_version": state.calculation_version,
                "finalized": state.finalized,
            }
        )
        for item in state.asset_states:
            asset_rows.append(
                {
                    "experiment_id": str(item.experiment_id),
                    "state_date": item.state_date.isoformat(),
                    "asset": item.asset,
                    "price": item.price,
                    "quantity": item.quantity,
                    "market_value": item.market_value,
                    "target_weight": item.target_weight,
                    "current_weight": item.current_weight,
                    "drift_percentage_points": item.drift_percentage_points,
                    "is_cash": item.is_cash,
                }
            )
    portfolio_path = destination / "daily_portfolio_states.csv"
    asset_path = destination / "daily_asset_states.csv"
    pd.DataFrame(portfolio_rows).to_csv(portfolio_path, index=False)
    pd.DataFrame(asset_rows).to_csv(asset_path, index=False)

    price_path = destination / "price_observations.csv"
    pd.DataFrame(
        [
            {
                "symbol": item.symbol,
                "observation_date": item.observation_date.isoformat(),
                "price": item.price,
                "quote_currency": item.quote_currency,
                "source": item.source,
                "retrieved_at": item.retrieved_at.isoformat(),
                "data_status": item.data_status.value,
            }
            for item in observations
        ]
    ).to_csv(price_path, index=False)

    event_path = destination / "experiment_events.csv"
    pd.DataFrame(
        [
            {
                "event_id": str(item.event_id),
                "experiment_id": str(item.experiment_id),
                "effective_date": item.effective_date.isoformat()
                if item.effective_date
                else None,
                "event_type": item.event_type,
                "event_metadata": json.dumps(
                    item.event_metadata, sort_keys=True, allow_nan=False
                ),
                "created_at": item.created_at.isoformat(),
            }
            for item in events
        ]
    ).to_csv(event_path, index=False)

    exported_files = [
        experiment_path,
        snapshot_path,
        allocations_path,
        portfolio_path,
        asset_path,
        price_path,
        event_path,
    ]
    manifest = {
        "bundle_schema_version": "1",
        "experiment_id": str(experiment_id),
        "generated_at": timestamp.isoformat().replace("+00:00", "Z"),
        "privacy": "private_by_default",
        "warning": (
            "This bundle may contain holdings, quantities, prices, and realized "
            "performance. Review before publication."
        ),
        "contains_secrets": False,
        "counts": {
            "allocations": len(allocation_rows),
            "portfolio_states": len(portfolio_rows),
            "asset_states": len(asset_rows),
            "price_observations": len(observations),
            "events": len(events),
        },
        "files": {
            path.name: {"sha256": _file_hash(path), "bytes": path.stat().st_size}
            for path in exported_files
        },
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


__all__ = ["export_experiment_bundle"]
