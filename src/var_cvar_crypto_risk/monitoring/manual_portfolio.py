"""Freeze an explicitly supplied portfolio without optimization or simulation."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

import pandas as pd

from .domain import (
    DomainValidationError,
    Experiment,
    OptimizationSnapshot,
    SnapshotAllocation,
    validate_date_boundaries,
)
from .prices import (
    NormalizedPriceData,
    fingerprint_price_slice,
    missing_symbols_on_date,
)
from .recipes import ManualMonitoringRecipe


def manual_snapshot_source_hash(
    normalized: NormalizedPriceData, experiment: Experiment, universe: tuple[str, ...]
) -> str:
    """Hash declared risk history and launch prices, excluding evaluation data."""
    columns = list(universe)
    if experiment.benchmark_symbol and experiment.benchmark_symbol not in columns:
        columns.append(experiment.benchmark_symbol)
    frame = normalized.prices
    history = (frame.index >= pd.Timestamp(experiment.training_start)) & (
        frame.index <= pd.Timestamp(experiment.training_end)
    )
    launch = frame.index == pd.Timestamp(experiment.launch_date)
    return fingerprint_price_slice(
        frame.loc[history | launch, columns],
        source=normalized.source,
        quote_currency=normalized.quote_currency,
    )


def build_manual_snapshot(
    *,
    experiment: Experiment,
    normalized: NormalizedPriceData,
    universe: tuple[str, ...] | list[str],
    recipe: ManualMonitoringRecipe,
    package_version: str,
    code_version: str,
    asset_types: Mapping[str, str] | None = None,
    activated_at: datetime | None = None,
) -> OptimizationSnapshot:
    """Convert exact manual weights to fixed quantities at a complete launch."""
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
    if normalized.source != recipe.source.provider:
        raise DomainValidationError(
            "manual portfolio source differs from the frozen price source"
        )
    if (
        normalized.quote_currency != recipe.source.quote_currency
        or experiment.base_currency != normalized.quote_currency
    ):
        raise DomainValidationError(
            "manual portfolio requires prices in its declared base currency"
        )
    assets = tuple(str(asset).strip().upper() for asset in universe)
    if len(assets) != len(set(assets)) or set(assets) != set(recipe.market_assets):
        raise DomainValidationError(
            "manual portfolio universe must exactly match its market weights"
        )
    assert experiment.optimization_as_of is not None
    assert experiment.launch_date is not None
    required = (
        (*assets, experiment.benchmark_symbol)
        if experiment.benchmark_symbol
        else assets
    )
    missing = missing_symbols_on_date(normalized, experiment.launch_date, required)
    if missing:
        raise DomainValidationError(
            "manual launch observation is incomplete; missing: " + ", ".join(missing)
        )
    if experiment.launch_date >= normalized.retrieved_at.date():
        raise DomainValidationError(
            "manual launch requires a completed UTC day, not a partial or future day"
        )
    launch_prices = normalized.prices.loc[pd.Timestamp(experiment.launch_date)]
    allocations = []
    for asset, weight in recipe.weights.items():
        is_cash = recipe.cash.enabled and asset == recipe.cash.symbol
        price = None if is_cash else float(launch_prices[asset])
        value = experiment.initial_capital * weight
        allocations.append(
            SnapshotAllocation(
                asset=asset,
                asset_type="cash"
                if is_cash
                else (asset_types or {}).get(asset, "crypto"),
                target_weight=weight,
                launch_price=price,
                initial_value=value,
                quantity=value if is_cash else value / price,
                is_cash=is_cash,
            )
        )
    snapshot = OptimizationSnapshot.create(
        experiment_id=experiment.experiment_id,
        package_version=package_version,
        code_version=code_version,
        objective="manual",
        solver="none",
        solver_status="manual_validated",
        source_data_hash=manual_snapshot_source_hash(normalized, experiment, assets),
        assumption_recipe_hash=recipe.fingerprint,
        assumptions={
            "construction_method": "manual",
            "weights": dict(recipe.weights),
            "risk_recipe": recipe.risk.to_dict(),
            "allocation_decision_date": experiment.optimization_as_of.isoformat(),
            "hindsight_warning": "A historical decision date does not prove weights were known then.",
        },
        constraints={
            "long_only": True,
            "weight_sum": 1.0,
            "cash": recipe.cash.to_dict(),
        },
        launch_forecast={
            "expected_return": None,
            "volatility": None,
            "var": None,
            "cvar": None,
            "confidence_level": recipe.risk.confidence_level,
            "horizon_days": recipe.risk.horizon_days,
            "reason": "Manual construction does not imply an optimizer forecast; see daily risk forecasts.",
        },
        scenario_metadata={
            "construction_method": "manual",
            "source_provider": normalized.source,
            "source_quote_currency": normalized.quote_currency,
            "launch_date": experiment.launch_date.isoformat(),
            "optimization_performed": False,
        },
        return_policy={
            "portfolio_method": "simple",
            "wealth_method": "simple",
            "optimization_method": None,
        },
        loss_convention={"name": "signed_loss_space"},
        residual_validation={
            "passed": True,
            "validation_type": "manual_allocation",
            "weights_normalized": False,
        },
        allocations=tuple(allocations),
    )
    return snapshot.activate(at=activated_at)
