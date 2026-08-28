"""Point-in-time adapter from frozen recipes to the existing optimizer.

No financial formula is implemented here.  The adapter controls data dates,
reconstructs existing assumption/scenario objects, calls existing optimizers,
checks their independent residual result, and freezes provenance.
"""

from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..optimization import (
    add_cash_asset,
    build_optimization_scenarios,
    maximize_return_with_cvar_constraint,
    maximize_sharpe_ratio,
    minimize_cvar,
    minimize_cvar_for_target_return,
)
from ..return_conventions import resolve_return_policy
from ..risk_conventions import LOSS_SPACE_CONVENTION
from .domain import (
    DomainValidationError,
    Experiment,
    OptimizationSnapshot,
    SnapshotAllocation,
    validate_date_boundaries,
)
from .hashing import sha256_fingerprint
from .prices import (
    NormalizedPriceData,
    fingerprint_price_slice,
    resolve_launch_prices,
)
from .recipes import OptimizationRecipe


def _json_safe(value: Any) -> Any:
    """Replace non-finite diagnostics with ``None`` for portable JSON."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, pd.Series):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _scenario_fingerprint(scenarios: pd.DataFrame) -> str:
    return sha256_fingerprint(
        {
            "columns": [str(column) for column in scenarios.columns],
            "values": [
                [float(item) for item in row]
                for row in scenarios.to_numpy(dtype=float).tolist()
            ],
        }
    )


def _strict_daily_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily Simple returns without pandas' implicit fill behavior."""
    if prices.isna().any().any():
        missing = {
            timestamp.date().isoformat(): [
                str(symbol) for symbol in prices.columns if pd.isna(row[symbol])
            ]
            for timestamp, row in prices.iterrows()
            if row.isna().any()
        }
        raise DomainValidationError(
            "optimization training prices contain incomplete observations: "
            + str(missing)
        )
    gaps = prices.index.to_series().diff().dropna().dt.days
    if (gaps != 1).any():
        bad_dates = [
            index.date().isoformat()
            for index in gaps.index[gaps != 1]
        ]
        raise DomainValidationError(
            "optimization training prices must be consecutive crypto UTC days; "
            "gaps end on: " + ", ".join(bad_dates)
        )
    returns = prices.divide(prices.shift(1)).subtract(1.0).iloc[1:]
    if returns.empty or returns.isna().any().any():
        raise DomainValidationError(
            "optimization requires at least two complete daily price observations"
        )
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise DomainValidationError("optimization returns contain non-finite values")
    return returns


def _run_optimizer(
    scenarios: pd.DataFrame,
    expected_returns: pd.Series,
    recipe: OptimizationRecipe,
) -> dict:
    config = recipe.optimizer
    common = {
        "confidence_level": config.confidence_level,
        "long_only": config.long_only,
        "min_weight": config.min_weight,
        "max_weight": config.max_weight,
        "include_cash": False,
        "solver": config.solver,
    }
    if config.objective == "min_cvar":
        return minimize_cvar(scenarios, **common)
    if config.objective == "max_return_cvar":
        assert config.cvar_limit is not None
        return maximize_return_with_cvar_constraint(
            scenarios,
            expected_returns=expected_returns,
            cvar_limit=config.cvar_limit,
            **common,
        )
    if config.objective == "min_cvar_target_return":
        assert config.target_return is not None
        return minimize_cvar_for_target_return(
            scenarios,
            expected_returns=expected_returns,
            target_return=config.target_return,
            **common,
        )
    if config.objective == "max_sharpe":
        risk_free_horizon = (1.0 + config.risk_free_rate) ** (
            recipe.scenario.horizon_days / 365.0
        ) - 1.0
        return maximize_sharpe_ratio(
            scenarios,
            expected_returns=expected_returns,
            risk_free_rate=risk_free_horizon,
            n_grid=config.sharpe_grid_points,
            **common,
        )
    raise DomainValidationError(
        f"unsupported optimizer objective {config.objective!r}"
    )


def build_point_in_time_snapshot(
    *,
    experiment: Experiment,
    normalized: NormalizedPriceData,
    universe: tuple[str, ...] | list[str],
    recipe: OptimizationRecipe,
    package_version: str,
    code_version: str,
    asset_types: Mapping[str, str] | None = None,
    activated_at: datetime | None = None,
) -> OptimizationSnapshot:
    """Build and activate an immutable snapshot without post-cutoff leakage."""
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
    assert experiment.training_start is not None
    assert experiment.training_end is not None
    assert experiment.optimization_as_of is not None
    assert experiment.launch_date is not None
    if normalized.source != recipe.source.provider:
        raise DomainValidationError(
            "normalized actual source does not match the frozen source recipe"
        )
    if normalized.quote_currency != recipe.source.quote_currency:
        raise DomainValidationError(
            "normalized quote currency does not match the frozen source recipe"
        )

    assets = tuple(dict.fromkeys(str(item).strip().upper() for item in universe))
    if not assets or any(not asset for asset in assets):
        raise DomainValidationError("a non-empty frozen universe is required")
    missing_columns = [asset for asset in assets if asset not in normalized.prices]
    if missing_columns:
        raise DomainValidationError(
            "optimization source is missing frozen assets: "
            + ", ".join(missing_columns)
        )
    if recipe.source.symbol_mapping:
        missing_mapping = [
            asset for asset in assets if asset not in recipe.source.symbol_mapping
        ]
        if missing_mapping:
            raise DomainValidationError(
                "source recipe lacks mappings for: " + ", ".join(missing_mapping)
            )

    training_prices = normalized.prices.loc[
        (normalized.prices.index >= pd.Timestamp(experiment.training_start))
        & (normalized.prices.index <= pd.Timestamp(experiment.training_end)),
        list(assets),
    ]
    if training_prices.empty:
        raise DomainValidationError("point-in-time training price slice is empty")
    actual_max = training_prices.index.max().date()
    if actual_max > experiment.optimization_as_of:
        raise DomainValidationError("future prices entered the optimization slice")
    daily_returns = _strict_daily_simple_returns(training_prices)
    input_max_date = daily_returns.index.max().date()
    if input_max_date > experiment.optimization_as_of:
        raise DomainValidationError("future returns entered an optimizer input")

    assumptions = recipe.assumptions.to_assumption_config()
    robust_daily_mean = assumptions.final_expected_returns(daily_returns)
    robust_covariance = assumptions.covariance(daily_returns)
    scenarios = build_optimization_scenarios(
        asset_returns=daily_returns,
        source=recipe.scenario.source,
        n_scenarios=recipe.scenario.n_scenarios,
        horizon_days=recipe.scenario.horizon_days,
        student_t_df=recipe.scenario.student_t_df,
        random_seed=recipe.scenario.random_seed,
        mean_vector=robust_daily_mean,
        covariance_matrix=robust_covariance,
        return_method="simple",
        covariance_policy=recipe.scenario.covariance_policy,
    )
    expected_returns = assumptions.final_expected_returns(scenarios)
    if recipe.cash.enabled:
        cash_return = recipe.cash.horizon_return(recipe.scenario.horizon_days)
        scenarios = add_cash_asset(
            scenarios,
            cash_return=cash_return,
            cash_name=recipe.cash.symbol,
        )
        expected_returns = pd.concat(
            [expected_returns, pd.Series({recipe.cash.symbol: cash_return})]
        )
    result = _run_optimizer(scenarios, expected_returns, recipe)
    status = str(result.get("status", "unknown"))
    validation = result.get("constraint_validation", {})
    if status not in {"optimal", "optimal_inaccurate"}:
        raise DomainValidationError(
            f"optimizer result cannot be activated: status={status!r}"
        )
    if status == "optimal_inaccurate" and not recipe.optimizer.accept_optimal_inaccurate:
        raise DomainValidationError(
            "optimal_inaccurate requires explicit persisted acknowledgement"
        )
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        raise DomainValidationError(
            "optimizer result failed independent residual validation"
        )
    weights = result.get("weights")
    if not isinstance(weights, pd.Series):
        raise DomainValidationError("optimizer result does not contain weights")
    weights = weights.reindex(scenarios.columns)
    if weights.isna().any() or not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise DomainValidationError("optimizer weights are incomplete or non-finite")

    launch = resolve_launch_prices(
        normalized,
        universe=assets,
        optimization_as_of=experiment.optimization_as_of,
        requested_launch_date=experiment.launch_date,
        benchmark_symbol=experiment.benchmark_symbol,
    )
    type_mapping = {
        str(key).strip().upper(): str(value).strip().lower()
        for key, value in (asset_types or {}).items()
    }
    allocations: list[SnapshotAllocation] = []
    for asset, weight_value in weights.items():
        weight = float(weight_value)
        initial_value = experiment.initial_capital * weight
        is_cash = asset == recipe.cash.symbol and recipe.cash.enabled
        if is_cash:
            allocations.append(
                SnapshotAllocation(
                    asset=asset,
                    asset_type="cash",
                    target_weight=weight,
                    launch_price=None,
                    initial_value=initial_value,
                    quantity=initial_value,
                    is_cash=True,
                )
            )
            continue
        launch_price = launch.prices[asset]
        allocations.append(
            SnapshotAllocation(
                asset=asset,
                asset_type=type_mapping.get(asset, "crypto"),
                target_weight=weight,
                launch_price=launch_price,
                initial_value=initial_value,
                quantity=initial_value / launch_price,
            )
        )

    input_dates = {
        "expected_returns_max_date": input_max_date.isoformat(),
        "covariance_max_date": input_max_date.isoformat(),
        "scenario_source_max_date": input_max_date.isoformat(),
        "solver_input_max_date": input_max_date.isoformat(),
        "optimization_as_of": experiment.optimization_as_of.isoformat(),
    }
    if any(
        date.fromisoformat(value) > experiment.optimization_as_of
        for key, value in input_dates.items()
        if key.endswith("max_date")
    ):
        raise DomainValidationError("future data entered snapshot provenance")

    source_hash = fingerprint_price_slice(
        training_prices,
        source=normalized.source,
        quote_currency=normalized.quote_currency,
    )
    return_policy = resolve_return_policy("automatic", "simple")
    snapshot = OptimizationSnapshot.create(
        experiment_id=experiment.experiment_id,
        package_version=package_version,
        code_version=code_version,
        objective=recipe.optimizer.objective,
        solver=str(result.get("solver", recipe.optimizer.solver or "auto")),
        solver_status=status,
        source_data_hash=source_hash,
        assumption_recipe_hash=recipe.fingerprint,
        assumptions={
            "recipe": recipe.assumptions.to_dict(),
            "daily_expected_returns": _json_safe(robust_daily_mean),
            "horizon_expected_returns": _json_safe(expected_returns),
            "input_dates": input_dates,
        },
        constraints={
            "optimizer": recipe.optimizer.to_dict(),
            "cash": recipe.cash.to_dict(),
        },
        launch_forecast={
            "expected_return": _json_safe(result.get("expected_return")),
            "volatility": _json_safe(result.get("volatility")),
            "var": _json_safe(result.get("VaR")),
            "cvar": _json_safe(result.get("CVaR")),
            "confidence_level": recipe.optimizer.confidence_level,
            "horizon_days": recipe.scenario.horizon_days,
        },
        scenario_metadata={
            **recipe.scenario.to_dict(),
            "scenario_count": len(scenarios),
            "scenario_hash": _scenario_fingerprint(scenarios),
            "source_provider": normalized.source,
            "source_quote_currency": normalized.quote_currency,
            "training_start": training_prices.index.min().date().isoformat(),
            "training_end": training_prices.index.max().date().isoformat(),
            "launch_date": launch.launch_date.isoformat(),
            "input_dates": input_dates,
        },
        return_policy={
            "handling_mode": return_policy.handling_mode,
            "diagnostic_method": return_policy.diagnostic_method,
            "portfolio_method": return_policy.portfolio_method,
            "wealth_method": return_policy.wealth_method,
            "scenario_method": return_policy.scenario_method,
            "optimization_method": return_policy.optimization_method,
        },
        loss_convention={
            "name": "signed_loss_space",
            "description": LOSS_SPACE_CONVENTION,
        },
        residual_validation=_json_safe(validation),
        allocations=tuple(allocations),
    )
    return snapshot.activate(at=activated_at)


__all__ = ["build_point_in_time_snapshot"]
