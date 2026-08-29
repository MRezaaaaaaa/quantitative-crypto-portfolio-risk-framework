"""Tests for immutable, portable monitoring recipes."""

from __future__ import annotations

import pytest

from var_cvar_crypto_risk.monitoring.domain import DomainValidationError
from var_cvar_crypto_risk.monitoring.recipes import (
    AssumptionRecipe,
    CashPolicy,
    OptimizationRecipe,
    OptimizerRecipe,
    RiskMonitoringRecipe,
    ScenarioRecipe,
    SourceRecipe,
    optimization_recipe_from_dict,
)


def test_recipe_hash_is_order_independent_and_reconstructs_assumptions() -> None:
    first = OptimizationRecipe(
        assumptions=AssumptionRecipe(manual_views={"btc": 0.01, "eth": 0.02}),
        source=SourceRecipe(
            provider="fixture",
            symbol_mapping={"btc": "bitcoin", "eth": "ethereum"},
        ),
    )
    second = OptimizationRecipe(
        assumptions=AssumptionRecipe(manual_views={"ETH": 0.02, "BTC": 0.01}),
        source=SourceRecipe(
            provider="fixture",
            symbol_mapping={"ETH": "ethereum", "BTC": "bitcoin"},
        ),
    )
    assert first.fingerprint == second.fingerprint
    config = first.assumptions.to_assumption_config()
    assert config.manual_views == {"BTC": 0.01, "ETH": 0.02}


def test_recipe_mapping_is_deeply_non_assignable() -> None:
    recipe = AssumptionRecipe(manual_views={"BTC": 0.01})
    with pytest.raises(TypeError):
        recipe.manual_views["BTC"] = 0.5  # type: ignore[index]


def test_persisted_recipe_round_trip_revalidates_identical_fingerprint() -> None:
    recipe = OptimizationRecipe(
        source=SourceRecipe(
            provider="fixture",
            symbol_mapping={"BTC": "bitcoin", "ETH": "ethereum"},
            refreshable=True,
            metadata={
                "fallback_provider": "yfinance",
                "fallback_symbol_mapping": {"BTC": "BTC-USD", "ETH": "ETH-USD"},
            },
        )
    )
    restored = optimization_recipe_from_dict(recipe.to_dict())
    assert restored.to_dict() == recipe.to_dict()
    assert restored.fingerprint == recipe.fingerprint


@pytest.mark.parametrize(
    "metadata",
    [
        {"api_key": "secret"},
        {"nested": {"access_token": "secret"}},
        {"database_url": "redacted"},
        {"endpoint": "postgresql://" + "user:pass@example/db"},
    ],
)
def test_source_recipe_rejects_secret_bearing_metadata(metadata) -> None:
    with pytest.raises(
        DomainValidationError, match="not allowed|credential-bearing"
    ):
        SourceRecipe(provider="fixture", metadata=metadata)


def test_cash_policy_uses_calendar_day_compounding() -> None:
    policy = CashPolicy(enabled=True, mode="annual_rate", annual_rate=0.10)
    assert policy.horizon_return(0) == 0.0
    assert policy.horizon_return(365) == pytest.approx(0.10)
    assert policy.horizon_return(30) == pytest.approx(1.10 ** (30 / 365) - 1)


def test_zero_cash_policy_rejects_nonzero_rate() -> None:
    with pytest.raises(DomainValidationError, match="annual_rate=0"):
        CashPolicy(enabled=True, mode="zero", annual_rate=0.01)


def test_recipe_requires_matching_risk_and_optimization_horizons() -> None:
    with pytest.raises(DomainValidationError, match="horizons must match"):
        OptimizationRecipe(
            scenario=ScenarioRecipe(horizon_days=7),
            risk=RiskMonitoringRecipe(horizon_days=1),
        )


def test_objective_specific_constraints_are_required() -> None:
    with pytest.raises(DomainValidationError, match="cvar_limit"):
        OptimizerRecipe(objective="max_return_cvar")
    with pytest.raises(DomainValidationError, match="target_return"):
        OptimizerRecipe(objective="min_cvar_target_return")
