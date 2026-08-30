"""Point-in-time optimization snapshot and leakage-control tests."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.monitoring.domain import (
    DomainValidationError,
    Experiment,
    ExperimentMode,
)
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


ACTIVATED_AT = datetime(2026, 1, 11, 12, tzinfo=timezone.utc)


def _experiment() -> Experiment:
    return Experiment.create(
        name="point in time",
        mode=ExperimentMode.HISTORICAL_OOS,
        base_currency="USD",
        initial_capital=100_000.0,
        training_start=date(2026, 1, 1),
        training_end=date(2026, 1, 10),
        optimization_as_of=date(2026, 1, 10),
        launch_date=date(2026, 1, 11),
        historical_evaluation_end=date(2026, 1, 13),
    )


def _recipe() -> OptimizationRecipe:
    return OptimizationRecipe(
        scenario=ScenarioRecipe(
            source="historical", horizon_days=1, random_seed=7
        ),
        risk=RiskMonitoringRecipe(horizon_days=1),
        source=SourceRecipe(
            provider="fixture",
            symbol_mapping={"BTC": "bitcoin", "ETH": "ethereum"},
        ),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BTC": [100, 101, 99, 102, 101, 104, 103, 105, 104, 106, 107, 108, 109],
            "ETH": [50, 51, 50, 52, 51, 53, 52, 54, 53, 55, 56, 57, 58],
        },
        index=pd.date_range("2026-01-01", periods=13, freq="D"),
        dtype=float,
    )


def _build(frame: pd.DataFrame):
    normalized = normalize_monitoring_prices(
        frame, source="fixture", retrieved_at=ACTIVATED_AT
    )
    return build_point_in_time_snapshot(
        experiment=_experiment(),
        normalized=normalized,
        universe=["BTC", "ETH"],
        recipe=_recipe(),
        package_version="1.0.0",
        code_version="abc123",
        activated_at=ACTIVATED_AT,
    )


def test_adapter_builds_activated_residual_validated_snapshot() -> None:
    snapshot = _build(_frame())
    assert snapshot.activated_at == ACTIVATED_AT
    assert snapshot.solver_status == "optimal"
    assert snapshot.residual_validation["passed"] is True
    assert sum(item.target_weight for item in snapshot.allocations) == pytest.approx(1)
    assert {item.asset for item in snapshot.allocations} == {"BTC", "ETH"}
    assert snapshot.return_policy["optimization_method"] == "simple"
    assert snapshot.loss_convention["name"] == "signed_loss_space"
    dates = snapshot.scenario_metadata["input_dates"]
    assert dates["solver_input_max_date"] == "2026-01-10"
    assert all(
        date.fromisoformat(value) <= date(2026, 1, 10)
        for key, value in dates.items()
        if key.endswith("max_date")
    )


def test_changing_post_launch_data_cannot_change_frozen_snapshot() -> None:
    first = _build(_frame())
    changed = _frame()
    changed.loc["2026-01-12":, "BTC"] = [1_000.0, 2_000.0]
    changed.loc["2026-01-12":, "ETH"] = [1.0, 2.0]
    second = _build(changed)
    assert first.source_data_hash == second.source_data_hash
    assert first.assumption_recipe_hash == second.assumption_recipe_hash
    assert first.assumptions == second.assumptions
    assert first.constraints == second.constraints
    assert first.launch_forecast == second.launch_forecast
    assert first.scenario_metadata == second.scenario_metadata
    assert first.allocations == second.allocations


def test_changing_training_data_changes_source_fingerprint() -> None:
    first = _build(_frame())
    changed = _frame()
    changed.loc["2026-01-05", "BTC"] = 150.0
    second = _build(changed)
    assert first.source_data_hash != second.source_data_hash


def test_training_gap_is_rejected_instead_of_mislabeled_as_daily() -> None:
    frame = _frame().drop(pd.Timestamp("2026-01-05"))
    with pytest.raises(DomainValidationError, match="consecutive crypto UTC days"):
        _build(frame)


def test_incomplete_training_date_is_rejected_without_forward_fill() -> None:
    frame = _frame()
    frame.loc["2026-01-05", "ETH"] = np.nan
    with pytest.raises(DomainValidationError, match="incomplete observations"):
        _build(frame)


def test_actual_source_must_match_frozen_source_recipe() -> None:
    normalized = normalize_monitoring_prices(
        _frame(), source="fallback-provider", retrieved_at=ACTIVATED_AT
    )
    with pytest.raises(DomainValidationError, match="actual source"):
        build_point_in_time_snapshot(
            experiment=_experiment(),
            normalized=normalized,
            universe=["BTC", "ETH"],
            recipe=_recipe(),
            package_version="1.0.0",
            code_version="abc123",
            activated_at=ACTIVATED_AT,
        )
